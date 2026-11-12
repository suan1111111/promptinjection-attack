#!/usr/bin/env python3
"""
Whisper 语音转文本服务模块
支持中英文语音识别
"""

import os
import tempfile
import logging
from pathlib import Path

from flask import Blueprint, request, jsonify

# 创建 Blueprint
whisper_bp = Blueprint('whisper', __name__, url_prefix='/api/whisper')

logger = logging.getLogger(__name__)

# =============================================================================
# 配置
# =============================================================================

def get_config():
    """获取配置"""
    from config import WHISPER_MODEL_SIZE, WHISPER_UPLOAD_FOLDER, WHISPER_MODELS_DIR
    return {
        'model_size': WHISPER_MODEL_SIZE,
        'upload_folder': str(WHISPER_UPLOAD_FOLDER),
        'model_dir': str(WHISPER_MODELS_DIR),
    }

# 全局变量
whisper_model = None


def ensure_ffmpeg_available():
    """Expose the project-local FFmpeg binary to Whisper subprocesses."""
    import imageio_ffmpeg
    import shutil
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    # openai-whisper invokes the literal name "ffmpeg"; imageio uses a versioned name.
    if ffmpeg_exe.name.lower() != ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'):
        tool_dir = Path(__file__).resolve().parents[2] / 'tools' / 'ffmpeg'
        tool_dir.mkdir(parents=True, exist_ok=True)
        named_exe = tool_dir / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
        if not named_exe.exists():
            shutil.copy2(ffmpeg_exe, named_exe)
        ffmpeg_exe = named_exe
    ffmpeg_dir = str(ffmpeg_exe.parent)
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    if ffmpeg_dir not in path_entries:
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    return str(ffmpeg_exe)


# =============================================================================
# 模型加载
# =============================================================================

def load_whisper_model():
    """加载 Whisper 模型"""
    global whisper_model
    
    if whisper_model is None:
        import torch
        config = get_config()
        model_size = config['model_size']
        
        logger.info(f"加载 Whisper 模型: {model_size}")
        try:
            # 保存并重置全局 dtype（避免被 OCR 的 bfloat16 影响）
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            
            import whisper
            whisper_model = whisper.load_model(
                model_size, download_root=config['model_dir'],
            ).float()

            # 恢复原始 dtype
            torch.set_default_dtype(original_dtype)
            
            logger.info("Whisper 模型加载成功 (Float32)")
        except Exception as e:
            logger.error(f"Whisper 模型加载失败: {e}")
            raise
        finally:
            if 'original_dtype' in locals():
                torch.set_default_dtype(original_dtype)
    
    return whisper_model

# =============================================================================
# API 路由
# =============================================================================

@whisper_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        config = get_config()
        return jsonify({
            'status': 'ok',
            'service': 'Whisper',
            'model': config['model_size'],
            'model_loaded': whisper_model is not None,
            'model_dir': config['model_dir'],
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@whisper_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    语音转文本接口
    
    请求参数:
    - audio: 音频文件 (multipart/form-data)
    - language: 语言代码 (可选，默认 auto 自动检测)
    
    返回:
    {
        "text": "识别的文本",
        "language": "检测到的语言",
        "segments": [...]  # 详细分段信息
    }
    """
    try:
        ensure_ffmpeg_available()
        # 检查是否有文件
        if 'audio' not in request.files:
            return jsonify({'error': '未找到音频文件'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 获取语言参数
        language = request.form.get('language', 'auto')
        if language == 'auto':
            language = None  # Whisper 自动检测
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            # 加载模型
            model = load_whisper_model()
            
            logger.info(f"开始转录: {audio_file.filename}, 语言: {language or 'auto'}")
            
            # 执行转录（显式设置 float32 避免与 OCR 的 bfloat16 冲突）
            import torch
            with torch.amp.autocast('cuda', enabled=False):
                # 确保模型在 float32 模式
                model = model.float()
                result = model.transcribe(
                    tmp_path,
                    language=language,
                    task='transcribe',
                    fp16=False,  # 禁用 fp16
                    verbose=False
                )
            
            # 提取结果
            text = result['text'].strip()
            detected_language = result.get('language', 'unknown')
            
            logger.info(f"转录完成: {len(text)} 字符, 语言: {detected_language}")
            
            # 返回结果
            response = {
                'text': text,
                'language': detected_language,
            }
            
            # 详细分段信息
            if 'segments' in result:
                response['segments'] = [
                    {
                        'start': seg.get('start'),
                        'end': seg.get('end'),
                        'text': seg.get('text', '').strip()
                    }
                    for seg in result['segments']
                ]
            
            return jsonify(response)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
    
    except Exception as e:
        logger.error(f"转录失败: {e}", exc_info=True)
        return jsonify({'error': f'转录失败: {str(e)}'}), 500

@whisper_bp.route('/models', methods=['GET'])
def list_models():
    """列出可用的 Whisper 模型"""
    config = get_config()
    models = ['tiny', 'base', 'small', 'medium', 'large']
    return jsonify({
        'available_models': models,
        'current_model': config['model_size']
    })

@whisper_bp.route('/languages', methods=['GET'])
def list_languages():
    """列出支持的语言"""
    languages = {
        'auto': '自动检测',
        'zh': '中文',
        'en': '英文',
        'ja': '日语',
        'ko': '韩语',
        'es': '西班牙语',
        'fr': '法语',
        'de': '德语',
        'ru': '俄语',
    }
    return jsonify(languages)

# =============================================================================
# 模块初始化
# =============================================================================

def init_whisper_service(preload_model: bool = False):
    """初始化 Whisper 服务"""
    logger.info("初始化 Whisper 服务模块...")
    
    # 确保目录存在
    config = get_config()
    os.makedirs(config['upload_folder'], exist_ok=True)
    
    if preload_model:
        try:
            load_whisper_model()
            logger.info("✓ Whisper 模型预加载完成")
        except Exception as e:
            logger.warning(f"Whisper 模型预加载失败: {e}")
