"""Print SnapGuard's five action counts and matched-line coverage."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import uuid
from services.text_rule_features import TextRuleStore


def request_ocr(image, base_url, timeout, function='free_ocr', resolution='tiny'):
    boundary = uuid.uuid4().hex
    data = Path(image).read_bytes()
    parts = []
    for key, value in [('function', function), ('resolution', resolution)]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="image.png"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode() + data + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    request = urllib.request.Request(base_url.rstrip('/') + '/api/ocr/process/stream', data=b''.join(parts), headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    chunks = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            event = json.loads(line[5:].strip())
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('type') == 'content':
                chunks.append(event['text'])
            if event.get('type') == 'done':
                return ''.join(chunks)
    raise RuntimeError('OCR stream ended without completion; no zero-valued result saved')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--ocr-text', help='Existing UTF-8 OCR text file; no model inference')
    parser.add_argument('--store')
    parser.add_argument('--backend', default='http://127.0.0.1:5000')
    parser.add_argument('--timeout', type=float, default=1200)
    args = parser.parse_args()
    try:
        text = Path(args.ocr_text).read_text(encoding='utf-8-sig') if args.ocr_text else None
        result = TextRuleStore(args.store).extract(args.image, text=text, ocr=lambda image: request_ocr(image, args.backend, args.timeout))
        for label, key in [('点击诱导命中行数', 'position_click_count'), ('密码输入命中行数', 'password_input_count'), ('链接邀请命中行数', 'link_invitation_count'), ('控制指令命中行数', 'control_instruction_count'), ('凭据请求命中行数', 'credential_request_count'), ('命中文字行占比', 'matched_text_ratio')]:
            print(f'{label}: {result[key]:.12g}')
        return 0
    except Exception as error:
        print(f'未完成: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
