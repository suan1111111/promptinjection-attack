import base64
import io

def normalize_messages(messages):
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            if any(part.get('type') == 'image_url' for part in content if isinstance(part, dict)):
                raise ValueError('Images require the local vision model, not the text model')
            content = "\n".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        normalized.append({"role": message.get("role", "user"), "content": str(content)})
    return normalized


def vision_messages(messages):
    from PIL import Image, ImageOps
    normalized = []
    for message in messages:
        content = message.get('content', '')
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get('type') == 'text':
                    parts.append({'type': 'text', 'text': part.get('text', '')})
                elif part.get('type') == 'image_url':
                    value = part['image_url']
                    url = value.get('url', '') if isinstance(value, dict) else value
                    if not url.startswith('data:image/') or ';base64,' not in url:
                        raise ValueError('Local vision accepts base64 image data URLs only')
                    raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        decoded = ImageOps.exif_transpose(image).convert('RGB')
                    parts.append({'type': 'image', 'image': decoded})
                else:
                    raise ValueError('Unsupported multimodal content type')
            content = parts
        normalized.append({'role': message.get('role', 'user'), 'content': content})
    return normalized


