"""Optional upload entry point for existing Flask backend."""
from flask import Blueprint, jsonify, request
from PIL import UnidentifiedImageError, Image
from .frequency_features import FrequencyFeatureStore, GROUPS

frequency_bp = Blueprint("frequency", __name__, url_prefix="/api/image-features")


@frequency_bp.post("/frequency")
def frequency():
    upload = request.files.get("file")
    if upload is None:
        return jsonify(error="A file field is required"), 400
    groups = request.form.get("groups", ",".join(GROUPS)).split(",")
    groups = [group.strip() for group in groups]
    try:
        result = FrequencyFeatureStore().extract_bytes(upload.read(), upload.filename, groups)
        return jsonify(result), 422 if result["failed_groups"] else 200
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        return jsonify(error=str(error)), 400
