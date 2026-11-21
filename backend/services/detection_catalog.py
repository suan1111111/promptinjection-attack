"""Versioned metadata contract for the detection tree; no model imports or execution."""
from copy import deepcopy


def _operation(identifier, label, dependencies, outputs, source, note=""):
    return {
        "id": identifier, "version": 1, "label": label, "kind": "compute",
        "dependencies": dependencies,
        "inputs": ["sample.image"] if not dependencies else dependencies,
        "outputs": [{"name": name, "type": kind, "nullable": nullable}
                    for name, kind, nullable in outputs],
        "implementation": {"source": source, "status": "adapter_available", "note": note},
        "executable": True,
        "parameters": {},
    }


OPERATIONS = [
    _operation("ocr.structured", "结构化 OCR", [], [
        ("text", "string", False), ("boxes", "array", False),
        ("image_size", "array", False)],
        "services/ocr_numeric_features.py",
        "目标契约：boxes 每项含 box=[x1,y1,x2,y2]、text、confidence；"
        "坐标对应 EXIF 校正后的原图像素；支持传入 grounding/TSV 或服务端 OCR 回调，跨请求统一缓存待实现。"),
    _operation("image.gray", "灰度图", [], [("gray", "artifact", False)],
        "services/gradient_features.py",
        "EXIF 校正、透明区域合成白底、Pillow L、数值范围 0..1；会话内共享。"),
    _operation("text.rules", "文字规则", ["ocr.structured"], [
        *[(name + "_count", "integer", False) for name in
          ("position_click", "password_input", "link_invitation", "control_instruction", "credential_request")],
        ("matched_text_ratio", "number", False)],
        "services/text_rule_features.py", "调用 extract_text_rules，使用会话内 OCR 输入。"),
    _operation("text.geometry", "文字位置与面积", ["ocr.structured"], [
        ("text_area_ratio", "number", False),
        ("relative_box_height_mean", "number", True),
        ("relative_box_height_max", "number", True),
        ("edge_text_box_ratio", "number", False)],
        "services/ocr_numeric_features.py", "调用 geometry_only，不执行对比度、梯度或熵。"),
    _operation("text.contrast", "文字背景对比度", ["ocr.structured", "image.gray"], [
        ("box_background_contrast_mean", "number", True)],
        "services/ocr_numeric_features.py", "调用 contrast_only；无可用背景区域时结果为空。"),
    _operation("image.gradient", "梯度统计与区域", ["image.gray"], [
        ("max_window_gradient", "number", False),
        ("high_gradient_window_ratio", "number", False),
        ("high_gradient_area_ratio", "number", False),
        ("windows", "artifact", False)],
        "services/gradient_features.py", "调用 compute_gradient；使用会话内灰度结果，保留窗口证据。"),
    _operation("text.gradient_overlap", "文字与梯度重合", ["ocr.structured", "image.gradient"], [
        ("gradient_text_coverage", "number", True),
        ("gradient_response_in_text_ratio", "number", True),
        ("gradient_iou", "number", True)],
        "services/ocr_numeric_features.py", "已有 overlap；分母为零时返回空值，不作为未匹配。"),
]


def validate_catalog(operations):
    """Reject duplicate IDs, dangling dependencies and cycles before serving metadata."""
    by_id = {item["id"]: item for item in operations}
    if len(by_id) != len(operations):
        raise ValueError("Duplicate operation ID")
    visiting, visited = set(), set()

    def visit(identifier):
        if identifier not in by_id:
            raise ValueError("Unknown dependency: " + identifier)
        if identifier in visiting:
            raise ValueError("Dependency cycle: " + identifier)
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in by_id[identifier]["dependencies"]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in by_id:
        visit(identifier)


validate_catalog(OPERATIONS)


def get_catalog():
    return {
        "schema_version": 1, "catalog_version": "tree-5",
        "operations": deepcopy(OPERATIONS), "patterns": [],
        "execution_available": True,
        "execution": {"endpoint": "/api/detection/compute", "cache_scope": "request",
                      "sequence_endpoint": "/api/detection/compute-sequence",
                      "validation_endpoint": "/api/detection/plans/validate",
                      "demo_tree_endpoint": "/api/detection/trees/demo",
                      "tree_validation_endpoint": "/api/detection/trees/validate",
                      "tree_runs_endpoint": "/api/detection/tree-runs",
                      "ocr_input": "optional; local DeepSeek grounding is used when OCR service is enabled"},
        "cache_identity_fields": ["image_content_hash", "operation_id", "operation_version",
                                  "parameters", "dependency_result_ids"],
    }


def get_operation(identifier):
    return next((deepcopy(item) for item in OPERATIONS if item["id"] == identifier), None)
