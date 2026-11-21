"""Discovery and synchronous calculation endpoints for the detection tree."""
import json
import sqlite3
from flask import Blueprint, jsonify, request, current_app
from .detection_catalog import get_catalog, get_operation

detection_bp = Blueprint("detection", __name__, url_prefix="/api/detection")


def _tree_error(error, fallback_code):
    from .detection_tree import TreeDependencyError
    result = {'code': fallback_code, 'message': str(error)}
    if isinstance(error, TreeDependencyError):
        result.update(code='invalid_dependency_order', issues=error.issues,
                      execution_order=error.execution_order)
    return result


@detection_bp.get('/trees/demo')
def get_demo_tree():
    from .detection_tree import demo_tree
    return jsonify(demo_tree())


@detection_bp.post('/trees/validate')
def check_tree():
    from .detection_tree import validate_tree
    try:
        order = validate_tree(request.get_json(silent=True))
        return jsonify(valid=True, execution_order=order, schema_version=1)
    except ValueError as error:
        return jsonify(valid=False, error=_tree_error(error, 'invalid_tree')), 400


@detection_bp.post('/trees/nodes')
def add_tree_node():
    from .detection_tree_edit import add_node
    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        result = add_node(payload.get('tree'), payload.get('parent_id'),
                          payload.get('node'), payload.get('index'))
        return jsonify(result), 200
    except ValueError as error:
        return jsonify(error={'code': 'invalid_node_addition', 'message': str(error)}), 400


@detection_bp.post('/trees/nodes/delete')
def delete_tree_node():
    from .detection_tree_edit import delete_node
    payload = request.get_json(silent=True)
    try:
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        return jsonify(delete_node(payload.get('tree'), payload.get('node_id'), payload.get('mode')))
    except ValueError as error:
        return jsonify(error={'code': 'invalid_node_deletion', 'message': str(error)}), 400


@detection_bp.route('/tree-configs', methods=['GET', 'POST'])
@detection_bp.route('/tree-configs/<config_id>', methods=['GET', 'PUT'])
def tree_configs(config_id=None):
    from .detection_tree_store import TreeStore, VersionConflict
    store = TreeStore(current_app.config.get('DETECTION_TREE_STORE'))
    try:
        if request.method == 'GET':
            return jsonify(store.read(config_id) if config_id else {'items': store.list(), 'schema_version': 1})
        result = store.save(request.get_json(silent=True), config_id)
        return jsonify(result), 201 if config_id is None else 200
    except VersionConflict as error:
        return jsonify(error={'code': 'version_conflict', 'message': str(error),
                              'current_version': error.current_version}), 409
    except FileNotFoundError:
        return jsonify(error={'code': 'tree_config_not_found', 'message': 'Tree configuration not found'}), 404
    except ValueError as error:
        return jsonify(error=_tree_error(error, 'invalid_tree_config')), 400
    except (OSError, sqlite3.Error):
        current_app.logger.exception('Tree configuration storage failed')
        return jsonify(error={'code': 'tree_config_storage_failed', 'message': 'Tree configuration storage failed'}), 500


@detection_bp.post('/tree-runs')
def tree_run():
    from PIL import Image, UnidentifiedImageError
    from .detection_tree import execute_tree, validate_tree
    from .detection_runs import RunStore
    from .detection_compute import json_value
    try:
        tree = json.loads(request.form.get('tree', 'null'))
        validate_tree(tree)
        upload = request.files.get('file')
        if upload is None:
            raise ValueError('file is required')
    except ValueError as error:
        return jsonify(error=_tree_error(error, 'invalid_request')), 400
    try:
        report = execute_tree(upload.read(), tree, request.form.get('ocr_input'),
                              request.form.get('format', 'grounding'),
                              current_app.config.get('DETECTION_OCR_PROVIDER'))
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        return jsonify(error={'code': 'computation_input_error', 'message': str(error)}), 422
    try:
        RunStore(current_app.config.get('DETECTION_RUN_STORE')).save(report)
    except OSError:
        current_app.logger.exception('Cannot persist detection run')
        return jsonify(error={'code': 'run_save_failed', 'message': 'Run could not be saved'}), 500
    return jsonify(json_value(report)), 201


@detection_bp.get('/tree-runs/<run_id>')
def get_tree_run(run_id):
    from .detection_runs import RunStore
    try:
        return jsonify(RunStore(current_app.config.get('DETECTION_RUN_STORE')).read(run_id))
    except (ValueError, FileNotFoundError):
        return jsonify(error={'code': 'run_not_found', 'message': 'Run not found'}), 404


@detection_bp.post('/plans/validate')
def validate_plan():
    from .detection_plan import validate_order
    payload = request.get_json(silent=True)
    try:
        order = validate_order(payload.get('operations') if isinstance(payload, dict) else None)
        return jsonify(schema_version=1, valid=True, operations=order)
    except ValueError as error:
        return jsonify(valid=False, error={'code': 'invalid_order', 'message': str(error)}), 400


@detection_bp.post('/compute-sequence')
def compute_sequence():
    from PIL import Image, UnidentifiedImageError
    from .detection_plan import execute_order, validate_order
    from .detection_compute import json_value
    upload = request.files.get('file')
    try:
        order = validate_order(json.loads(request.form.get('operations', 'null')))
        if upload is None:
            raise ValueError('file is required')
    except ValueError as error:
        return jsonify(error={'code': 'invalid_request', 'message': str(error)}), 400
    try:
        result = execute_order(upload.read(), order, request.form.get('ocr_input'),
                               request.form.get('format', 'grounding'),
                               current_app.config.get('DETECTION_OCR_PROVIDER'))
        return jsonify(json_value(result))
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        return jsonify(error={'code': 'computation_input_error', 'message': str(error)}), 422


@detection_bp.post('/compute')
def compute():
    from PIL import UnidentifiedImageError, Image
    from .detection_compute import ComputationSession, json_value
    upload = request.files.get('file')
    identifier = request.form.get('operation_id', '')
    if upload is None or get_operation(identifier) is None:
        return jsonify(error={'code': 'invalid_request', 'message': 'file and a known operation_id are required'}), 400
    try:
        session = ComputationSession(upload.read(), request.form.get('ocr_input'),
                                     request.form.get('format', 'grounding'),
                                     current_app.config.get('DETECTION_OCR_PROVIDER'))
        result = session.compute(identifier)
        return jsonify(schema_version=1, operation_id=identifier, status='completed',
                       values=json_value(result), computations=session.events,
                       cache_scope='request')
    except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        return jsonify(error={'code': 'computation_input_error', 'message': str(error)}), 422
    except Exception:
        current_app.logger.exception('Detection computation failed')
        return jsonify(error={'code': 'computation_failed', 'message': 'Computation failed; see server log'}), 500


@detection_bp.get("/capabilities")
def capabilities():
    return jsonify(get_catalog())


@detection_bp.get("/capabilities/<operation_id>")
def capability(operation_id):
    operation = get_operation(operation_id)
    if operation is None:
        return jsonify(error={"code": "operation_not_found", "message": "Unknown operation",
                              "operation_id": operation_id}), 404
    return jsonify(schema_version=1, operation=operation)
