"""Validate fixed execution order, then read prerequisites directly by operation ID."""
import logging
import time
from .detection_catalog import get_operation
from .detection_compute import ComputationSession

logger = logging.getLogger(__name__)


def validate_order(order):
    if not isinstance(order, list) or not order or len(order) > 1000:
        raise ValueError('operations must be a non-empty list with at most 1000 entries')
    seen = set()
    for index, identifier in enumerate(order):
        if not isinstance(identifier, str) or get_operation(identifier) is None:
            raise ValueError(f'Unknown operation at position {index + 1}')
        missing = [key for key in get_operation(identifier)['dependencies'] if key not in seen]
        if missing:
            raise ValueError(f'{identifier} requires earlier operations: {", ".join(missing)}')
        seen.add(identifier)
    return list(order)


def execute_order(content, order, raw=None, format='grounding', ocr_provider=None):
    order = validate_order(order)  # Validate the entire schedule before any computation.
    session = ComputationSession(content, raw, format, ocr_provider)
    states, records = {}, []
    for sequence, identifier in enumerate(order, 1):
        dependencies = get_operation(identifier)['dependencies']
        record = {'sequence': sequence, 'operation_id': identifier,
                  'reads': list(dependencies), 'elapsed_ms': 0.0}
        if identifier in states:
            record.update(status=states[identifier], reused=identifier in session.results)
            record['reason'] = 'already_completed' if record['reused'] else 'previous_attempt_unsuccessful'
        else:
            blocked = [key for key in dependencies if states.get(key) != 'completed']
            if blocked:
                record.update(status='blocked', reused=False, blocked_by=blocked)
            else:
                start = time.perf_counter()
                try:
                    # _execute reads named prerequisites from session.results; no recursive search.
                    session.results[identifier] = session._execute(identifier)
                    record.update(status='completed', reused=False)
                except Exception:
                    logger.exception('Operation failed: %s', identifier)
                    record.update(status='failed', reused=False, error={
                        'code': 'operation_failed', 'message': 'Calculation failed; inspect server log and input'})
                record['elapsed_ms'] = (time.perf_counter() - start) * 1000
            states[identifier] = record['status']
        records.append(record)
    return {'schema_version': 1, 'status': 'completed' if all(
        value == 'completed' for value in states.values()) else 'incomplete',
        'operations': order, 'computations': records, 'results': session.results,
        'cache_scope': 'request'}
