"""Ordered tree execution with explicit prerequisites and auditable demo rules."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import logging
import math
import operator
import time
import uuid
from .detection_catalog import get_operation
from .detection_compute import ComputationSession

COMPARE = {'gt': operator.gt, 'gte': operator.ge, 'lt': operator.lt,
           'lte': operator.le, 'eq': operator.eq}
logger = logging.getLogger(__name__)


def now():
    return datetime.now(timezone.utc).isoformat()


class TreeDependencyError(ValueError):
    """Machine-readable dependency violations for editor and execution APIs."""
    def __init__(self, issues, execution_order):
        super().__init__('Tree dependencies must be computed before their consumers')
        self.issues = issues
        self.execution_order = execution_order


def _validate_dependencies(steps, order):
    producers = {}
    for node_id, operation, dependencies in steps:
        if operation is not None:
            producers.setdefault(operation, []).append(node_id)
    available, issues = set(), []
    for node_id, operation, dependencies in steps:
        for dependency in dependencies:
            if dependency in available:
                continue
            candidates = producers.get(dependency, [])
            reason = 'not_computed_before' if candidates else 'missing_computation'
            issues.append({
                'node_id': node_id, 'operation_id': operation,
                'dependency_operation_id': dependency,
                'producer_node_ids': candidates, 'reason': reason,
                'message': (f'节点 {node_id} 需要先完成 {dependency}；'
                            + ('该计算没有排在此节点之前。' if candidates else '树中未配置该计算。')),
            })
        if operation is not None:
            available.add(operation)
    if issues:
        raise TreeDependencyError(issues, order)


def validate_tree(tree, *, check_dependencies=True):
    if not isinstance(tree, dict) or tree.get('schema_version') != 1:
        raise ValueError('tree.schema_version must be 1')
    nodes = tree.get('nodes')
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 500:
        raise ValueError('nodes must contain 1..500 nodes')
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id']:
            raise ValueError('Each node requires a non-empty string id')
        if node['id'] in by_id:
            raise ValueError('Duplicate node id: ' + node['id'])
        if node['id'].startswith('completion:'):
            raise ValueError('completion: is a reserved node ID prefix')
        if node.get('type') not in ('group', 'compute', 'pattern'):
            raise ValueError('Unknown node type')
        children = node.get('children', [])
        if not isinstance(children, list) or any(not isinstance(c, str) for c in children):
            raise ValueError('children must be node IDs')
        if node['type'] == 'pattern' and children:
            raise ValueError('Pattern nodes must be leaves')
        by_id[node['id']] = node
    root = tree.get('root_id')
    if not isinstance(root, str) or root not in by_id or by_id[root]['type'] != 'group':
        raise ValueError('root_id must reference a group')
    order, visited, steps = [], set(), []
    stack = [root]
    while stack:
        identifier = stack.pop()
        if identifier not in by_id or identifier in visited:
            raise ValueError('Missing child, cycle or shared tree node: ' + identifier)
        visited.add(identifier)
        node = by_id[identifier]
        order.append(identifier)
        if node['type'] == 'compute':
            operation_id = node.get('operation')
            if not isinstance(operation_id, str) or get_operation(operation_id) is None:
                raise ValueError('Unknown operation at ' + identifier)
            steps.append((identifier, operation_id, get_operation(operation_id)['dependencies']))
        elif node['type'] == 'pattern':
            if node.get('demo') is not True:
                raise ValueError('Only explicitly marked demo patterns are supported')
            rule = node.get('rule')
            if not isinstance(rule, dict) or rule.get('combine') not in ('all', 'any'):
                raise ValueError('rule.combine must be all or any')
            conditions = rule.get('conditions')
            if not isinstance(conditions, list) or not 1 <= len(conditions) <= 100:
                raise ValueError('A rule requires 1..100 conditions')
            for condition in conditions:
                if not isinstance(condition, dict):
                    raise ValueError('Invalid condition')
                op = condition.get('operation')
                if not isinstance(op, str) or get_operation(op) is None:
                    raise ValueError('Unknown rule input operation')
                fields = {f['name'] for f in get_operation(op)['outputs'] if f['type'] in ('integer', 'number')}
                if not isinstance(condition.get('field'), str) or not isinstance(condition.get('operator'), str) or condition['field'] not in fields or condition['operator'] not in COMPARE:
                    raise ValueError('Unknown numeric field or comparison')
                threshold = condition.get('value')
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                    raise ValueError('Rule threshold must be finite')
            steps.append((identifier, None, list(dict.fromkeys(c['operation'] for c in conditions))))
        stack.extend(reversed(node.get('children', [])))
    if len(visited) != len(nodes):
        raise ValueError('Disconnected tree nodes')
    completion = tree.get('completion_operations', [])
    if not isinstance(completion, list) or len(completion) > 100:
        raise ValueError('Invalid completion_operations')
    for index, op in enumerate(completion):
        if not isinstance(op, str) or get_operation(op) is None:
            raise ValueError('Unknown completion operation')
        steps.append((f'completion:{index}', op, get_operation(op)['dependencies']))
    if check_dependencies:
        _validate_dependencies(steps, order)
    return order


def evaluate(rule, results):
    evidence = []
    for condition in rule['conditions']:
        value = results.get(condition['operation'], {}).get(condition['field'])
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        passed = bool(COMPARE[condition['operator']](value, condition['value'])) if valid else None
        evidence.append({**condition, 'actual': value, 'passed': passed})
    values = [item['passed'] for item in evidence]
    if rule['combine'] == 'all':
        result = False if False in values else (None if None in values else True)
    else:
        result = True if True in values else (None if None in values else False)
    return result, evidence


def execute_tree(content, tree, raw=None, format='grounding', ocr_provider=None):
    tree = deepcopy(tree)
    order = validate_tree(tree)
    session = ComputationSession(content, raw, format, ocr_provider)
    by_id = {node['id']: node for node in tree['nodes']}
    report = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
              'sample_id': hashlib.sha256(content).hexdigest(), 'tree': tree,
              'started_at': now(), 'records': [], 'events': [], 'matched_pattern': None,
              'demo': True, 'result_scope': 'single_run', 'status': 'running'}
    states = {}

    def emit(node_id, status, **extra):
        report['events'].append({'sequence': len(report['events']) + 1, 'node_id': node_id,
                                 'status': status, 'at': now(), **extra})

    def compute(op, record):
        record['operation_id'] = op
        record['reads'] = list(get_operation(op)['dependencies'])
        if op in states:
            record.update(status=states[op], reused=op in session.results)
            if op not in session.results:
                record['reason'] = 'previous_attempt_unsuccessful'
            return
        missing = [dep for dep in record['reads'] if dep not in session.results]
        if missing:
            record.update(status='blocked', blocked_by=missing, reused=False)
        else:
            try:
                session.results[op] = session._execute(op)
                record.update(status='completed', reused=False)
            except Exception:
                logger.exception('Tree operation failed: %s', op)
                record.update(status='failed', reused=False, error={'code': 'operation_failed',
                              'message': 'Calculation failed; inspect input and server log'})
        states[op] = record['status']

    def run_node(identifier, node, phase='tree'):
        record = {'node_id': identifier, 'type': node['type'], 'phase': phase,
                  'sequence': len(report['records']) + 1, 'started_at': now()}
        if report['matched_pattern'] is not None:
            record.update(status='skipped', reason='matched_pattern', elapsed_ms=0)
        else:
            emit(identifier, 'running', phase=phase)
            start = time.perf_counter()
            if node['type'] == 'group':
                record['status'] = 'visited'
            elif node['type'] == 'compute':
                compute(node['operation'], record)
            else:
                matched, evidence = evaluate(node['rule'], session.results)
                record.update(status='matched' if matched is True else 'unmatched' if matched is False else 'blocked',
                              evidence=evidence, demo=True)
                if matched is True:
                    report['matched_pattern'] = identifier
            record['elapsed_ms'] = (time.perf_counter() - start) * 1000
        record['finished_at'] = now()
        report['records'].append(record)
        emit(identifier, record['status'], phase=phase)

    for identifier in order:
        run_node(identifier, by_id[identifier])
    # Completion is explicit and ordered; only the configured feature set is promised.
    for index, op in enumerate(tree.get('completion_operations', [])):
        run_node(f'completion:{index}', {'type': 'compute', 'operation': op}, 'completion')
    incomplete = any(r['status'] in ('failed', 'blocked') for r in report['records'])
    report.update(status='matched' if report['matched_pattern'] else 'incomplete' if incomplete else 'unmatched',
                  has_errors=incomplete, finished_at=now(), results=session.results,
                  completion_satisfied=all(op in session.results for op in tree.get('completion_operations', [])),
                  stop_reason='demo_pattern_matched' if report['matched_pattern'] else
                              'execution_incomplete' if incomplete else 'all_checks_finished')
    return report


def demo_tree():
    def pattern(identifier, operation, field, threshold):
        return {'id': identifier, 'type': 'pattern', 'demo': True, 'children': [],
                'label': '测试模式 ' + identifier, 'rule': {'combine': 'all', 'conditions': [
                    {'operation': operation, 'field': field, 'operator': 'gt', 'value': threshold}]}}
    return {'schema_version': 1, 'strategy_id': 'demo-tree', 'version': 1, 'root_id': 'root',
            'nodes': [
                {'id': 'root', 'type': 'group', 'children': ['ocr', 'gray']},
                {'id': 'ocr', 'type': 'compute', 'operation': 'ocr.structured', 'children': ['rules', 'geometry']},
                {'id': 'rules', 'type': 'compute', 'operation': 'text.rules', 'children': ['A']},
                pattern('A', 'text.rules', 'matched_text_ratio', 0),
                {'id': 'geometry', 'type': 'compute', 'operation': 'text.geometry', 'children': ['B']},
                pattern('B', 'text.geometry', 'text_area_ratio', .3),
                {'id': 'gray', 'type': 'compute', 'operation': 'image.gray', 'children': ['gradient']},
                {'id': 'gradient', 'type': 'compute', 'operation': 'image.gradient', 'children': ['overlap']},
                {'id': 'overlap', 'type': 'compute', 'operation': 'text.gradient_overlap', 'children': ['C']},
                pattern('C', 'text.gradient_overlap', 'gradient_text_coverage', .5)],
            'completion_operations': ['text.contrast']}
