"""Stateless draft edits. Applying a draft still requires full tree validation."""
from copy import deepcopy
from uuid import uuid4

from .detection_tree import validate_tree


def delete_node(tree, node_id, mode):
    validate_tree(tree, check_dependencies=False)
    if not isinstance(node_id, str) or node_id == tree['root_id']:
        raise ValueError('A non-root node ID is required')
    if mode not in ('subtree', 'promote_children'):
        raise ValueError('mode must be subtree or promote_children')
    draft = deepcopy(tree)
    nodes = {n['id']: n for n in draft['nodes']}
    if node_id not in nodes:
        raise ValueError('Node not found')
    parent = next(n for n in draft['nodes'] if node_id in n.get('children', []))
    index = parent['children'].index(node_id)
    removed = [node_id]
    if mode == 'subtree':
        stack = list(nodes[node_id].get('children', []))
        while stack:
            child = stack.pop()
            removed.append(child)
            stack.extend(nodes[child].get('children', []))
        replacement = []
    else:
        replacement = nodes[node_id].get('children', [])
    parent['children'][index:index + 1] = replacement
    removed_set = set(removed)
    draft['nodes'] = [n for n in draft['nodes'] if n['id'] not in removed_set]
    validate_tree(draft, check_dependencies=False)
    return {'schema_version': 1, 'status': 'draft', 'tree': draft,
            'removed_node_ids': removed, 'parent_id': parent['id'],
            'requires_validation': True}


def add_node(tree, parent_id, node, index=None):
    # Drafts may have unresolved dependencies, but must remain actual trees.
    validate_tree(tree, check_dependencies=False)
    if not isinstance(parent_id, str):
        raise ValueError('parent_id must be a node ID')
    if not isinstance(node, dict) or node.get('type') not in ('compute', 'pattern'):
        raise ValueError('node.type must be compute or pattern')
    if node.get('children', []) != []:
        raise ValueError('A new node must have no children')
    draft = deepcopy(tree)
    parent = next((n for n in draft['nodes'] if n['id'] == parent_id), None)
    if parent is None:
        raise ValueError('Parent node not found')
    if parent['type'] == 'pattern':
        raise ValueError('Pattern nodes must remain leaves')
    children = parent.setdefault('children', [])
    if index is None:
        index = len(children)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(children):
        raise ValueError('index must be an integer between 0 and the child count')
    added = deepcopy(node)
    if 'id' not in added:
        added['id'] = uuid4().hex
    added['children'] = []
    draft['nodes'].append(added)
    children.insert(index, added['id'])
    validate_tree(draft, check_dependencies=False)
    return {'schema_version': 1, 'status': 'draft', 'tree': draft,
            'node_id': added['id'], 'parent_id': parent_id, 'index': index,
            'requires_validation': True}
