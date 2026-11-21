"""Persistent tree configurations with transactional optimistic concurrency."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from .detection_tree import now, validate_tree


class VersionConflict(ValueError):
    def __init__(self, current_version):
        super().__init__('Configuration changed; reload before saving')
        self.current_version = current_version


class TreeStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / 'data' / 'detection_trees'

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / 'trees.sqlite3', timeout=10)
        connection.execute('CREATE TABLE IF NOT EXISTS configs (id TEXT PRIMARY KEY, document TEXT NOT NULL)')
        return connection

    @staticmethod
    def _check_id(identifier):
        if not isinstance(identifier, str) or re.fullmatch('[0-9a-f]{32}', identifier) is None:
            raise ValueError('Invalid configuration ID')

    def read(self, identifier):
        self._check_id(identifier)
        connection = self._connect()
        try:
            row = connection.execute('SELECT document FROM configs WHERE id=?', (identifier,)).fetchone()
            if row is None:
                raise FileNotFoundError('Tree configuration not found')
            return json.loads(row[0])
        finally:
            connection.close()

    def list(self):
        connection = self._connect()
        try:
            records = [json.loads(row[0]) for row in connection.execute('SELECT document FROM configs')]
            keys = ('id', 'name', 'version', 'status', 'created_at', 'updated_at', 'published_version')
            return [{key: record[key] for key in keys}
                    for record in sorted(records, key=lambda r: (r['updated_at'], r['id']), reverse=True)]
        finally:
            connection.close()

    def save(self, payload, identifier=None):
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        name, status = payload.get('name'), payload.get('status')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
            raise ValueError('name must contain 1..200 characters')
        if status not in ('draft', 'published'):
            raise ValueError('status must be draft or published')
        tree = deepcopy(payload.get('tree'))
        validate_tree(tree, check_dependencies=status == 'published')
        expected = payload.get('expected_version')
        if identifier is not None:
            self._check_id(identifier)
            if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
                raise ValueError('expected_version must be a positive integer')
        connection = self._connect()
        try:
            # Serialize read/check/write across processes, not just Flask threads.
            connection.execute('BEGIN IMMEDIATE')
            timestamp = now()
            if identifier is None:
                identifier = uuid4().hex
                record = {'schema_version': 1, 'id': identifier, 'version': 0,
                          'created_at': timestamp, 'published_tree': None, 'published_version': None}
            else:
                row = connection.execute('SELECT document FROM configs WHERE id=?', (identifier,)).fetchone()
                if row is None:
                    raise FileNotFoundError('Tree configuration not found')
                record = json.loads(row[0])
                if record['version'] != expected:
                    raise VersionConflict(record['version'])
            record.update(name=name.strip(), status=status, tree=tree,
                          version=record['version'] + 1, updated_at=timestamp)
            if status == 'published':
                record.update(published_tree=deepcopy(tree), published_version=record['version'])
            document = json.dumps(record, ensure_ascii=False, allow_nan=False)
            connection.execute('INSERT OR REPLACE INTO configs (id, document) VALUES (?, ?)', (identifier, document))
            connection.commit()
            return record
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
