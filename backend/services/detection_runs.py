"""Atomic on-disk run snapshots. Stored results are evidence, not reusable cache."""
import json
import os
from pathlib import Path
import re
import tempfile
from .detection_compute import json_value


class RunStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / 'data' / 'detection_runs'

    def save(self, report):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(report['run_id'])
        fd, temporary = tempfile.mkstemp(dir=self.root, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(json_value(report), stream, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _path(self, identifier):
        if not isinstance(identifier, str) or re.fullmatch('[0-9a-f]{32}', identifier) is None:
            raise ValueError('Invalid run ID')
        return self.root / (identifier + '.json')

    def read(self, identifier):
        return json.loads(self._path(identifier).read_text(encoding='utf-8'))
