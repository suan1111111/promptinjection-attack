"""Persistent numerical summaries of upstream SnapGuard text matches."""
import hashlib
import json
from pathlib import Path
import sqlite3
from .snapguard_text_matching import detect_malicious_actions
from contextlib import contextmanager


@contextmanager
def connection(path):
    db = sqlite3.connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()

ACTION_TYPES = ('position_click', 'password_input', 'link_invitation', 'control_instruction', 'credential_request')
# Include implementation and aggregation semantics to invalidate legacy results.
RULE_ID = hashlib.sha256(Path(__file__).with_name('snapguard_text_matching.py').read_bytes() + b'line-count-and-original-line-coverage-v1').hexdigest()


def extract_text_rules(text):
    hits = detect_malicious_actions(text)
    result = {category + '_count': sum(hit['action_type'] == category for hit in hits) for category in ACTION_TYPES}
    # Upstream exposes whole matched lines, not character-level matching spans.
    matched_length = sum(len(line) for line in text.splitlines() if detect_malicious_actions(line))
    result['matched_text_ratio'] = matched_length / len(text) if text else 0.0
    return result


class TextRuleStore:
    def __init__(self, root=None):
        root = Path(root) if root else Path(__file__).resolve().parents[2] / 'data' / 'image_features'
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'text_rules.sqlite3'
        with connection(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS ocr (image_id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS rules (image_id TEXT, text_id TEXT, rule_id TEXT, result TEXT, PRIMARY KEY(image_id,text_id,rule_id))')

    def extract(self, image, text=None, ocr=None, source='supplied OCR text'):
        image_id = hashlib.sha256(Path(image).read_bytes()).hexdigest()
        with connection(self.path) as db:
            row = db.execute('SELECT text FROM ocr WHERE image_id=?', (image_id,)).fetchone()
        if text is None:
            if row:
                text = row[0]
            elif ocr:
                text = ocr(image)  # Failed inference never overwrites completed data.
                source = 'PrivaSee free_ocr'
            else:
                raise ValueError('No cached OCR text; supply text or an OCR provider')
        if not isinstance(text, str):
            raise ValueError('OCR did not return text')
        with connection(self.path) as db:
            db.execute('INSERT OR REPLACE INTO ocr VALUES (?,?,?)', (image_id, text, source))
        text_id = hashlib.sha256(text.encode()).hexdigest()
        with connection(self.path) as db:
            cached = db.execute('SELECT result FROM rules WHERE image_id=? AND text_id=? AND rule_id=?', (image_id, text_id, RULE_ID)).fetchone()
        if cached:
            return json.loads(cached[0])
        result = extract_text_rules(text)
        with connection(self.path) as db:
            db.execute('INSERT OR REPLACE INTO rules VALUES (?,?,?,?)', (image_id, text_id, RULE_ID, json.dumps(result)))
        return result
