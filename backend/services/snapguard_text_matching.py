"""SnapGuard OCR matching functions, preserved verbatim from upstream.
Source: https://github.com/Dora238/SnapGuard-anoy/blob/main/detector_image/snapguard.py
Git blob: 80f282f5ea02b3ee4db177409d1ea540245d3ebd
Only the dependency-free text matching functions are included.
"""
import re
from typing import List, Dict

def normalize_for_action(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\[\]\(\)]", "", s)
    return s


def is_position_click(text: str) -> bool:
    """
    Detect position-based clicks and click inducement
    under aggressive OCR normalization (no spaces).
    """

    patterns = [
        r"click[\[\(]\d+[\]\)]",
        r"clickhere",
        r"pleaseclick",
        r"clickto",
        r"clickthe",
        r"taphere",
        r"presshere",
    ]

    return any(re.search(p, text) for p in patterns)


def is_password_input(text: str) -> bool:
    return (
        any(k in text for k in ["password", "passwd", "pwd"])
        and any(v in text for v in ["enter", "input", "type"])
    )


def is_link_invitation(text: str) -> bool:
    return (
        "http" in text
        or "www" in text
        or ("link" in text and any(v in text for v in ["click", "open", "visit"]))
    )


def is_control_instruction(text: str) -> bool:
    """
    Detect control / override instructions under
    aggressive OCR normalization (no spaces).
    """

    keywords = [
        "alert",
        "ignore",
        "follow",
        "donotask",
        "dontask",
        "withoutasking",
        "withoutuser",
        "donotinform",
        "noneedtoask",
        "dontinform",
        "noaskuser",
        "askagainmanytimes",
    ]

    return any(k in text for k in keywords)


def is_api_key_request(norm_line: str) -> bool:
    """
    Detect requests for API keys / credentials under OCR noise.
    """

    credential_patterns = [
        r"api\s*key",
        r"apikey",
        r"access\s*token",
        r"auth\s*token",
        r"secret",
        r"private\s*key",
    ]

    request_patterns = [
        r"share",
        r"provide",
        r"send",
        r"give",
        r"pass",
        r"supply",
        r"need",
        r"require",
    ]

    has_credential = any(re.search(p, norm_line) for p in credential_patterns)
    has_request = any(re.search(p, norm_line) for p in request_patterns)

    if has_credential and has_request:
        return True

    if re.search(r"\bsk-[a-z0-9]{10,}\b", norm_line):
        return True

    return False


def detect_malicious_actions(ocr_text: str) -> List[Dict[str, str]]:
    """
    Return detailed malicious action matches.
    Each item:
      {
        "action_type": str,
        "matched_text": str
      }
    """
    hits = []

    for line in ocr_text.splitlines():
        norm = normalize_for_action(line)

        if is_position_click(norm):
            hits.append({
                "action_type": "position_click",
                "matched_text": line
            })

        if is_password_input(norm):
            hits.append({
                "action_type": "password_input",
                "matched_text": line
            })

        if is_link_invitation(norm):
            hits.append({
                "action_type": "link_invitation",
                "matched_text": line
            })

        if is_control_instruction(norm):
            hits.append({
                "action_type": "control_instruction",
                "matched_text": line
            })

        if is_api_key_request(norm):
            hits.append({
                "action_type": "credential_request",
                "matched_text": line
            })

    return hits
