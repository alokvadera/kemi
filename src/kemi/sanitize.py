import re
from typing import List, Pattern


_SUSPICIOUS_PATTERNS: List[Pattern[str]] = [
    re.compile(r"(?i)\bignore\s+(all\s+)?previous\s+instructions\b"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
    re.compile(r"(?i)^\s*system\s*:", re.MULTILINE),
    re.compile(r"(?i)^\s*assistant\s*:", re.MULTILINE),
    re.compile(r"(?i)\[INST\]"),
    re.compile(r"(?i)^\s*###\s*instruction", re.MULTILINE),
    re.compile(r"(?i)\bignore\s+all\b"),
]

_ROLE_PREFIXES: List[Pattern[str]] = [
    re.compile(r"(?i)^\s*user\s*:\s*", re.MULTILINE),
    re.compile(r"(?i)^\s*assistant\s*:\s*", re.MULTILINE),
    re.compile(r"(?i)^\s*system\s*:\s*", re.MULTILINE),
    re.compile(r"(?i)^\s*bot\s*:\s*", re.MULTILINE),
]


def is_suspicious(content: str) -> bool:
    """Check if content contains potential prompt injection patterns.

    Does not modify the content. Returns True if any suspicious pattern found.
    """
    if len(content) < 8:
        return False

    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            return True

    return False


def sanitize(content: str, strict: bool = False) -> str:
    """Remove or neutralize potential prompt injection patterns.

    Default strict=False: removes suspicious patterns only.
    strict=True: additionally removes any line starting with role prefix.

    Protects legitimate short statements (< 8 words) that contain no instruction pattern.
    """
    word_count = len(content.split())

    if word_count < 8 and not is_suspicious(content):
        return content

    result = content

    for pattern in _SUSPICIOUS_PATTERNS:
        result = pattern.sub("[SANITIZED]", result)

    if strict:
        for pattern in _ROLE_PREFIXES:
            result = pattern.sub("[ROLE]", result)

    return result
