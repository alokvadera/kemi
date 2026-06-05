"""Prompt injection detection and sanitization with audit logging.

Security layer for protecting AI agents from prompt injection attacks.
Provides detection, sanitization, and comprehensive audit logging.
"""

import hashlib
import logging
import re
from re import Pattern
from typing import Any

logger = logging.getLogger(__name__)

# Detectable injection pattern categories
_SUSPICIOUS_PATTERNS: list[tuple[Pattern[str], str]] = [
    # Instruction override attempts
    (re.compile(r"(?i)\bignore\s+(all\s+)?previous\s+instructions\b"), "instruction_override"),
    (re.compile(r"(?i)\byou\s+are\s+now\b"), "role_override"),
    (re.compile(r"(?i)\bignore\s+all\b"), "ignore_all"),
    # Role playing / jailbreak attempts
    (re.compile(r"(?i)^\s*system\s*:", re.MULTILINE), "system_prefix"),
    (re.compile(r"(?i)^\s*assistant\s*:", re.MULTILINE), "assistant_prefix"),
    (re.compile(r"(?i)\[INST\]"), "inst_token"),
    (re.compile(r"(?i)^\s*###\s*instruction", re.MULTILINE), "markdown_instruction"),
]

_ROLE_PREFIXES: list[tuple[Pattern[str], str]] = [
    (re.compile(r"(?i)^\s*user\s*:\s*", re.MULTILINE), "user_role"),
    (re.compile(r"(?i)^\s*assistant\s*:\s*", re.MULTILINE), "assistant_role"),
    (re.compile(r"(?i)^\s*system\s*:\s*", re.MULTILINE), "system_role"),
    (re.compile(r"(?i)^\s*bot\s*:\s*", re.MULTILINE), "bot_role"),
]


def _log_detection(
    content_length: int,
    content_hash: str,
    pattern_name: str,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Log prompt injection detection event for audit purposes.

    Args:
        content_length: Length of the content (avoids logging actual content)
        content_hash: SHA256 hash of content for identification without exposure
        pattern_name: Name of the pattern that matched
        action: What action was taken (detected, sanitized, rejected)
        details: Additional context for the log entry
    """
    log_data: dict[str, Any] = {
        "event": "prompt_injection_detection",
        "pattern": pattern_name,
        "action": action,
        "content_length": content_length,
        "content_hash": content_hash[:16],  # Only first 16 chars of hash
    }
    if details:
        log_data.update(details)

    # Use appropriate log level based on severity
    if action == "rejected":
        logger.error("Prompt injection attempt detected and rejected: %s", log_data)
    else:
        logger.warning("Prompt injection pattern detected: %s", log_data)


def _get_content_hash(content: str) -> str:
    """Get SHA256 hash of content for audit logging without exposing content."""
    return hashlib.sha256(content.encode()).hexdigest()


def is_suspicious(content: str) -> bool:
    """Check if content contains potential prompt injection patterns.

    Does not modify the content. Returns True if any suspicious pattern found.
    Logs detection event for audit purposes (only metadata, not content).
    """
    if len(content) < 8:
        return False

    for pattern, pattern_name in _SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            _log_detection(
                len(content),
                _get_content_hash(content),
                pattern_name,
                "detected",
            )
            return True

    return False


def sanitize(content: str, strict: bool = False) -> str:
    """Remove or neutralize potential prompt injection patterns.

    Default strict=False: removes suspicious patterns only.
    strict=True: additionally removes any line starting with role prefix.

    Protects legitimate short statements (< 8 words) that contain no instruction pattern.

    Logs all sanitization events for audit purposes (only metadata, not content).
    """
    word_count = len(content.split())

    if word_count < 8 and not is_suspicious(content):
        return content

    result = content
    detected_patterns: list[str] = []

    for pattern, pattern_name in _SUSPICIOUS_PATTERNS:
        if pattern.search(result):
            detected_patterns.append(pattern_name)
            result = pattern.sub("[SANITIZED]", result)

    if strict:
        for pattern, pattern_name in _ROLE_PREFIXES:
            if pattern.search(result):
                detected_patterns.append(pattern_name)
                result = pattern.sub("[ROLE]", result)

    # Log the sanitization event
    if detected_patterns:
        _log_detection(
            len(content),
            _get_content_hash(content),
            ", ".join(detected_patterns),
            "sanitized",
            {"strict_mode": strict, "patterns_found": len(detected_patterns)},
        )

    return result


def sanitize_with_rejection(content: str, strict: bool = False) -> tuple[str, bool]:
    """Sanitize content and indicate whether it was suspicious.

    Args:
        content: The content to sanitize
        strict: Whether to use strict mode (also remove role prefixes)

    Returns:
        Tuple of (sanitized_content, was_suspicious)
    """
    word_count = len(content.split())

    # Fast path: short non-suspicious content
    if word_count < 8:
        # Check if suspicious without logging (to avoid double logging)
        is_susp = False
        for pattern, _ in _SUSPICIOUS_PATTERNS:
            if pattern.search(content):
                is_susp = True
                break
        if not is_susp:
            return content, False

    # Track if content is suspicious (without double logging)
    was_suspicious = False
    content_hash = _get_content_hash(content)

    # Check for suspicious patterns, only log once
    for pattern, pattern_name in _SUSPICIOUS_PATTERNS:
        if pattern.search(content):
            was_suspicious = True
            _log_detection(
                len(content),
                content_hash,
                pattern_name,
                "sanitized_with_rejection",
                {"strict_mode": strict},
            )
            break  # Only log once, not per pattern

    # Sanitize (may log additional role prefix patterns in strict mode)
    sanitized = sanitize(content, strict)

    return sanitized, was_suspicious
