"""Security boundary for the autonomous orchestrator (spec §18).

External content (web pages, AI responses) is **data**. This module is
where that guarantee is made checkable:

- :func:`scan_content` flags directive-like fragments (the same injection
  patterns the P5 comms security layer detects) so they are *recorded*
  on Evidence without ever being acted on;
- :func:`assert_no_control_payload` is the belt-and-braces check the
  orchestrator runs before content influences any decision structure:
  directive-shaped payloads raise instead of executing;
- the orchestrator's stage handlers take no callable, path, or command
  parameters from evidence at all — content can only fill *text* fields
  (queries, focus), which are then only ever compared or passed to
  adapters as search strings.

The orchestrator cannot grant itself permissions: it executes no shell,
opens no sockets beyond the wrapped P4/P5 adapters, reads no credential
material, and every state-changing action (improvement application)
remains behind the P7 human approval gate.
"""

from __future__ import annotations

import re
from typing import Any

# Directive-shaped fragments (kept in sync with the P5 comms security
# module's intent; orchestrator-local copy so the boundary does not depend
# on comms being installed).
_DIRECTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore (all|any|previous)( other)?( previous)? instructions",
        r"disregard (all|any|previous)( other)?( previous)? instructions",
        r"you are now",
        r"new instructions?:",
        r"system prompt",
        r"override (your|the) (instructions|settings|configuration)",
        r"execute (the )?(following|this) (command|code)",
        r"run (the )?(following|this) (command|code)",
        r"delete (all|the) (files|data|memory)",
        r"disable (all )?(safety|security)",
        r"reveal (your|the) (api ?key|credentials|secret)",
        r"print (your|the) (api ?key|credentials|secret)",
    )
)

#: Keys that must never appear in evidence provenance (secret-shaped).
_SENSITIVE_KEYS = ("api_key", "apikey", "secret", "password", "token", "credential")


def scan_content(text: str) -> list[str]:
    """Return directive-like fragments found in ``text`` (recorded, not run)."""
    hits: list[str] = []
    for pattern in _DIRECTIVE_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(match.group(0))
    return hits


def sanitize_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Strip secret-shaped keys from provenance dicts before persistence."""
    return {
        key: value
        for key, value in provenance.items()
        if not any(marker in key.lower() for marker in _SENSITIVE_KEYS)
    }


def assert_no_control_payload(payload: Any) -> None:
    """Raise if a decision-input payload carries directive-shaped content.

    Called by the orchestrator before external content feeds plan or
    decision structures. The response to a violation is to *refuse and
    record* — the payload is never interpreted as instructions.
    """
    if isinstance(payload, str):
        hits = scan_content(payload)
        if hits:
            raise ValueError(
                f"external content contains directive-like fragment(s) "
                f"({hits[:2]}); refusing to use it as control input"
            )
    elif isinstance(payload, dict):
        for value in payload.values():
            assert_no_control_payload(value)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            assert_no_control_payload(item)


__all__ = [
    "assert_no_control_payload",
    "sanitize_provenance",
    "scan_content",
]
