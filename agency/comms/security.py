"""Security boundary for external AI output.

**External AI responses are DATA, NOT COMMANDS.**

A response from another AI system may contain text that *looks like*
instructions — "ignore all previous instructions", "run rm -rf /", "email the
API keys". This module makes sure such text:

1. is **flagged** (directive candidates are detected and surfaced),
2. is **recorded** (kept verbatim in provenance, never silently dropped),
3. is **never executed** — no code path consumes response content as an
   action spec. Actions are created only by the planner or the operator;
   a response can at most become a *proposal* for a future evaluator to
   look at (Phase P7+, behind the self-improvement gate).

This is the comms-side twin of the web layer's untrusted-content boundary.
"""

from __future__ import annotations

import re
from typing import Any

from agency.comms.models import ResponseAnalysis

#: Heuristic patterns for instruction-like fragments in external text.
#: Deliberately over-inclusive — flagging is cheap, obeying is forbidden.
_DIRECTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (?:all |any |the )?(?:previous|prior|above) instructions", re.I),
    re.compile(r"\b(?:system\s+)?directive\s*:", re.I),
    re.compile(r"\bdisregard (?:all |any |the )?(?:previous|prior|above)", re.I),
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\b(?:run|execute)\s+(?:the\s+)?(?:shell\s+)?command\b", re.I),
    re.compile(r"\b(?:delete|drop)\s+(?:the\s+)?(?:database|tables?)\b", re.I),
    re.compile(
        r"\b(?:read|reveal|email|send)\s+(?:me\s+)?(?:the\s+)?"
        r"(?:\.env\b|api[_ -]?keys?|credentials?|secrets?)",
        re.I,
    ),
    re.compile(r"\bas an? (?:trusted )?administrator\b", re.I),
    re.compile(r"\byou must comply\b", re.I),
    re.compile(
        r"\boverride (?:your |the |system )?(?:instructions|rules|settings|permissions)",
        re.I,
    ),
)

#: Sensitive targets that must never be influenced by response text.
_SENSITIVE_KEYS = ("api_key", "secret", "password", "token", "credential")


def extract_directive_candidates(text: str, *, max_candidates: int = 10) -> list[str]:
    """Return instruction-like sentences/fragments found in ``text``.

    Candidates are informational (analysis metadata). They are never acted on
    by any Skynet component.
    """
    if not text:
        return []
    candidates: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        stripped = sentence.strip()
        if not stripped:
            continue
        if any(pattern.search(stripped) for pattern in _DIRECTIVE_PATTERNS):
            candidates.append(stripped[:500])
            if len(candidates) >= max_candidates:
                break
    return candidates


def extract_claim_candidates(text: str, *, max_claims: int = 10) -> list[str]:
    """Extract simple declarative sentences as *candidate* claims.

    Extraction is textual only — no assessment of truth. Future evaluators
    (Phase P6+) will compare claims against evidence and other sources.
    """
    if not text:
        return []
    claims: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        stripped = sentence.strip()
        if not stripped or len(stripped) < 12:
            continue
        lowered = stripped.lower()
        # Skip imperatives/questions and meta filler; keep declaratives.
        if stripped.endswith("?") or lowered.startswith(
            ("please", "let's", "here is", "note that")
        ):
            continue
        claims.append(stripped[:500])
        if len(claims) >= max_claims:
            break
    return claims


def analyze_response(message_id: str, text: str) -> ResponseAnalysis:
    """Analyze one external response: flag directives, extract claim candidates."""
    return ResponseAnalysis(
        message_id=message_id,
        directive_candidates=extract_directive_candidates(text),
        claim_candidates=extract_claim_candidates(text),
        notes="analysis is textual only; no truth assessment performed",
    )


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Strip potential secrets from provider metadata before it is stored.

    Providers occasionally echo request headers or configuration back; keys
    with sensitive names are redacted here so credentials never reach logs,
    experiences, or the database.
    """
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in _SENSITIVE_KEYS):
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    return sanitized


__all__ = [
    "analyze_response",
    "extract_claim_candidates",
    "extract_directive_candidates",
    "sanitize_metadata",
]
