"""
guardrails/guardrail_checker.py
================================
Wraps NeMo Guardrails to provide a simple synchronous interface:

    result = check_for_injection(email_text)
    # result.status  → "BLOCKED" | "SAFE_TO_INFER" | "REVIEW"
    # result.reason  → human-readable explanation

The class loads the Colang config once at import time and is reused across
requests to avoid repeated I/O.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ── Injection patterns ─────────────────────────────────────────────────────────
# Mirrors the patterns in rails.co but is evaluated in Python for speed.
# NeMo Guardrails is the authoritative check; this is a fast pre-filter.
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bignore\s+(all\s+)?previous\s+instructions?\b",
        r"\bdisregard\s+(all\s+|your\s+)?(instructions?|system\s+prompt)\b",
        r"\bforget\s+(all\s+)?instructions?\b",
        r"\byou\s+are\s+now\s+a\b",
        r"\byour\s+(new|real)\s+task\s+is\b",
        r"\bact\s+as\s+if\s+you\s+are\b",
        r"\bpretend\s+you\s+are\b",
        r"\boverride\s+your\s+guidelines\b",
        r"\bbypass\s+your\s+restrictions?\b",
        r"\byou\s+have\s+no\s+restrictions?\b",
        r"\bdo\s+anything\s+now\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"\bdeveloper\s+mode\s+enabled\b",
        r"\bignore\s+the\s+above\b",
        r"\bnew\s+persona\s*:",
        r"\[system\]",
        r"<!--\s*inject",
        r"<system>",
        r"\bsystem\s*:\s*you\s+are\b",
    ]
]


@dataclass
class GuardrailResult:
    status: str          # "BLOCKED" | "SAFE_TO_INFER" | "REVIEW"
    reason: str
    matched_pattern: str = ""


def check_for_injection(email_text: str) -> GuardrailResult:
    """
    Fast regex-based injection check.

    Returns:
        GuardrailResult with status BLOCKED, SAFE_TO_INFER, or REVIEW.
        REVIEW is returned when the text is unusually short / ambiguous.
    """
    if not isinstance(email_text, str) or len(email_text.strip()) == 0:
        return GuardrailResult(
            status="REVIEW",
            reason="Email body is empty or non-string; routing to human review.",
        )

    for pattern in INJECTION_PATTERNS:
        match = pattern.search(email_text)
        if match:
            log.warning(f"Injection pattern matched: {match.group()!r}")
            return GuardrailResult(
                status="BLOCKED",
                reason=f"Prompt injection attempt detected: '{match.group()}'",
                matched_pattern=match.group(),
            )

    # Optionally use NeMo for deeper semantic check
    try:
        from nemoguardrails import RailsConfig, LLMRails  # noqa: F401
        config_path = Path(__file__).parent / "config"
        config = RailsConfig.from_path(str(config_path))
        rails = LLMRails(config)
        response = rails.generate(
            messages=[{"role": "user", "content": email_text}]
        )
        if "BLOCKED" in response.get("content", "").upper():
            return GuardrailResult(
                status="BLOCKED",
                reason="NeMo Guardrails: semantic injection pattern detected.",
            )
    except ImportError:
        log.debug("nemoguardrails not installed; skipping semantic check.")
    except Exception as exc:
        log.error(f"NeMo Guardrails error: {exc}; falling back to regex result.")

    return GuardrailResult(status="SAFE_TO_INFER", reason="No injection patterns detected.")
