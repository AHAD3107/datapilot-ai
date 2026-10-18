"""Central outbound-inference privacy and data-egress policy for DataPilot AI."""
from __future__ import annotations

import os

_privacy_mode_override: bool | None = None


def set_privacy_mode(enabled: bool | None) -> None:
    """Explicitly configure application-level Privacy Mode (e.g. from UI toggle)."""
    global _privacy_mode_override
    _privacy_mode_override = enabled if enabled is not None else None


def is_privacy_mode_enabled() -> bool:
    """Return True if external AI calls (Gemini, TabPFN) are blocked by application policy.

    Precedence order:
    1. Runtime in-memory override (e.g. Streamlit sidebar toggle or test fixture).
    2. Environment variable DATAPILOT_AIRGAP or DATAPILOT_PRIVACY_MODE ('1', 'true', 'yes').
    3. Default: False (external calls permitted only if tokens are configured).
    """
    global _privacy_mode_override
    if _privacy_mode_override is not None:
        return _privacy_mode_override

    for var in ("DATAPILOT_AIRGAP", "DATAPILOT_PRIVACY_MODE"):
        val = os.getenv(var, "").strip().lower()
        if val in {"1", "true", "yes", "on"}:
            return True
        if val in {"0", "false", "no", "off"}:
            return False

    return False


def is_external_ai_allowed() -> bool:
    """Return True only if external AI calls are permitted by policy."""
    return not is_privacy_mode_enabled()


def get_privacy_status_description() -> dict[str, str]:
    """Return a clear human-readable description of what the application blocks."""
    if is_privacy_mode_enabled():
        return {
            "status": "enabled",
            "badge": "Privacy Mode Active (External AI Disabled)",
            "detail": (
                "Application policy blocks all outbound calls to hosted TabPFN and cloud LLM APIs. "
                "Local profiling, Random Forest, naive baselines, and deterministic explanations "
                "remain fully functional with zero external data egress."
            ),
        }
    return {
        "status": "disabled",
        "badge": "External AI Enabled",
        "detail": (
            "Configured partner services may receive audit data for inference: "
            "hosted TabPFN receives training features; Google Gemini receives "
            "sanitized aggregate metrics (no raw CSV rows)."
        ),
    }
