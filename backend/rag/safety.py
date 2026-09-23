"""
AgriN RAG — CIBRC Safety Validation
Post-processing check: flag any banned pesticides in Gemini advisory output.
"""

import os
import json
import re
import logging

logger = logging.getLogger("agrin.rag.safety")

_banned = None
_restricted = None


def _load_cibrc():
    global _banned, _restricted
    if _banned is not None:
        return

    data_path = os.path.join(os.path.dirname(__file__), "data", "cibrc_banned.json")
    try:
        with open(data_path) as f:
            data = json.load(f)
        _banned = {name.lower() for name in data.get("banned_pesticides", [])}
        _restricted = {name.lower() for name in data.get("restricted_pesticides", [])}
        logger.info(f"CIBRC loaded: {len(_banned)} banned, {len(_restricted)} restricted")
    except Exception as e:
        logger.warning(f"CIBRC data not loaded: {e}")
        _banned = set()
        _restricted = set()


def check_advisory(advisory_text: str) -> dict:
    """
    Check advisory text for banned/restricted pesticides.
    Returns dict with findings and a cleaned advisory if issues found.
    """
    _load_cibrc()

    text_lower = advisory_text.lower()
    found_banned = []
    found_restricted = []

    for chemical in _banned:
        # Word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(chemical) + r'\b'
        if re.search(pattern, text_lower):
            found_banned.append(chemical.title())

    for chemical in _restricted:
        pattern = r'\b' + re.escape(chemical) + r'\b'
        if re.search(pattern, text_lower):
            found_restricted.append(chemical.title())

    result = {
        "safe": len(found_banned) == 0,
        "banned_found": found_banned,
        "restricted_found": found_restricted,
        "advisory": advisory_text,
    }

    if found_banned:
        warning = (
            "\n\n⚠️ SAFETY WARNING: The following chemicals mentioned above "
            f"are BANNED in India by CIBRC: {', '.join(found_banned)}. "
            "Do NOT use them. Consult your local KVK for approved alternatives."
        )
        result["advisory"] = advisory_text + warning
        logger.warning(f"Banned pesticides found in advisory: {found_banned}")

    if found_restricted:
        note = (
            f"\n\nNote: {', '.join(found_restricted)} "
            "are restricted-use pesticides in India. Use only as directed "
            "and follow all safety precautions on the label."
        )
        if not found_banned:  # don't double-append if we already added warning
            result["advisory"] = advisory_text + note

    return result


def get_banned_list_for_prompt() -> str:
    """Return banned pesticide names for injection into Gemini prompt."""
    _load_cibrc()
    if not _banned:
        return ""
    return ", ".join(sorted(name.title() for name in _banned))
