"""Shared input parsing and gating logic. No streamlit import."""

from __future__ import annotations

import re
from typing import MutableMapping

from portfolio_ui.dataset import ActiveDataset
from portfolio_ui.sources.base import Capability
from portfolio_ui.state import get_active_dataset

_SEPARATORS = re.compile(r"[,\s]+")


def parse_tickers(raw: str) -> list[str]:
    """Turn free-typed ticker input into a clean, ordered, unique list."""
    parts = [p.strip().upper() for p in _SEPARATORS.split(raw or "") if p.strip()]
    seen: dict[str, None] = {}
    for part in parts:
        seen.setdefault(part, None)
    return list(seen)


def capability_blocked_reason(source, capability: Capability) -> str | None:
    """Why this action is disabled, or None if it is allowed.

    Actions are never hidden (spec section 4.3) - they are disabled with the
    reason shown, so a feature never looks nonexistent.
    """
    if not source.is_available():
        return source.unavailable_reason()
    if not source.supports(capability):
        return (
            f"{source.name} does not support {capability.value} - "
            f"switch to a source that does"
        )
    return None


def require_active_dataset(store: MutableMapping) -> ActiveDataset | None:
    """The active dataset, or None when the user has not built one yet."""
    return get_active_dataset(store)
