"""NKI kernel wrappers for nanowhale.

Each kernel lives in its own submodule. This package also exposes the
runtime toggle helper `is_enabled(name)` driven by the `NKI_KERNELS`
environment variable (comma-separated kernel names, e.g.
`NKI_KERNELS=cross_entropy,swiglu`).

Imports of `nki` / `nkilib` / `torch_neuronx` are deferred to first call
inside each wrapper, so this package is safe to import on CPU.
"""

from __future__ import annotations

import os

from .cross_entropy import nki_cross_entropy

# Registry of valid kernel names. Add new entries here when introducing a
# new wrapper. Unknown names in NKI_KERNELS will raise at import time so
# typos don't silently no-op.
KNOWN_KERNELS: frozenset[str] = frozenset({"cross_entropy"})


def _parse_enabled() -> frozenset[str]:
    raw = os.environ.get("NKI_KERNELS", "").strip()
    if not raw:
        return frozenset()
    names = frozenset(s.strip() for s in raw.split(",") if s.strip())
    unknown = names - KNOWN_KERNELS
    if unknown:
        raise ValueError(
            f"NKI_KERNELS contains unknown kernel name(s): {sorted(unknown)}. "
            f"Known: {sorted(KNOWN_KERNELS)}"
        )
    return names


_ENABLED: frozenset[str] = _parse_enabled()


def is_enabled(name: str) -> bool:
    """True iff `name` is in the NKI_KERNELS env var."""
    return name in _ENABLED


def enabled_kernels() -> frozenset[str]:
    """Snapshot of currently-enabled kernels (for logging / banners)."""
    return _ENABLED


__all__ = ["nki_cross_entropy", "is_enabled", "enabled_kernels", "KNOWN_KERNELS"]
