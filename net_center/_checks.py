"""Input checks shared by the numerical modules.

Each helper returns a normalised value or raises ``ValueError`` with a message
that names the offending argument. Before this module existed, the same checks
were repeated in ``center.py``, ``distances.py``, ``solve.py`` and
``topology.py``; keeping one copy guarantees that every entry point applies the
same rule.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("threading", "loky")


def _is_int(value) -> bool:
    """True for Python/NumPy integers, but not for booleans (``True == 1``)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def positive_int(value, name: str) -> int:
    """Return ``value`` as ``int`` if it is an integer of at least 1."""
    if not _is_int(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def non_negative_int(value, name: str) -> int:
    """Return ``value`` as ``int`` if it is an integer of at least 0."""
    if not _is_int(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def backend(value, name: str = "backend") -> str:
    """Return a joblib backend name after checking that net-center supports it."""
    if value not in BACKENDS:
        raise ValueError(f"{name} must be 'threading' or 'loky'")
    return value


def index_array(values, what: str) -> np.ndarray:
    """Return node indices as a flat ``int64`` array.

    Floating-point input is refused rather than truncated: ``2.7`` silently
    becoming node ``2`` would be a plausible wrong answer.
    """
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{what} must contain integer node indices")
    return raw.astype(np.int64, copy=False).reshape(-1)


def demand_weights(weights, n_demands: int) -> np.ndarray:
    """Return median weights as a flat ``float64`` array after checking them.

    Weights must match the demand points one to one, be finite and
    non-negative, and include at least one positive value (otherwise every
    node would tie at zero and the median would be arbitrary).
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(w) != n_demands:
        raise ValueError(
            f"got {len(w)} weights for {n_demands} demand points; "
            "they must correspond one to one"
        )
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinity")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    if not (w > 0).any():
        raise ValueError("at least one demand weight must be positive")
    return w
