"""Compatibility bridge so pydub can import pyaudioop on Python 3.13+.

Python 3.13 removed the stdlib ``audioop`` module. The ``audioop-lts`` package
restores the implementation but keeps the module name ``audioop``. Some third
party libraries (e.g., pydub) fall back to ``pyaudioop`` if ``audioop`` is
missing. This module simply re-exports everything from ``audioop`` so that the
fallback path succeeds in deterministic environments.
"""

try:
    from audioop import *  # noqa: F401,F403
except ImportError as exc:  # pragma: no cover - defensive guard
    raise ImportError(
        "audioop module not found. Install 'audioop-lts' in this environment."
    ) from exc

