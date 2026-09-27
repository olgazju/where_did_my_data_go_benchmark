"""Where Did My Data Go? — a benchmark for evidence-based missing-entity investigation.

Small and readable. The single source of truth is the pipeline *simulator*
(:mod:`poc.simulator`); every observable artifact and reference label derives from one
execution so nothing can drift. The official prompt is :mod:`poc.official_prompt` (P2) and
the official evidence validator is :mod:`poc.validators`.
"""

__version__ = "0.1.0"
