"""ScribeIntake evaluation harness package.

Imported in-process by the harness/runner (Splits 07/08) — it depends on ``core``
directly (``from scribeintake...``), never over HTTP, so eval runs are isolated and
parallel-safe (spec section 15).

**Naming note:** the spec uses ``evals/`` (plural); these split docs use ``eval/``
(singular), which shadows the Python builtin ``eval()`` *within modules that import it*.
No module in this package calls the builtin ``eval()``, so the shadow is inert. The repo
was scaffolded with ``eval/`` (Split 01: ``eval/tests/`` + root ``testpaths``); Split 06
keeps it for consistency across splits 06–14 (Split 06 session log).
"""

__all__ = ["scenario", "gate_check", "models", "harness", "metrics", "run"]
