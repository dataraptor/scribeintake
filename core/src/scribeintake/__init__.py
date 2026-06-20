"""ScribeIntake core engine.

Framework-free package holding the data contracts, SQLite access layer, cost
accounting, and (in later splits) the deterministic safety gate, intake engine,
RAG subsystem, and per-turn orchestrator. Imported directly by the API layer and
the eval harness; it knows nothing about HTTP or the UI.
"""

__version__ = "1.0.0"
