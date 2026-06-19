# scribeintake
Agentic pre-visit clinical intake chat with a deterministic safety gate, cited-guideline RAG, and a published eval harness — outputs a SOAP summary + triage band.

## Safety scope (v1)

The safety subsystem (`core/src/scribeintake/safety/`) is **deterministic and LLM-free**: a
reproducible signal extractor feeds a declarative rule engine, so the must-escalate set is
gated in code, not by the model. The honest reliability claim is **"0 missed on the frozen
must-escalate set, with deterministic rules; end-to-end adversarial recall reported
distributionally"** — not "100% recall, period".

**Adult patients only.** Pediatric/neonatal red flags (infant fever, dehydration,
weight-based dosing) are deliberately out of scope for v1 — they need a different rule set.

This is an **educational demo, not a diagnostic or crisis service.**
