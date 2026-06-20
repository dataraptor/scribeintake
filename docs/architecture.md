# ScribeIntake architecture

> The one thing to take away: **the safety guarantee comes from code, not from the model
> choosing to call a tool.** Every patient turn is run through a deterministic extractor and rule
> engine **before** the LLM is ever invoked. "0 missed on the frozen must-escalate set" is a
> literal `assert` in the test suite, not "the model usually catches it."

---

## Per-turn pipeline

Each patient message flows through one code path: the in-process orchestrator
([`core/.../orchestrator.py`](../core/src/scribeintake/orchestrator.py)). The **EXTRACT** and
**GATE** boxes contain **no LLM call**; everything downstream of the gate is the agent.

```mermaid
flowchart TD
    P([Patient turn]) --> EX["EXTRACT (code, regex + numbers)<br/>safety/extractor.py, NO LLM"]
    EX --> GATE{"GATE (code, declarative rules)<br/>safety/rules.py, NO LLM"}

    GATE -- "EMERGENCY / CRISIS" --> SC["Short-circuit:<br/>core safety template + tel:911 / tel:988<br/>agent NEVER runs"]
    SC --> HALT([Emergency sheet + composer halted])

    GATE -- "CLEAR / CAUTION" --> AG["Agent loop, Sonnet*<br/>agent.py, 3 tools"]
    AG --> T1["record_intake (code)"]
    AG --> T2["retrieve_guideline → local RAG<br/>BM25 + bge embed + bge rerank (all local)"]
    AG --> T3["assess_escalation (LLM second net)"]
    T1 --> CMP{"Intake complete?<br/>intake/state_machine.py (code)"}
    T2 --> CMP
    T3 --> CMP
    CMP -- no --> Q([Adaptive follow-up question])
    CMP -- yes --> BS["build_summary, Opus*, structured outputs<br/>schema-valid SOAP guaranteed"]
    BS --> TR["suggest_triage, floor-clamped (code)<br/>band never below the safety floor"]
    TR --> OUT([Cited SOAP + triage band])

    style EX fill:#dbeafe,stroke:#1e40af
    style GATE fill:#dbeafe,stroke:#1e40af
    style SC fill:#fee2e2,stroke:#b91c1c
    style CMP fill:#dbeafe,stroke:#1e40af
    style TR fill:#dbeafe,stroke:#1e40af
    style AG fill:#fef9c3,stroke:#a16207
    style T3 fill:#fef9c3,stroke:#a16207
    style BS fill:#fef9c3,stroke:#a16207
```

**Legend:** 🟦 blue = **deterministic (code)**, gated at 100%; 🟨 yellow = **LLM** (reported
distributionally, never gated); 🟥 red = the safety short-circuit.

\* **Model pins:** the intake loop targets `claude-sonnet-4-6`, summary/triage/judge target
`claude-opus-4-8`. The wired demo deployment is **Azure GPT-5.5** (a single reasoning model for both
roles). The seam is model-agnostic, so pointing it back at Claude is a config change.

### The seven steps

1. **Extract.** `safety/extractor.py` turns the raw turn into a typed `Signals` object using regex
   and numeric parsing. No model, no network. Deterministic and unit-frozen.
2. **Gate.** `safety/rules.py` evaluates the declarative red-flag rule set against those signals. It
   emits `CLEAR / CAUTION / CRISIS / EMERGENCY`.
3. **Short-circuit.** On EMERGENCY/CRISIS the agent **never runs**; the response is a core safety
   template (verbatim wording, `tel:911` / `tel:988`) and the escalation floor is pinned.
4. **Agent loop.** Otherwise the Sonnet/GPT loop runs with three tools (`record_intake`,
   `retrieve_guideline`, `assess_escalation`). `assess_escalation` is a **second, independent** net
   for oblique danger: it can raise the floor but can never lower the code gate.
5. **Completion check.** `intake/state_machine.py` (code) decides when the required canonical slots
   are filled; until then the agent asks the next adaptive follow-up.
6. **Summary.** `build_summary` is a terminal Opus call using **native structured outputs**, so the
   SOAP is schema-valid by construction (not validated after the fact).
7. **Triage.** `suggest_triage` clamps the predicted band to **never fall below** the safety floor
   set in steps 2 to 4 (monotonic per session).

---

## In-process wiring

The orchestrator is a plain importable module. **Everything imports it directly. There is no
service-to-service HTTP between Python components.**

```mermaid
flowchart LR
    subgraph clients["import core directly"]
        EVAL["eval/: harness ×N<br/>(parallel-safe, isolated SQLite)"]
        OBS["observability/: read-only cost/latency"]
    end
    API["api/: thin FastAPI adapter<br/>(reshape only, no engine logic)"]
    CORE["core/: orchestrator + safety + intake + RAG + tools"]
    UI["app/: ScribeIntake.dc.html<br/>(browser; renders only)"]
    DB[("SQLite<br/>sessions · tool_calls · safety_events · summaries")]

    EVAL --> CORE
    OBS --> CORE
    API --> CORE
    CORE <--> DB
    UI -- "HTTP / SSE" --> API
```

- `eval/` and `observability/` **import `core` in-process**. That is what makes eval runs isolated
  and parallel-safe (fresh SQLite per run; the orchestrator is **stateless per turn**: load state →
  run → save).
- The **browser** frontend can't import a Python module, so it is the one component that talks over
  **HTTP/SSE**, through `api/`, a *thin adapter* that only reshapes what the orchestrator already
  produced (the "no safety/model logic in `api/`" rule is a passing grep test). The load-bearing
  principle (no HTTP where it matters, so evals stay parallel-safe) is preserved.

---

## Two-tier CI

The headline honesty mechanism: it is **structurally impossible to gate a commit on an LLM
number.**

```mermaid
flowchart TD
    C([commit / PR]) --> G["deterministic-gate job<br/>NO API key, free"]
    G --> L["ruff check"]
    G --> T["pytest -m 'not live'  (637 tests)"]
    G --> E["python -m eval.run --deterministic-only<br/>4 gated metrics, exits non-zero on ANY regression"]

    N([nightly cron]) --> F["eval-full job<br/>WITH key"]
    F --> R["python -m eval.run --n 3<br/>distributional + judge + RAGAS + κ"]
    F --> CO["python -m observability<br/>cost report + dashboard"]
    R --> A["upload leaderboard + runs + cost as artifacts<br/>(reports drops; never hard-fails on wobble)"]

    style G fill:#dbeafe,stroke:#1e40af
    style F fill:#fef9c3,stroke:#a16207
```

- **Per-commit (free, deterministic):** lint, the deterministic test tier, and the four gated eval
  metrics (rule correctness, frozen must-escalate, triage-floor-never-violated, schema validity).
  Breaks the build on any regression.
- **Nightly (key-gated, distributional):** the full ×N eval, the LLM judge, RAGAS retrieval evals,
  and the cost/observability report. **Reported, never asserted.** A one-point sampling wobble must
  not break the build; only the still-gated deterministic metrics are build-breaking.

See [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## Safety invariants (hold after every change, never regress)

1. `safety/extractor.py` contains **no LLM call**.
2. The gate runs in **code, upstream of the LLM**, so patient text can never disable it (the
   strongest prompt-injection defense: there is no instruction to override).
3. On a gate EMERGENCY the agent **never runs** (short-circuit), asserted.
4. Escalation is **monotonic per session**: a floor, once set, never lowers.
5. Any exception in the safety path **fails safe** (escalates to in-person care), never silently
   continuing as CLEAR.
6. The predicted triage band is **never below** the safety floor.
