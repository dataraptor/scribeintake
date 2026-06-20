# Gold eval scenarios

> ⚠️ **Synthetic & educational.** These ~58 cases are *synthetic* clinical vignettes authored
> for evaluation — **not** real patient data and **not** clinician-authored gold. Credibility
> comes from **method, not authority** (see *Authoring method* below). The product never
> diagnoses or prescribes; this dataset only measures the engine.

The gold dataset is the eval harness's headline **input** (spec §15 / Appendix C). Each case is
a schema-valid YAML (`eval/scenario.py` → `Scenario`) with patient `turns`, an `expect`ed
escalation/triage, an optional loose `gold_soap`, and a cited `provenance`. The harness (Split
07) drives each case's turns through the **real** per-turn pipeline; this split authors and
validates the data and — crucially — **cross-checks the labels against the real `safety` gate**
with no API key (`tests/test_must_escalate_gate.py`).

## Mix (this set)

| Folder / category | Count | Escalation | `escalation_source` | `gold_soap` |
|---|---|---|---|---|
| `must_escalate/` (**frozen**) | 16 | EMERGENCY | `[gate]` | — (none) |
| `urgent/` | 9 | URGENT | `[gate]` | yes (`gp_urgent`) |
| `routine/` | 12 | CLEAR | `[]` | yes (`self_care`/`gp_routine`) |
| `benign/` (false-alarm probes) | 5 | CLEAR | `[]` | yes |
| `adversarial/` | 10 | mixed | `[gate]` or `[gate, agent]` | only the CLEAR correction |
| `heldout/` (anti-overfit slice) | 6 | mixed | mixed | per declared category |

**By category** (folder-derived; held-out cases declare their real category): must_escalate 18 ·
urgent 10 · routine 13 · benign 6 · adversarial 11. **Total 58**, held-out 6 (drawn across
must_escalate ×2, urgent, routine, benign, adversarial). All EMERGENCY rules (16) are exercised.

`adversarial/` breaks down as: 5 **oblique-recall** stressors (`[gate, agent]` — danger phrased
so the regex may miss it; either net may catch it), 3 **prompt-injection** cases (`[gate]` —
override text then a real danger; the gate fires regardless), and 2 **contradiction/correction**
cases (one CLEAR latest-wins, one where the correction *reveals* danger).

## Authoring method (why these are credible without clinicians)

1. **Every label derives from a cited public-domain threshold** (Appendix B: MedlinePlus / CDC /
   NIH / NHLBI / NIDDK / 988). The `provenance` field names the source whose criterion sets the
   escalation/triage. Numbers/clinical facts (BP ≥180/120, glucose <54, SpO₂ <92) aren't
   copyrightable; we encode the *number*, not copyrighted prose.
2. **Turns drafted to read like a real patient** (hedging, partial info, corrections) and then
   **every label human-reviewed** against the rule it targets. The labels are not a keyword dump.
3. **Cross-checked against the code, not just asserted.** `test_must_escalate_gate.py` runs each
   case's turns through the real `scribeintake.safety` gate (pure code, no LLM) and asserts the
   declared label is consistent: `must_escalate` → EMERGENCY, `benign`+`routine` → CLEAR,
   `urgent` → URGENT, gate-only `adversarial` → its declared level. A reviewer can therefore
   trust the labels are self-consistent with the engine, reproducibly and for free.
4. **`gold_soap` is intentionally loose** (chief complaint, a few HPI fields, band, `must_cite`).
   Exact SOAP-field match is a *distributional* metric the harness reports per-field (Split
   07/08), not an authored exact string — over-specifying would flake the eval on every prompt
   tweak. `must_cite: true` declares the harness should later verify each observation carries a
   real `chunk_id` (verified once RAG quality is judged, Splits 07/08).

## Freeze policy (the honesty story)

`must_escalate/` is the **deterministic gate's behavioural contract** and is **FROZEN**:

- Each file carries the header *"FROZEN — gate contract. Do not edit to make a build pass; a
  failure here is a code bug."*
- **Never tune the frozen set to make a failing build pass.** If a frozen case fails the gate
  cross-check, the bug is in the **code** (rules/extractor), not the case.
- If a case labelled `must_escalate` turns out **not** to fire at the code gate (e.g. the wording
  is too oblique for the regex), it does **not** belong in the frozen subset — move it to
  `adversarial/` with `escalation_source: [gate, agent]`. The frozen subset is *only* the
  code-deterministic cases; that boundary is the whole honesty story (spec §10/§15).

## Held-out slice (anti-overfit)

`heldout/` cases carry `heldout: true` and **must never be shown to the agent's prompt or
few-shots** (Splits 03/07 must exclude them). They declare their real `category:` explicitly
(`heldout` is a *flag*, not a category — the loader never derives `category` from the folder
name). They are a bounded anti-overfit check: the 58 cases are a test set, not the universe, and
the held-out slice plus the never-tuned frozen subset are what make the reported numbers
trustworthy.

## Schema & loading

`Scenario` (`eval/scenario.py`) reuses the engine's enums (`EscalationLevel`, `TriageBand`,
`EscalationSource`) so labels can't drift from the code's types; `extra="forbid"` makes typos
fail loudly. `load_scenarios(dir)` recurses, derives `category` from the parent folder when
absent (never for `heldout/`), forces `heldout=True` under `heldout/`, validates `id == filename
stem`, and **raises a clear error naming the offending file** on any failure.

```bash
python -m pytest eval/tests -m "not live"   # schema validity + gate cross-check (no API key)
```
