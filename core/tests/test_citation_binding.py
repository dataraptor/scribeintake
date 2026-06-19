"""Citation binding (Split 05 §4, acceptance #3) — real chunk_id or `uncited`, never fabricated.

The Opus/structured call is mocked to return observations with empty citations; code then binds
each grounded observation to a real retrieved ``chunk_id`` and flags the rest ``uncited``
(``citation = None``) — with **no** fabricated source ever appearing.
"""

from __future__ import annotations

import json

from fakes import FakeLLMClient, FakeStructuredClient, text_response, tool_response

from scribeintake.agent import AgentLoop
from scribeintake.models import (
    SOAP,
    Citation,
    IntakeState,
    Observation,
    RetrievedChunk,
    SlotValue,
    Subjective,
    Triage,
    TriageBand,
)
from scribeintake.orchestrator import run_turn
from scribeintake.rag.embed import HashingEmbedder
from scribeintake.rag.rerank import LexicalReranker
from scribeintake.rag.retrieve import ChunkRecord, build_retriever
from scribeintake.tools import default_registry
from scribeintake.tools.build_summary import (
    bind_observation_citations,
    bind_triage_citation,
    build_summary,
)
from scribeintake.tools.suggest_triage import TriageSuggestion

CARDIAC = RetrievedChunk(
    chunk_id="chk_cardiac",
    text="Chest pain that spreads to the arm or jaw with sweating may be a heart attack. "
    "Call 911 right away.",
    source="MedlinePlus",
    url="https://medlineplus.gov/heartattack.html",
    score=0.9,
)
ER_PAGE = RetrievedChunk(
    chunk_id="chk_er",
    text="Go to the emergency department for serious problems such as severe chest pain, "
    "trouble breathing, or a sudden severe headache.",
    source="MedlinePlus",
    url="https://medlineplus.gov/ency/patientinstructions/000593.htm",
    score=0.8,
)

GROUNDED = "New exertional chest pain spreading to the arm with sweating; call 911."
SCREENING = "Acute coronary red-flags screened; none triggered this session."


def _state() -> IntakeState:
    st = IntakeState(session_id="s")
    st.slots["chief_complaint"] = SlotValue(value="chest tightness")
    return st


# --------------------------------------------------------------- unit: binding fn
def test_grounded_observation_binds_uncited_stays_none():
    soap = SOAP(
        observations=[
            Observation(text=GROUNDED, citation=Citation()),  # model emitted empty citation
            Observation(text=SCREENING, citation=Citation()),
        ]
    )
    bind_observation_citations(soap, [CARDIAC])

    grounded, screening = soap.observations
    assert grounded.citation is not None
    assert grounded.citation.chunk_id == "chk_cardiac"
    assert grounded.citation.source == "MedlinePlus" and grounded.citation.url
    # the generic screening note shares no specific terms → stays uncited, no fake source
    assert screening.citation is None


def test_no_chunks_normalises_all_to_uncited():
    soap = SOAP(observations=[Observation(text=GROUNDED, citation=Citation(source="x"))])
    bind_observation_citations(soap, [])
    assert soap.observations[0].citation is None  # never an empty/fake citation


def test_triage_rationale_citation_binds_or_empty():
    t = Triage(rationale="Severe chest pain — go to the emergency department now.")
    bind_triage_citation(t, [ER_PAGE, CARDIAC])
    assert len(t.citations) == 1
    assert t.citations[0].chunk_id in {"chk_er", "chk_cardiac"}

    t2 = Triage(rationale="Routine follow-up for a stable, mild concern.")
    bind_triage_citation(t2, [CARDIAC])
    assert t2.citations == []  # nothing supports it → no fabricated source


# --------------------------------------------------- integration: build_summary binds
def test_build_summary_binds_observations_from_chunks():
    model_soap = SOAP(
        subjective=Subjective(chief_complaint="chest tightness"),
        observations=[Observation(text=GROUNDED), Observation(text=SCREENING)],
    )
    client = FakeStructuredClient({"SOAP": model_soap})

    res = build_summary(_state(), client=client, generated_at="t", chunks=[CARDIAC, ER_PAGE])

    cited = [o for o in res.soap.observations if o.citation is not None]
    uncited = [o for o in res.soap.observations if o.citation is None]
    assert len(cited) == 1 and cited[0].citation.chunk_id == "chk_cardiac"
    assert len(uncited) == 1  # the screening note
    # the retrieved passages were shown to the model
    assert "Reference guidance" in client.calls[0]["messages"][0]["content"]
    # no observation carries an empty/blank fabricated citation
    for o in res.soap.observations:
        assert o.citation is None or (o.citation.chunk_id and o.citation.source)


# ----------------------------------- orchestrator e2e: retriever → tool ($0) + bound SOAP
def _kb_retriever():
    recs = [CARDIAC_REC, ER_REC]
    emb = HashingEmbedder()
    return build_retriever(
        recs,
        embedder=emb,
        reranker=LexicalReranker(),
        dense_vectors=emb.embed_documents([r.text for r in recs]),
    )


CARDIAC_REC = ChunkRecord(
    chunk_id="chk_cardiac",
    text="Chest pain that spreads to the arm or jaw with sweating may be a heart attack. "
    "Call 911 right away.",
    source="MedlinePlus",
    url="https://medlineplus.gov/heartattack.html",
)
ER_REC = ChunkRecord(
    chunk_id="chk_er",
    text="Go to the emergency department for serious problems such as severe chest pain, "
    "trouble breathing, or a sudden severe headache.",
    source="MedlinePlus",
    url="https://medlineplus.gov/ency/patientinstructions/000593.htm",
)

_CHEST_FILL = [
    {"slot": "chief_complaint", "value": "chest tightness", "confidence": "high"},
    {"slot": "hpi.onset", "value": "this morning", "confidence": "high"},
    {"slot": "hpi.severity", "value": "4/10", "confidence": "high"},
    {"slot": "hpi.radiation", "value": "spreads to the left arm", "confidence": "high"},
    {"slot": "medications", "value": "none", "confidence": "high"},
    {"slot": "allergies", "value": "none", "confidence": "high"},
]


def test_run_turn_routes_retriever_to_tool_and_binds_soap(conn, session):
    # The agent records all slots AND calls retrieve_guideline in one turn; completion follows.
    agent = AgentLoop(
        FakeLLMClient(
            [
                tool_response(
                    [
                        ("record_intake", {"updates": _CHEST_FILL}),
                        ("retrieve_guideline", {"query": "chest pain arm", "k": 3}),
                    ]
                ),
                text_response("Anything else?"),
            ]
        ),
        default_registry(),
    )
    model_soap = SOAP(
        subjective=Subjective(chief_complaint="chest tightness"),
        observations=[
            Observation(
                text="New exertional chest pain spreading to the arm; same-day clinician "
                "evaluation advised. Call 911 if it worsens."
            ),
            Observation(text="Acute coronary red-flags screened; none triggered this session."),
        ],
    )
    client = FakeStructuredClient(
        {"SOAP": model_soap, "TriageSuggestion": TriageSuggestion(band=TriageBand.gp_urgent)}
    )

    turn = run_turn(
        session,
        "I have some chest tightness this morning.",
        conn=conn,
        agent=agent,
        summary_client=client,
        retriever=_kb_retriever(),
    )

    assert turn.status == "completed"
    # the SOAP carries at least one real bound chunk_id, and no fabricated source
    obs = turn.soap["observations"]
    cited = [o for o in obs if o["citation"] is not None]
    assert cited and cited[0]["citation"]["chunk_id"].startswith("chk_")
    assert all(o["citation"] is None or o["citation"]["source"] for o in obs)

    # retrieve_guideline ran as a local tool: real chunk_ids in its result, cost $0
    row = conn.execute(
        "SELECT result_json, cost_usd, model FROM tool_calls "
        "WHERE session_id = ? AND tool = 'retrieve_guideline'",
        (session,),
    ).fetchone()
    assert row is not None
    assert row["cost_usd"] == 0.0 and row["model"] is None
    chunks = json.loads(row["result_json"])["chunks"]
    assert chunks and chunks[0]["chunk_id"].startswith("chk_")
