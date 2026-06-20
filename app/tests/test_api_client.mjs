/*
 * Deterministic unit tests for app/api-client.js (Split 11).
 *
 * No browser, no network: we shim `window`/`document`/`fetch`, load the IIFE, then exercise the
 * pure response->view-model adapters (the renames the component depends on) plus the SSE frame
 * parser via a fake streamed Response. Run: `node app/tests/test_api_client.mjs`.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import assert from "node:assert/strict";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(path.join(__dirname, "..", "api-client.js"), "utf8");

let passed = 0;
const cases = [];
function test(name, fn) {
  cases.push({ name, fn });
}

// --- minimal DOM/window shim + loader --------------------------------------------------------
function loadClient({ apiBase, search = "", fetchImpl } = {}) {
  const win = { location: { search } };
  if (typeof apiBase === "string") win.API_BASE = apiBase;
  const doc = { querySelector: () => null };
  const g = {
    window: win,
    document: doc,
    location: { search },
    URLSearchParams,
    fetch: fetchImpl || (async () => ({ ok: false, status: 500 })),
    TextDecoder,
    JSON,
    Math,
    encodeURIComponent,
  };
  // The IIFE references bare globals (window/document/location/fetch). Run it with those in scope.
  const fn = new Function(
    "window",
    "document",
    "location",
    "fetch",
    "URLSearchParams",
    "TextDecoder",
    SRC
  );
  fn(win, doc, g.location, g.fetch, URLSearchParams, TextDecoder);
  return win.SI_API;
}

// --- config ----------------------------------------------------------------------------------
test("API_BASE: explicit window.API_BASE wins (incl. empty string = same-origin)", () => {
  assert.equal(loadClient({ apiBase: "" }).API_BASE, "");
  assert.equal(loadClient({ apiBase: "http://x:8000" }).API_BASE, "http://x:8000");
});

test("API_BASE: ?api= overrides when window.API_BASE unset", () => {
  assert.equal(loadClient({ search: "?api=http://h:9000" }).API_BASE, "http://h:9000");
});

test("no DEMO_MODE: the client never exposes an offline/demo switch", () => {
  assert.equal("DEMO_MODE" in loadClient({ apiBase: "", search: "?demo=1" }), false);
  assert.equal(loadClient({ apiBase: "" }).DEMO_MODE, undefined);
});

// --- adapters --------------------------------------------------------------------------------
const API = loadClient({ apiBase: "" });

test("stripFromTurn: maps strip fields + styles signals (present-only)", () => {
  const turn = {
    level: "EMERGENCY",
    crisis: false,
    strip: {
      level: "EMERGENCY",
      agentNet: false,
      crisis: false,
      ruleId: "acs_chest_pain",
      ruleLevel: "EMERGENCY",
      ruleSource: "gate",
      signalsView: [{ name: "chest_pain", mark: "✓" }, { name: "sbp", mark: "186" }],
      toolsNote: "gate short-circuited — agent never ran",
    },
  };
  const s = API.stripFromTurn(turn);
  assert.equal(s.ruleId, "acs_chest_pain");
  assert.equal(s.level, "EMERGENCY");
  assert.equal(s.toolsNote, "gate short-circuited — agent never ran");
  assert.equal(s.signalsView.length, 2);
  assert.ok(s.signalsView[0].style.includes("color")); // each signal gets a render style
  assert.equal(s.signalsView[0].name, "chest_pain");
});

test("toEmergency: keeps backend wording verbatim, derives only the two colours", () => {
  const em = API.toEmergency({
    crisis: false,
    kicker: "Possible emergency",
    heading: "This may be a medical emergency",
    body: "Call 911 now.",
    hasNote: false,
    actions: [{ label: "Call 911", href: "tel:911" }],
    caption: "Deterministic rule · acs_chest_pain · no model call · 0 missed on frozen set",
  });
  assert.equal(em.heading, "This may be a medical emergency");
  assert.equal(em.actions[0].href, "tel:911");
  assert.equal(em.headColor, "#D92D20"); // emergency colour
  assert.equal(em.actionColor, "#E5484D");
  // crisis variant flips colours but keeps wording
  const crisis = API.toEmergency({ crisis: true, heading: "You deserve support", actions: [] });
  assert.equal(crisis.headColor, "#B5650B");
  assert.equal(crisis.actionColor, "#fff");
});

test("toSummary: maps SOAP keys onto the summary view-model", () => {
  const soap = {
    band: "gp_urgent",
    subjective: [
      { key: "chief_complaint", value: "chest tightness", low: false },
      { key: "hpi.severity", value: "4/10", low: true },
    ],
    objective: "BP 186/122 (home)",
    observations: [
      { text: "Same-day eval advised.", cited: true, uncited: false, source: "MedlinePlus", chunk: "chk_1", url: "u" },
      { text: "No red flags.", cited: false, uncited: true },
    ],
    low_confidence_fields: ["hpi.severity"],
    red_flags_checked: 21,
    red_flags_triggered: 0,
  };
  const s = API.toSummary(soap);
  assert.equal(s.band, "gp_urgent");
  assert.equal(s.bandDot, "#E8930C");
  assert.equal(s.subjective[1].color, "#B5650B"); // low-confidence row tinted
  assert.equal(s.lowText, "hpi.severity");
  assert.equal(s.observations[0].cited, true);
  assert.equal(s.observations[1].uncited, true);
  assert.equal(s.redChecked, 21);
  assert.equal(s.redTriggered, 0);
});

test("toTraceRows: formats latency + carries cost/event/local", () => {
  const rows = API.toTraceRows([
    { tool: "build_summary", model: "gpt-5.5", latencyMs: 4200, costUsd: 0.0055, local: false, event: false },
    { tool: "retrieve_guideline", model: null, latencyMs: 5, costUsd: 0, local: true, event: false },
    { tool: "safety_event · gate", model: null, latencyMs: 1, costUsd: 0, local: true, event: true },
  ]);
  assert.equal(rows[0].latency, "4.2s");
  assert.equal(rows[1].latency, "5ms");
  assert.equal(rows[1].model, "—");
  assert.equal(rows[2].event, true);
  assert.equal(rows[0].cost, 0.0055);
});

test("toLeaderboard: derives ldDet/ldDist from metrics[] without fabricating a sparkline", () => {
  const lb = {
    metrics: [
      { label: "Rule correctness", group: "deterministic", display: "100%" },
      { label: "Frozen must-escalate", group: "deterministic", display: "0 miss" },
      { label: "E2E recall", group: "distributional", display: "pending" },
    ],
  };
  const out = API.toLeaderboard(lb);
  assert.equal(out.ldDet.length, 2);
  assert.equal(out.ldDet[0].value, "100%");
  assert.equal(out.ldDist.length, 1);
  // No fake sparkline: a metric with no `spark` of its own renders an empty one (honest "pending").
  assert.equal(out.ldDist[0].spark, "");
  assert.deepEqual(API.toLeaderboard(null), {
    ldDet: null,
    ldDist: null,
    framing: "",
    evalLabel: "",
  });
});

test("toLeaderboard: prefers the artifact's precomputed arrays + surfaces framing/evalLabel", () => {
  const lb = {
    meta: { scenario_count: 65, n_runs: 0 },
    framing: "0 missed on the frozen set; end-to-end recall pending.",
    ldDet: [{ label: "Rule correctness", value: "100%" }],
    ldDist: [{ label: "E2E recall", value: "pending", spark: null }],
  };
  const out = API.toLeaderboard(lb);
  assert.equal(out.ldDet[0].value, "100%");
  assert.equal(out.ldDist[0].value, "pending");
  assert.equal(out.ldDist[0].spark, ""); // null spark normalised to "" (no fabricated bars)
  assert.equal(out.framing, lb.framing);
  assert.equal(out.evalLabel, "EVAL · 65 SCENARIOS · 0× RUNS");
});

// --- SSE parsing via sendMessage with a fake streamed Response --------------------------------
function streamResponse(frames, { contentType = "text/event-stream" } = {}) {
  const enc = new TextEncoder();
  let i = 0;
  return {
    ok: true,
    status: 200,
    headers: { get: (k) => (k.toLowerCase() === "content-type" ? contentType : null) },
    body: {
      getReader: () => ({
        read: async () => {
          if (i < frames.length) return { value: enc.encode(frames[i++]), done: false };
          return { value: undefined, done: true };
        },
      }),
    },
  };
}

test("sendMessage: parses token frames then a terminal turn frame", async () => {
  const frames = [
    "event: token\ndata: " + JSON.stringify({ text: "Hello " }) + "\n\n",
    "event: token\ndata: " + JSON.stringify({ text: "world" }) + "\n\n",
    "event: turn\ndata: " + JSON.stringify({ content: "Hello world", level: "CLEAR" }) + "\n\n",
  ];
  const api = loadClient({ apiBase: "", fetchImpl: async () => streamResponse(frames) });
  const tokens = [];
  let turn = null;
  await api.sendMessage("sess1", "hi", { onToken: (t) => tokens.push(t), onTurn: (x) => (turn = x) });
  assert.deepEqual(tokens, ["Hello ", "world"]);
  assert.equal(turn.content, "Hello world");
});

test("sendMessage: non-stream JSON fallback still chunks + delivers the turn", async () => {
  const json = { content: "abcdefghijklmnopqrstuvwxyz0123456789", level: "CLEAR" };
  const resp = {
    ok: true,
    status: 200,
    headers: { get: () => "application/json" },
    json: async () => json,
  };
  const api = loadClient({ apiBase: "", fetchImpl: async () => resp });
  const tokens = [];
  let turn = null;
  await api.sendMessage("s", "hi", { onToken: (t) => tokens.push(t), onTurn: (x) => (turn = x) });
  assert.ok(tokens.length >= 2); // chunked client-side
  assert.equal(tokens.join(""), json.content);
  assert.equal(turn.level, "CLEAR");
});

test("sendMessage: error frame routes to onError (never a blank failure)", async () => {
  const frames = ["event: error\ndata: " + JSON.stringify({ message: "snag", kind: "reconnect" }) + "\n\n"];
  const api = loadClient({ apiBase: "", fetchImpl: async () => streamResponse(frames) });
  let err = null;
  await api.sendMessage("s", "hi", { onError: (e) => (err = e) });
  assert.equal(err.message, "snag");
});

test("sendMessage: non-ok status surfaces friendly detail via onError", async () => {
  const resp = { ok: false, status: 404, json: async () => ({ error: "session_not_found", detail: "No session." }) };
  const api = loadClient({ apiBase: "", fetchImpl: async () => resp });
  let err = null;
  await api.sendMessage("s", "hi", { onError: (e) => (err = e) });
  assert.equal(err.message, "No session.");
  assert.equal(err.kind, "not_found");
});

// run every test (sync or async) sequentially, await results, then report
(async () => {
  for (const { name, fn } of cases) {
    try {
      await fn();
      passed++;
      console.log("  ok  -", name);
    } catch (e) {
      console.error("  FAIL-", name, "\n", e && e.message ? e.message : e);
      process.exitCode = 1;
    }
  }
  console.log(`\n${passed}/${cases.length} checks passed`);
  if (passed !== cases.length) process.exitCode = 1;
})();
