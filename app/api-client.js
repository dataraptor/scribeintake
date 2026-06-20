/*
 * ScribeIntake — frontend API client (Split 11).
 *
 * The ONLY data layer for the connected frontend. It owns every call to the FastAPI service
 * (Split 10) and every rename from the API's JSON onto the component's existing view-model
 * (stripVM / vm / the summary + trace + proof shapes). The DC component imports nothing — it
 * reaches this module through the `window.SI_API` global (the DCLogic class is compiled with
 * `new Function`, so it resolves `SI_API` from the global scope).
 *
 * Centralising the renames here (not scattered through the class) is the whole point: Split 11 is
 * a data-layer swap, not a UI reshape. The frontend holds no offline/demo data — every value it
 * renders comes from these calls. If `window.SI_API` is absent the component shows an honest error
 * rather than fabricating a reply.
 */
(function () {
  "use strict";

  // --- config -------------------------------------------------------------------------------
  function qsParam(name) {
    try {
      return new URLSearchParams(window.location.search).get(name);
    } catch (e) {
      return null;
    }
  }
  function metaContent(name) {
    try {
      const el = document.querySelector('meta[name="' + name + '"]');
      return el ? el.getAttribute("content") : null;
    } catch (e) {
      return null;
    }
  }
  // Resolution order: explicit window.API_BASE (incl. "") > ?api= > <meta> > "" (same-origin).
  // The primary serve is FastAPI hosting both the page and the API, so same-origin relative
  // fetches are the safe default. For a split-origin dev setup pass ?api=http://localhost:8000.
  function resolveApiBase() {
    if (typeof window.API_BASE === "string") return window.API_BASE;
    const q = qsParam("api");
    if (q != null) return q;
    const m = metaContent("api-base");
    if (m != null) return m;
    return "";
  }
  const API_BASE = resolveApiBase();

  function url(path) {
    return API_BASE.replace(/\/$/, "") + path;
  }

  // --- network ------------------------------------------------------------------------------
  async function createSession() {
    const res = await fetch(url("/session"), { method: "POST" });
    if (!res.ok) throw new Error("createSession failed: " + res.status);
    const j = await res.json();
    return { sessionId: j.sessionId, disclaimer: j.disclaimer };
  }

  /*
   * POST a patient message and consume the SSE stream:
   *   onToken(textDelta)  per `token` frame (drives the existing typing/stream render)
   *   onTurn(turnResponse) on the terminal `turn` frame (the full TurnResponse JSON)
   *   onError({message})   on a network/stream/upstream error (never a blank failure, §18)
   * The browser EventSource API only does GET, so we read the POST response body with a
   * ReadableStream reader and parse the `event:`/`data:` frames by hand. If the server answers
   * with JSON instead of a stream (or streaming is unavailable) we fall back to the JSON path.
   */
  async function sendMessage(sessionId, text, handlers) {
    const onToken = (handlers && handlers.onToken) || function () {};
    const onTurn = (handlers && handlers.onTurn) || function () {};
    const onError = (handlers && handlers.onError) || function () {};
    let res;
    try {
      res = await fetch(url("/session/" + encodeURIComponent(sessionId) + "/message"), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ text: text }),
      });
    } catch (e) {
      onError({ message: reconnectMsg(), kind: "reconnect" });
      return;
    }
    if (!res.ok) {
      // 404 unknown session / 503 upstream: read the friendly detail, surface via onError.
      let detail = reconnectMsg();
      try {
        const j = await res.json();
        detail = j.detail || detail;
      } catch (e) {}
      onError({ message: detail, kind: res.status === 404 ? "not_found" : "reconnect" });
      return;
    }
    const ctype = res.headers.get("content-type") || "";
    if (ctype.indexOf("text/event-stream") === -1 || !res.body || !res.body.getReader) {
      // Non-streaming fallback: a single JSON TurnResponse. Chunk it client-side so the typing
      // render still animates, then deliver the turn.
      const j = await res.json();
      if (j && j.error) {
        onError({ message: j.detail || reconnectMsg(), kind: "reconnect" });
        return;
      }
      chunkText(j.content || "", onToken);
      onTurn(j);
      return;
    }
    await readSse(res.body, { onToken, onTurn, onError });
  }

  async function readSse(body, h) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let sep;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          dispatchFrame(frame, h);
        }
      }
      if (buf.trim()) dispatchFrame(buf, h);
    } catch (e) {
      h.onError({ message: reconnectMsg(), kind: "reconnect" });
    }
  }

  function dispatchFrame(frame, h) {
    let event = "message";
    const dataLines = [];
    frame.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) event = line.slice(6).trim();
      else if (line.indexOf("data:") === 0) dataLines.push(line.slice(5).trim());
    });
    if (!dataLines.length) return;
    let data;
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch (e) {
      return;
    }
    if (event === "token") h.onToken(data.text || "");
    else if (event === "turn") h.onTurn(data);
    else if (event === "error") h.onError({ message: data.message || reconnectMsg(), kind: data.kind });
  }

  function chunkText(text, onToken, size) {
    size = size || 24;
    for (let i = 0; i < text.length; i += size) onToken(text.slice(i, i + size));
  }

  async function getSummary(sessionId) {
    const res = await fetch(url("/session/" + encodeURIComponent(sessionId) + "/summary"));
    if (res.status === 404) {
      const j = await res.json().catch(function () {
        return {};
      });
      return { ok: false, detail: j.detail || "No summary yet — the intake is not complete." };
    }
    if (!res.ok) return { ok: false, detail: "Could not load the summary." };
    const soap = await res.json();
    return { ok: true, summary: toSummary(soap) };
  }

  async function getTrace(sessionId) {
    const res = await fetch(url("/session/" + encodeURIComponent(sessionId) + "/trace"));
    if (!res.ok) return { rows: [], traceCost: "", nTurns: 0 };
    const t = await res.json();
    return {
      rows: toTraceRows(t.rows || []),
      traceCost: t.traceCostLabel || "",
      nTurns: t.nTurns || 0,
      totalCostUsd: t.totalCostUsd || 0,
    };
  }

  // The committed proof artifacts (real leaderboard + cost report), served by the API under
  // /proof/* (see api/main.py). Returns the view-model-ready ldDet/ldDist, the leaderboard's own
  // framing sentence, an EVAL header derived from its meta, and the cache-aware traceCost label.
  // Everything here is the real artifact — there are no fabricated numbers.
  async function getProof() {
    const out = { ldDet: null, ldDist: null, traceCost: "", framing: "", evalLabel: "" };
    try {
      const lb = await fetch(url("/proof/leaderboard.json")).then(function (r) {
        return r.ok ? r.json() : null;
      });
      const board = toLeaderboard(lb);
      out.ldDet = board.ldDet;
      out.ldDist = board.ldDist;
      out.framing = board.framing;
      out.evalLabel = board.evalLabel;
    } catch (e) {}
    try {
      const cr = await fetch(url("/proof/cost_report.json")).then(function (r) {
        return r.ok ? r.json() : null;
      });
      if (cr && cr.trace_cost_label) out.traceCost = cr.trace_cost_label;
    } catch (e) {}
    return out;
  }

  // --- adapters (API JSON -> existing view-model) -------------------------------------------
  const DOT_MAP = {
    self_care: "#28A86A",
    gp_routine: "#17191E",
    gp_urgent: "#E8930C",
    ER: "#E5484D",
  };

  // The inline-strip object consumed by stripVM(): phase/expanded are set by the component.
  function stripFromTurn(turn) {
    const strip = turn.strip || {};
    const sv = (strip.signalsView || []).map(function (s) {
      return { name: s.name, mark: s.mark, style: "color:#17191E" };
    });
    return {
      level: strip.level || turn.level,
      agentNet: !!strip.agentNet,
      crisis: !!(strip.crisis || turn.crisis),
      ruleId: strip.ruleId || "—",
      ruleLevel: strip.ruleLevel || turn.level,
      ruleSource: strip.ruleSource || "",
      signalsView: sv,
      toolsNote: strip.toolsNote || "",
    };
  }

  // The emergency/crisis sheet object. Wording (kicker/heading/body/actions/caption) comes
  // verbatim from the API (the core templates); only the two display colours are derived.
  function toEmergency(em) {
    const crisis = !!em.crisis;
    return {
      crisis: crisis,
      kicker: em.kicker || "",
      heading: em.heading || "",
      body: em.body || "",
      note: em.note || "",
      hasNote: !!em.hasNote,
      actions: (em.actions || []).map(function (a) {
        return { label: a.label, href: a.href };
      }),
      caption: em.caption || "",
      headColor: crisis ? "#B5650B" : "#D92D20",
      actionColor: crisis ? "#fff" : "#E5484D",
    };
  }

  // The summary-sheet state object (mockup openSummary output shape).
  function toSummary(soap) {
    const subjective = (soap.subjective || []).map(function (r) {
      return { key: r.key, value: r.value, color: r.low ? "#B5650B" : "#17191E" };
    });
    const low = soap.low_confidence_fields || [];
    const band = soap.band || "gp_routine";
    return {
      band: band,
      bandDot: DOT_MAP[band] || "#17191E",
      subjective: subjective,
      low: low,
      lowText: low.join(", "),
      objective: soap.objective || "none reported",
      observations: (soap.observations || []).map(function (o) {
        return {
          text: o.text,
          cited: !!o.cited,
          uncited: !!o.uncited,
          source: o.source || "",
          chunk: o.chunk || "",
          url: o.url || "",
        };
      }),
      redChecked: soap.red_flags_checked || 0,
      redTriggered: soap.red_flags_triggered || 0,
    };
  }

  function toTraceRows(rows) {
    return rows.map(function (r) {
      return {
        tool: r.tool,
        model: r.model || "—",
        latency: fmtLatency(r.latencyMs),
        cost: r.costUsd || 0,
        local: !!r.local,
        event: !!r.event,
      };
    });
  }

  function fmtLatency(ms) {
    if (ms == null) return "—";
    if (ms >= 1000) return (ms / 1000).toFixed(1) + "s";
    return Math.round(ms) + "ms";
  }

  // The real leaderboard.json (Split 07/08) -> the proof-tab view-model. Prefers the artifact's own
  // precomputed ldDet/ldDist arrays (which already carry the honest values incl. "pending" and a
  // null spark); falls back to deriving them from metrics[]. Also surfaces the artifact's `framing`
  // sentence and an EVAL header built from its meta — no hardcoded scenario counts or scores.
  function toLeaderboard(lb) {
    const empty = { ldDet: null, ldDist: null, framing: "", evalLabel: "" };
    if (!lb) return empty;
    let ldDet = Array.isArray(lb.ldDet)
      ? lb.ldDet.map(function (r) {
          return { label: r.label, value: r.value };
        })
      : null;
    let ldDist = Array.isArray(lb.ldDist)
      ? lb.ldDist.map(function (r) {
          return { label: r.label, value: r.value, spark: r.spark || "" };
        })
      : null;
    if ((!ldDet || !ldDist) && Array.isArray(lb.metrics)) {
      const det = [];
      const dist = [];
      lb.metrics.forEach(function (m) {
        if (m.group === "deterministic") det.push({ label: m.label, value: m.display });
        else if (m.group === "distributional")
          dist.push({ label: m.label, value: m.display, spark: m.spark || "" });
      });
      ldDet = ldDet || (det.length ? det : null);
      ldDist = ldDist || (dist.length ? dist : null);
    }
    const meta = lb.meta || {};
    let evalLabel = "";
    if (meta.scenario_count != null) {
      const runs = meta.n_runs != null ? meta.n_runs : 0;
      evalLabel = "EVAL · " + meta.scenario_count + " SCENARIOS · " + runs + "× RUNS";
    }
    return {
      ldDet: ldDet && ldDet.length ? ldDet : null,
      ldDist: ldDist && ldDist.length ? ldDist : null,
      framing: lb.framing || "",
      evalLabel: evalLabel,
    };
  }

  function reconnectMsg() {
    return "We hit a snag reaching the assistant. Your information is saved — please send that again.";
  }

  // --- public surface -----------------------------------------------------------------------
  window.SI_API = {
    API_BASE: API_BASE,
    createSession: createSession,
    sendMessage: sendMessage,
    getSummary: getSummary,
    getTrace: getTrace,
    getProof: getProof,
    // adapters (also exported for tests / reuse)
    stripFromTurn: stripFromTurn,
    toEmergency: toEmergency,
    toSummary: toSummary,
    toTraceRows: toTraceRows,
    toLeaderboard: toLeaderboard,
  };
})();
