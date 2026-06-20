"""A tiny **static** trace/cost dashboard (Split 09, spec section 16).

Renders a self-contained HTML page (inline CSS bars, no JS, no service) over a
:class:`~observability.models.CostReport`: cache savings (with vs without), the cache-read
share, latency percentiles vs targets, a tool-usage histogram, and — when present — the fleet
$/session trend. It reads existing data only; resisting a live metrics service is the point.
"""

from __future__ import annotations

from pathlib import Path

from .models import CostReport

_CSS = """
:root{--bg:#0f1115;--card:#181b22;--ink:#e7e9ee;--mut:#9aa1ad;--ok:#1f8a54;--warn:#E5484D;
--bar:#3b82f6;--bar2:#5c616b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.card{background:var(--card);border-radius:12px;padding:16px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);
margin:0 0 12px}
.big{font-size:28px;font-weight:700}.row{display:flex;justify-content:space-between;margin:6px 0}
.bar{height:10px;border-radius:6px;background:var(--bar)}
.track{background:#23262e;border-radius:6px;overflow:hidden;margin:4px 0 10px}
.lbl{color:var(--mut)}.ok{color:var(--ok)}.warn{color:var(--warn)}.mono{font-family:ui-monospace,monospace}
"""


def _bar(value: float, maximum: float, color: str = "var(--bar)") -> str:
    pct = 0.0 if maximum <= 0 else min(100.0, 100.0 * value / maximum)
    return (
        f'<div class="track"><div class="bar" '
        f'style="width:{pct:.1f}%;background:{color}"></div></div>'
    )


def render_html(r: CostReport) -> str:
    """Return the dashboard HTML for a cost report."""
    s = r.savings
    lat = r.latency
    saved_pct = s.pct_saved * 100
    hit_pct = r.cache_hit_rate * 100

    # cost card
    cost_card = (
        f'<div class="card"><h2>$/session — cache savings</h2>'
        f'<div class="big">{saved_pct:.0f}% saved</div>'
        f'<div class="row"><span class="lbl">with caching</span>'
        f'<span class="mono">${s.with_cache_usd:.4f}</span></div>'
        f'{_bar(s.with_cache_usd, s.no_cache_usd, "var(--ok)")}'
        f'<div class="row"><span class="lbl">without caching</span>'
        f'<span class="mono">${s.no_cache_usd:.4f}</span></div>'
        f'{_bar(s.no_cache_usd, s.no_cache_usd, "var(--bar2)")}'
        f'<div class="row"><span class="lbl">cache-read share</span>'
        f'<span class="mono">{hit_pct:.0f}%</span></div></div>'
    )

    # latency card
    def _lat_row(name: str, p50: float, p95: float, target: int) -> str:
        cls = "warn" if p95 > target > 0 else "ok"
        return (
            f'<div class="row"><span class="lbl">{name}</span>'
            f'<span class="mono {cls}">p50 {p50:.0f} · p95 {p95:.0f} ms</span></div>'
            f'{_bar(p95, max(target, p95), "var(--warn)" if p95 > target > 0 else "var(--bar)")}'
        )

    lat_card = (
        '<div class="card"><h2>Latency p50/p95 (§18)</h2>'
        + _lat_row("intake/turn", lat.intake_p50_ms, lat.intake_p95_ms,
                   lat.targets.get("intake_p95_ms", 0))
        + _lat_row("summary call", lat.summary_p50_ms, lat.summary_p95_ms,
                   lat.targets.get("summary_ms", 0))
        + (f'<div class="row"><span class="lbl">first summary (compile)</span>'
           f'<span class="mono">{lat.first_summary_ms:.0f} ms</span></div>'
           if lat.first_summary_ms is not None else "")
        + "</div>"
    )

    # tool-usage histogram
    max_tool = max(r.tool_usage.values(), default=1)
    tool_rows = "".join(
        f'<div class="row"><span class="lbl mono">{t}</span><span class="mono">{n}</span></div>'
        f'{_bar(n, max_tool)}'
        for t, n in sorted(r.tool_usage.items(), key=lambda kv: -kv[1])
    )
    tool_card = f'<div class="card"><h2>Tool usage</h2>{tool_rows}</div>'

    # fleet trend
    fleet_card = ""
    if r.fleet and r.fleet.per_session_cost_usd:
        mx = max(r.fleet.per_session_cost_usd)
        bars = "".join(
            f'<span title="${c:.4f}" style="display:inline-block;width:8px;margin-right:2px;'
            f'background:var(--bar);height:{(0 if mx <= 0 else 60 * c / mx):.0f}px;'
            'vertical-align:bottom"></span>'
            for c in r.fleet.per_session_cost_usd
        )
        fleet_card = (
            f'<div class="card"><h2>Fleet $/session trend</h2>'
            f'<div style="height:64px">{bars}</div>'
            f'<div class="row"><span class="lbl">mean $/session</span>'
            f'<span class="mono">${r.fleet.mean_cost_usd:.4f}</span></div>'
            f'<div class="row"><span class="lbl">sessions</span>'
            f'<span class="mono">{r.fleet.n_sessions}</span></div></div>'
        )

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>ScribeIntake — Cost & Observability</title><style>{_CSS}</style></head><body>"
        "<h1>ScribeIntake — Cost &amp; Observability</h1>"
        f"<p class='sub mono'>{r.trace_cost_label} &middot; source <b>{r.source}</b> "
        f"&middot; {r.generated_at}</p>"
        f"<div class='grid'>{cost_card}{lat_card}{tool_card}{fleet_card}</div>"
        "</body></html>"
    )


def write_dashboard(report: CostReport, out_path: str | Path) -> Path:
    """Write the static dashboard HTML; returns the path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path
