"""Build a single self-contained static HTML demo from data/demo_fixture.json.

Output: static_demo/index.html — no server, no dependencies, opens in any
browser. Reproduces the Avea-branded Compare results page (header, question
box, Run button that reveals the canned results via inline JS, ranked-picks
table with brand-match ticks, suggestions card, deeper-analysis section,
sources + full-answers detail blocks).

This is the artifact to zip and send: `zip -r demo.zip static_demo`.
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from surrogate.demo_render import (  # noqa: E402
    trajectory_inner_html, TRAJECTORY_CSS, TRAJECTORY_JS, OVERLAY_HTML,
    graph_panels, counterfactual_html, CHART_CSS,
)
FIXTURE = ROOT / "data/demo_fixture.json"
LOGO = ROOT / "static/avea_logo.png"
OUT_DIR = ROOT / "static_demo"


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def matched_set(pairs, idx):
    return {str(p[idx]).lower() for p in (pairs or []) if len(p) > idx}


def render_trajectory(traj) -> str:
    if not traj:
        return ""
    return (
        "<div class='traj'><h3>How our surrogate reached this</h3>"
        "<p class='cap'>Every step the model took: its live reasoning, and the "
        "action each thought triggered. <strong>Click any highlighted phrase</strong> "
        "to see the exact sources the model pulled at that step. "
        "The frontier models don't expose this.</p>"
        + trajectory_inner_html(traj) + "</div>"
    )


def build() -> str:
    rec = json.loads(FIXTURE.read_text())
    sysd = rec["systems"]
    m = rec["matches"]
    sug = rec["suggestions"]
    deep = rec.get("deep") or {}
    brand = rec["brand"]
    question = rec["question"]

    logo_b64 = base64.b64encode(LOGO.read_bytes()).decode() if LOGO.exists() else ""

    sur = sysd["surrogate"]["ranked"]
    oai = sysd["openai"]["ranked"]
    cla = sysd["claude"]["ranked"]
    oai_matched = matched_set(m["sur_openai"].get("matched_pairs"), 1)
    cla_matched = matched_set(m["sur_claude"].get("matched_pairs"), 1)

    def fr_cell(pick, matched):
        if not pick:
            return ""
        e = esc(pick)
        return f"<strong>{e}</strong> &#10003;" if pick.lower() in matched else e

    # ranked-picks table
    n = max(len(sur), len(oai), len(cla))
    rows = []
    for i in range(n):
        s_c = esc(sur[i]) if i < len(sur) else ""
        o_c = fr_cell(oai[i] if i < len(oai) else "", oai_matched)
        c_c = fr_cell(cla[i] if i < len(cla) else "", cla_matched)
        rows.append(f"<tr><td>{i+1}</td><td>{s_c}</td><td>{o_c}</td><td>{c_c}</td></tr>")
    picks_table = (
        "<table class='picks'><thead><tr><th>#</th><th>Surrogate</th>"
        "<th>ChatGPT</th><th>Claude</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    caption = (
        f"&#10003; = brand-level match with the surrogate's list &nbsp;·&nbsp; "
        f"Surrogate&#8596;ChatGPT {m['sur_openai']['overlap']}/{len(m['sur_openai']['a'])} "
        f"&nbsp;·&nbsp; Surrogate&#8596;Claude {m['sur_claude']['overlap']}/{len(m['sur_claude']['a'])} "
        f"&nbsp;·&nbsp; ChatGPT&#8596;Claude {m['openai_claude']['overlap']}/{len(m['openai_claude']['a'])}"
    )

    # brief suggestions card
    actions = "".join(f"<li>{esc(a)}</li>" for a in sug.get("actions", []))
    card = (
        f"<div class='avea-card'><h3>What {esc(brand)} should do</h3>"
        f"<p>{esc(sug.get('why',''))}</p><ul>{actions}</ul></div>"
    )

    # deeper analysis
    deep_html = ""
    if deep:
        gaps = deep.get("competitive_gaps") or []
        gap_rows = "".join(
            f"<tr><td><strong>{esc(g.get('asset'))}</strong></td>"
            f"<td>{esc(', '.join(g.get('brands_with_it') or []))}</td>"
            f"<td>{esc(g.get('why_it_matters'))}</td>"
            f"<td>{esc(g.get('gap_for_brand'))}</td></tr>"
            for g in gaps
        )
        gaps_tbl = (
            "<h4>What winning brands have that you don't</h4>"
            "<table class='picks'><thead><tr><th>Asset</th><th>Who has it</th>"
            "<th>Why it wins AI visibility</th><th>Your gap</th></tr></thead>"
            f"<tbody>{gap_rows}</tbody></table>"
        ) if gaps else ""

        plan = sorted(deep.get("priority_plan") or [], key=lambda x: x.get("rank", 99))
        plan_html = "".join(
            f"<p><strong>{esc(p.get('rank'))}.</strong> {esc(p.get('action'))}<br>"
            f"<em>{esc(p.get('horizon'))} · {esc(p.get('effort'))} effort · "
            f"{esc(p.get('impact'))}</em></p>"
            for p in plan
        )
        plan_html = f"<h4>Do this first, in priority order</h4>{plan_html}" if plan else ""

        rivals = deep.get("rival_deep_dive") or []
        rivals_html = "".join(
            f"<p><strong>{esc(r.get('brand'))}</strong><br>"
            f"<em>Why AI ranks them:</em> {esc(r.get('why_ai_ranks_them'))}<br>"
            f"<em>Their visible assets:</em> {esc(r.get('their_visible_assets'))}<br>"
            f"<em>How to compete:</em> {esc(r.get('how_to_compete'))}</p>"
            for r in rivals
        )
        rivals_html = f"<h4>Rival deep-dive</h4>{rivals_html}" if rivals else ""

        deep_html = (
            "<details class='block'><summary>Deeper suggestions: competitive "
            "gaps, priority plan, rival deep-dive</summary>"
            f"<p>{esc(deep.get('summary',''))}</p>{gaps_tbl}{plan_html}{rivals_html}"
            "</details>"
        )

    # sources consulted
    o_urls = sorted(set(sysd["openai"].get("urls") or []))
    c_urls = sorted(set(sysd["claude"].get("urls") or []))

    def _n_searches(sysrec):
        n = 0
        for tc in sysrec.get("tool_calls") or []:
            a = tc.get("action") or {}
            n += len(a.get("queries") or ([a["query"]] if a.get("query") else []))
            if tc.get("kind") in ("tool_use", "web_search_call"):
                n = max(n, 1)
        return n

    o_searches = _n_searches(sysd["openai"])
    # ChatGPT only exposes URLs it explicitly cites; note searches so "0 cited"
    # doesn't read as "didn't search".
    o_hdr = (f"ChatGPT ran {o_searches} web search(es) and cited "
             f"{len(o_urls)} URL(s)" if o_searches
             else f"ChatGPT cited {len(o_urls)} URL(s)")
    src = (
        "<details class='block'><summary>Sources each model consulted</summary>"
        f"<p class='cap'>Note: Claude's API exposes the pages its search returned; "
        f"ChatGPT's only exposes URLs it explicitly cites in its answer, so its "
        f"count can be 0 even when it searched.</p>"
        f"<p><strong>{o_hdr}:</strong></p><ul>"
        + "".join(f"<li><a href='{esc(u)}' target='_blank'>{esc(u)}</a></li>" for u in o_urls)
        + f"</ul><p><strong>Claude consulted {len(c_urls)} URL(s):</strong></p><ul>"
        + "".join(f"<li><a href='{esc(u)}' target='_blank'>{esc(u)}</a></li>" for u in c_urls)
        + "</ul></details>"
    )

    # full answers — for the surrogate, its "reasoning" is the interactive
    # step-by-step trajectory (click a phrase → sources). Frontiers show their
    # verbatim thinking since we can't bind theirs to sources.
    def ans_block(key, label):
        s = sysd[key]
        body = esc(s.get("answer") or "(empty)").replace("\n", "<br>")
        think = ""
        if key == "surrogate":
            traj = s.get("trajectory") or []
            if traj:
                think = (
                    "<p><strong>Reasoning, step by step, with sources.</strong> "
                    "<span class='cap'>Click any highlighted phrase to see the "
                    "exact sources that step pulled.</span></p>"
                    f"<div class='traj'>{trajectory_inner_html(traj)}</div>"
                )
            elif s.get("thinking"):
                think = ("<p><strong>Reasoning (verbatim):</strong></p>"
                         f"<pre>{esc(s['thinking'])}</pre>")
        elif s.get("thinking"):
            think = ("<p><strong>Reasoning (verbatim):</strong></p>"
                     f"<pre>{esc(s['thinking'])}</pre>")
        return f"<h4>{label} · {esc(s.get('model',''))}</h4><p>{body}</p>{think}"
    answers = (
        "<details class='block'><summary>Full answers &amp; reasoning (verbatim)</summary>"
        + ans_block("surrogate", "Surrogate")
        + ans_block("openai", "ChatGPT")
        + ans_block("claude", "Claude")
        + "</details>"
    )

    pills = "".join(
        f"<span class='pill done'>{lbl} · done</span>"
        for lbl in ["Surrogate", "ChatGPT", "Claude", "Match scoring", "Deep analysis"]
    )

    logo_tag = (f"<img src='data:image/png;base64,{logo_b64}' alt='AVEA'/>"
                if logo_b64 else "AVEA")

    # Graph panels (brand visibility + domain authority, plus any conditional
    # ones) sit right after the suggestions card, side by side, as evidence.
    _panels = graph_panels(rec)
    charts = f"<div class='charts-row'>{''.join(_panels)}</div>" if _panels else ""
    # before/after counterfactual panel (empty until a GPU run produces it)
    charts += counterfactual_html(rec)

    # The surrogate trajectory now lives inside the Full answers section (as the
    # surrogate's reasoning), so it's not a standalone block here.
    results = (
        f"<div class='pills'>{pills}</div>"
        f"{picks_table}<p class='cap'>{caption}</p>"
        f"{card}{charts}{deep_html}{src}{answers}"
    )

    return _TEMPLATE.format(
        question=esc(question),
        results=results,
        logo=logo_tag,
        traj_css=TRAJECTORY_CSS + CHART_CSS,
        traj_js=TRAJECTORY_JS,
        overlay=OVERLAY_HTML,
    )


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVEA · AI Visibility Analyzer</title>
<link href="https://fonts.googleapis.com/css2?family=Epilogue:wght@500;600;700&family=Mulish:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{ --teal:#2DA5B6; --teal-d:#238D9C; --ink:#1A1A1A; --line:#ECE9E4; --soft:#F8F7F5; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:'Mulish',sans-serif; color:var(--ink); background:#fff;
          margin:0; padding:4.2rem 2rem 4rem; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1,h2,h3,h4 {{ font-family:'Epilogue',sans-serif; letter-spacing:-0.01em; }}
  .header {{ display:flex; flex-direction:column; align-items:center; gap:1.2rem;
             margin:0 0 2.4rem; font-family:'Epilogue',sans-serif; font-weight:600;
             font-size:1.45rem; letter-spacing:.04em; }}
  .header img {{ height:2.4rem; }}
  .header .accent {{ color:var(--teal); }}
  label.q {{ font-size:1.12rem; font-weight:600; display:block; margin-bottom:.4rem; }}
  input[type=text] {{ width:100%; padding:.6rem .8rem; font-size:1rem; font-family:'Mulish',sans-serif;
                      border:1px solid #ccc; border-radius:8px; }}
  .btn {{ background:var(--teal); color:#fff; border:none; border-radius:999px;
          padding:.55rem 2rem; font-weight:700; font-size:1rem; cursor:pointer; margin-top:1rem; }}
  .btn:hover {{ background:var(--teal-d); }}
  .pills {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:1.5rem 0 1.2rem; }}
  .pill {{ border-radius:999px; padding:.2rem .85rem; font-size:.8rem; font-weight:700;
           background:#F0EEEA; color:#6B6B6B; }}
  .pill.done {{ background:rgba(45,165,182,.13); color:#1B7F8C; }}
  table.picks {{ width:100%; border-collapse:collapse; margin:.4rem 0; }}
  table.picks th {{ text-align:left; font-family:'Epilogue',sans-serif; font-size:.84rem;
                    letter-spacing:.05em; text-transform:uppercase; color:#333;
                    border-bottom:2px solid var(--teal); padding:.5rem .65rem; }}
  table.picks td {{ border-bottom:1px solid var(--line); padding:.48rem .65rem; font-size:.93rem;
                    vertical-align:top; }}
  table.picks tr:nth-child(even) td {{ background:var(--soft); }}
  .cap {{ color:#6B6B6B; font-size:.85rem; margin:.5rem 0 1.5rem; }}
  .avea-card {{ background:var(--soft); border-left:4px solid var(--teal); border-radius:10px;
               padding:1.1rem 1.4rem; margin:1.6rem 0; }}
  .avea-card h3 {{ margin:0 0 .6rem; font-size:1.4rem; }}
  .avea-card li {{ margin-bottom:.45rem; }}
  details.block {{ margin-top:1.2rem; border-top:1px solid var(--line); padding-top:1rem; }}
  details.block summary {{ font-size:1.05rem; font-weight:600; cursor:pointer; }}
  details.block h4 {{ font-size:1.18rem; margin:1.6rem 0 .6rem; }}
  pre {{ white-space:pre-wrap; background:var(--soft); padding:.8rem; border-radius:8px;
         font-size:.82rem; overflow-x:auto; }}
  a {{ color:var(--teal); word-break:break-all; }}
  #results {{ display:none; }}
  .traj {{ margin:1.8rem 0; }}
  .traj h3 {{ font-size:1.4rem; margin:0 0 .3rem; }}
  {traj_css}
</style></head>
<body><div class="wrap">
  <div class="header">{logo}<span class="accent">AI Visibility Analyzer</span></div>

  <label class="q" for="q">Your question</label>
  <input type="text" id="q" value="{question}" />
  <label class="q" for="b" style="margin-top:1rem;">Brand to track</label>
  <input type="text" id="b" value="Avea" />
  <button class="btn" type="button" id="runbtn">Run</button>

  <div id="results">{results}</div>
</div>
<script>
/* Run button: bound here in its own script so a later error can't disable it,
   and with a guarded plain scroll (iOS WebKit can throw on the options form). */
(function(){{
  var b=document.getElementById('runbtn');
  if(!b) return;
  b.addEventListener('click', function(){{
    var r=document.getElementById('results');
    if(r) r.style.display='block';
    try{{ b.scrollIntoView({{behavior:'smooth'}}); }}
    catch(e){{ try{{ b.scrollIntoView(); }}catch(_){{}} }}
  }});
}})();
</script>
{overlay}
<script>{traj_js}</script>
</body></html>
"""


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "index.html"
    out.write_text(build())
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    print(f"Zip it:  zip -r demo.zip static_demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
