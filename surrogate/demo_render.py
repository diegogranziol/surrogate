"""Shared renderer for the surrogate transparency timeline.

Both the static HTML demo (scripts/build_static_demo.py) and the Streamlit app
use this so the two stay identical. Clicking a highlighted action-phrase opens
a centered modal (blurred backdrop) showing that step's exact sources — no
inline reflow.

Exports:
  trajectory_inner_html(traj)     -> the timeline items + hidden source data
  OVERLAY_HTML                    -> the modal/overlay container (include once)
  TRAJECTORY_CSS / TRAJECTORY_JS  -> styles + the openSrc/closeOv handlers
  trajectory_component_html(traj) -> standalone doc for st.components.v1.html
"""
from __future__ import annotations

import html
import re

_CUE = re.compile(
    r"(I(?:'ll| will| need to| should| am going to)?\s+"
    r"(?:search|look up|look for|find|check|verify|read|extract|gather)\w*"
    r"|let me (?:search|look|check|verify|find|gather)\w*"
    r"|searching|looking up)",
    re.I,
)

# Per tool, the cue verbs that genuinely describe THAT step's action — used to
# pick the single most-relevant phrase to highlight (a step has one source set,
# so one trigger; multiple highlights that open the same thing mislead).
_TOOL_VERBS = {
    "search": ("search", "look for", "look up", "find", "searching"),
    "extract_entity": ("read", "look", "check", "extract", "examine", "gather"),
    "fetch_url": ("read", "look", "fetch", "open"),
    "verify_fact": ("verify", "check", "confirm"),
}


def _highlight_one_cue(esc_text: str, tool: str, pid: str) -> tuple[str, bool]:
    """Wrap exactly ONE cue phrase (the one best matching this tool's action)
    as the clickable trigger. Returns (html, matched?)."""
    matches = list(_CUE.finditer(esc_text))
    if not matches:
        return esc_text, False
    verbs = _TOOL_VERBS.get(tool, ())
    target = next((m for m in matches
                   if any(v in m.group(0).lower() for v in verbs)), matches[0])
    span = (f"<span class='cue click' onclick=\"openSrc('{pid}',event)\" "
            f"title='click to see the sources for this step'>{target.group(0)}</span>")
    return esc_text[:target.start()] + span + esc_text[target.end():], True
_TOOL_ICON = {"search": "&#128269;", "extract_entity": "&#127991;",
              "fetch_url": "&#128196;", "verify_fact": "&#10003;",
              "check_missing_fields": "&#128203;", "think": "&#128173;",
              "stop_and_answer": "&#9989;"}


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _fact_chip(f: dict) -> str:
    """One extracted entity -> a compact 'Type: name · ★rating (n) · price' chip."""
    bits = []
    name = (f.get("name") or "").strip()
    if name:
        bits.append(_esc(name[:70]))
    rating = f.get("rating")
    if rating:
        rc = f.get("review_count")
        bits.append(f"&#9733;{_esc(rating)}" + (f" ({_esc(rc)})" if rc else ""))
    price, cur, pr = f.get("price"), f.get("currency"), f.get("price_range")
    if price:
        bits.append((f"{_esc(cur)} " if cur else "") + _esc(price))
    elif pr:
        bits.append(_esc(pr))
    t = _esc(f.get("type") or "")
    label = (f"{t}: " if t else "") + " · ".join(bits) if bits else t
    return f"<span class='evfact'>{label}</span>" if label else ""


def _evidence_html(step: dict) -> str:
    """Inline evidence for a step: extracted structured facts, or a verdict."""
    tool = step.get("tool")
    if tool == "extract_entity":
        chips = "".join(_fact_chip(f) for f in (step.get("facts") or []))
        if not chips and step.get("og_title"):
            chips = f"<span class='evfact'>page: {_esc(step['og_title'][:90])}</span>"
        if chips:
            return f"<div class='ev'><span class='evk'>extracted</span>{chips}</div>"
    elif tool == "verify_fact":
        v = step.get("verify") or {}
        verdict = (v.get("verdict") or "").lower()
        if verdict:
            label = {"yes": "supported", "partial": "partially supported",
                     "no": "not supported"}.get(verdict, verdict)
            cls = {"yes": "vf-yes", "partial": "vf-partial", "no": "vf-no"}.get(verdict, "")
            mark = {"yes": "&#10003;", "partial": "&#126;", "no": "&#10007;"}.get(verdict, "")
            claim = _esc((v.get("claim") or "")[:90])
            return (f"<div class='vf {cls}'>{mark} verified &ldquo;{claim}&rdquo;: "
                    f"{label}</div>")
    return ""


def trajectory_inner_html(traj: list[dict]) -> str:
    """Timeline items (no wrapper, no <style>/<script>). Each step that pulled
    sources holds a hidden `.srcdata` div the modal reads on click."""
    if not traj:
        return ""
    items = []
    for s in traj:
        step = s.get("step", 0)
        pid = f"src{step}"
        urls = s.get("urls") or []
        icon = _TOOL_ICON.get(s.get("tool"), "&#8226;")
        action = _esc(s.get("action") or "")

        # Only steps that pulled sources get a (single) clickable highlight;
        # steps without sources stay plain to avoid implying interactivity.
        think = _esc(s.get("think") or "")
        matched = False
        if urls:
            think, matched = _highlight_one_cue(think, s.get("tool"), pid)
        think = think.replace("\n\n", "</p><p>").replace("\n", "<br>")

        tail, data = "", ""
        if urls:
            if not matched:
                tail = (f" <span class='cue click' onclick=\"openSrc('{pid}',event)\" "
                        f"title='click to see the sources for this step'>"
                        f"{icon} {action} &#9656;</span>")
            links = "".join(
                f"<li><a href='{_esc(u)}' target='_blank' rel='noopener'>{_esc(u)}</a></li>"
                for u in urls
            )
            data = (
                f"<div class='srcdata' id='{pid}'>"
                f"<div class='srchd'>{icon} {action} · {len(urls)} "
                f"source{'s' if len(urls) != 1 else ''}</div><ul>{links}</ul></div>"
            )
        elif action:
            tail = f" <span class='actnote'>{icon} {action}</span>"

        evidence = _evidence_html(s)

        items.append(
            f"<div class='tstep'><div class='tnum'>{step + 1}</div>"
            f"<div class='tbody'><div class='tthink'><p>{think}{tail}</p></div>"
            f"{evidence}{data}</div></div>"
        )
    return "".join(items)


# A single floating popover, anchored next to whichever phrase was clicked.
OVERLAY_HTML = "<div id='pop' class='pop' onclick='event.stopPropagation()'></div>"

TRAJECTORY_JS = (
    "function openSrc(id,ev){ev.stopPropagation();"
    "var d=document.getElementById(id);if(!d)return;"
    "var p=document.getElementById('pop');p.innerHTML=d.innerHTML;"
    "p.style.display='block';"
    "var r=ev.target.getBoundingClientRect();"
    "var pw=p.offsetWidth||380,ph=p.offsetHeight||200;"
    "var left=Math.max(10,Math.min(r.left,window.innerWidth-pw-10));"
    "var top=r.bottom+8;"
    "if(top+ph>window.innerHeight-10){top=Math.max(10,r.top-ph-8);}"
    "p.style.left=left+'px';p.style.top=top+'px';}"
    "function closePop(){var p=document.getElementById('pop');if(p)p.style.display='none';}"
    "document.addEventListener('click',closePop);"
    "window.addEventListener('scroll',function(e){var p=document.getElementById('pop');"
    "if(p&&p.contains(e.target))return;closePop();},true);"
    "document.addEventListener('keydown',function(e){if(e.key==='Escape')closePop();});"
)

TRAJECTORY_CSS = """
:root { --teal:#2DA5B6; --line:#ECE9E4; --soft:#F8F7F5; }
.tstep { display:flex; gap:.9rem; padding:.7rem 0; border-bottom:1px solid var(--line); }
.tnum { flex:0 0 1.7rem; height:1.7rem; border-radius:999px; background:var(--teal);
        color:#fff; font-weight:700; font-size:.9rem; display:flex;
        align-items:center; justify-content:center; }
.tbody { flex:1; min-width:0; }
.tthink { color:#3a3a3a; font-size:.93rem; line-height:1.5; }
.tthink p { margin:0 0 .5rem; }
.cue { background:rgba(45,165,182,.16); color:#176874; font-weight:600;
       border-radius:4px; padding:0 .2rem; }
.cue.click { cursor:pointer; border-bottom:1.5px dashed var(--teal); }
.cue.click:hover { background:rgba(45,165,182,.32); }
.actnote { color:#6B6B6B; font-weight:600; white-space:nowrap; }
/* inline evidence: extracted structured facts */
.ev { margin-top:.45rem; font-size:.85rem; }
.ev .evk { font-weight:700; color:#176874; text-transform:uppercase;
           font-size:.68rem; letter-spacing:.05em; margin-right:.5rem; }
.evfact { display:inline-block; background:var(--soft); border:1px solid var(--line);
          border-radius:6px; padding:.13rem .5rem; margin:.15rem .3rem .15rem 0; }
/* inline verification verdict */
.vf { margin-top:.45rem; font-size:.85rem; font-weight:600; border-radius:6px;
      padding:.28rem .65rem; display:inline-block; }
.vf-yes { background:rgba(45,165,182,.14); color:#176874; }
.vf-partial { background:rgba(204,150,30,.16); color:#8a6a12; }
.vf-no { background:rgba(212,55,71,.12); color:#b02a38; }
.srcdata { display:none; }
/* floating popover, anchored to the clicked phrase */
.pop { position:fixed; display:none; z-index:9999; max-width:440px;
       background:#fff; border:1px solid var(--line); border-top:3px solid var(--teal);
       border-radius:10px; box-shadow:0 14px 44px rgba(0,0,0,.22);
       padding:.9rem 1.1rem; animation:pop .12s ease-out; }
@keyframes pop { from { transform:translateY(5px); opacity:0; } to { transform:translateY(0); opacity:1; } }
.pop .srchd { font-weight:700; color:#176874; font-size:.92rem; margin-bottom:.5rem; }
.pop ul { margin:0; padding-left:1.1rem; max-height:300px; overflow:auto; }
.pop li { margin-bottom:.35rem; font-size:.85rem; }
a { color:var(--teal); word-break:break-all; }
"""


def domain_authority_html(openai_urls, claude_urls, *, top_n: int = 8,
                          category: str = "this") -> str:
    """CSS-bar chart of the domains ChatGPT + Claude actually cited, ranked by
    frequency. Dependency-free. This is the evidence behind the 'get featured
    on these domains' advice."""
    from collections import Counter
    from urllib.parse import urlparse

    def _dom(u):
        try:
            h = urlparse(u).netloc.lower()
        except Exception:
            return ""
        return h[4:] if h.startswith("www.") else h

    counts: Counter = Counter()
    for u in (openai_urls or []):
        d = _dom(u)
        if d:
            counts[d] += 1
    for u in (claude_urls or []):
        d = _dom(u)
        if d:
            counts[d] += 1
    items = [(d, n) for d, n in counts.most_common(top_n) if d]
    if not items:
        return ""
    mx = max(n for _, n in items)

    # --- crisp inline-SVG horizontal bar chart (dependency-free) -------------
    label_w, gap, bar_max, val_w = 196, 10, 360, 42
    row_h, pad_t, pad_b = 34, 10, 8
    bar_x = label_w + gap
    width = bar_x + bar_max + val_w
    height = pad_t + len(items) * row_h + pad_b

    def _short(d, lim=26):
        return d if len(d) <= lim else d[: lim - 1] + "…"

    bars = []
    for i, (d, n) in enumerate(items):
        y = pad_t + i * row_h
        cy = y + row_h / 2
        bw = max(4, bar_max * n / mx)
        top = i == 0  # highlight the leader
        fill = "#1B8090" if top else "#2DA5B6"
        bars.append(
            f"<text x='{label_w}' y='{cy:.0f}' dominant-baseline='central' "
            f"text-anchor='end' class='svglbl'>{_esc(_short(d))}</text>"
            f"<rect x='{bar_x}' y='{y + 7}' width='{bw:.1f}' height='{row_h - 14}' "
            f"rx='5' fill='{fill}'></rect>"
            f"<text x='{bar_x + bw + 8:.1f}' y='{cy:.0f}' dominant-baseline='central' "
            f"class='svgval'>{n}</text>"
        )
    svg = (
        f"<svg viewBox='0 0 {width} {height}' class='dchart' "
        f"role='img' preserveAspectRatio='xMinYMin meet'>{''.join(bars)}</svg>"
    )
    contributors = []
    if openai_urls:
        contributors.append("ChatGPT")
    if claude_urls:
        contributors.append("Claude")
    who = " and ".join(contributors) if contributors else "the AI models"
    return (
        "<div class='chart'>"
        f"<div class='charthd'>Where AI looks when answering {_esc(category)}</div>"
        f"<div class='chartsub'>Domains {who} cited, ranked by frequency — "
        "the sources a brand needs to appear on to get recommended.</div>"
        f"{svg}</div>"
    )


def _brand_consensus(systems: dict, matches: dict, *, top_n: int = 9) -> list[tuple]:
    """Cluster brands across the 3 systems via the judge's matched pairs and
    count how many systems surface each (0–3). Returns [(display_name, count)]
    sorted by count desc then name, capped at top_n. Shared by the brand-
    visibility chart and the classic-rank panel so both show the same brands."""
    lists = {
        "surrogate": systems.get("surrogate", {}).get("ranked") or [],
        "openai": systems.get("openai", {}).get("ranked") or [],
        "claude": systems.get("claude", {}).get("ranked") or [],
    }
    if not any(lists.values()):
        return []

    parent: dict = {}
    info: dict = {}

    def key(sys, item):
        return (sys, item.strip().lower())

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        parent.setdefault(a, a); parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for sys, items in lists.items():
        for it in items:
            k = key(sys, it)
            parent.setdefault(k, k)
            info.setdefault(k, {"name": it, "sys": sys})

    for sa, sb, mkey in (("surrogate", "openai", "sur_openai"),
                         ("surrogate", "claude", "sur_claude"),
                         ("openai", "claude", "openai_claude")):
        for p in (matches.get(mkey, {}).get("matched_pairs") or []):
            if len(p) >= 2:
                ka, kb = key(sa, str(p[0])), key(sb, str(p[1]))
                parent.setdefault(ka, ka); info.setdefault(ka, {"name": p[0], "sys": sa})
                parent.setdefault(kb, kb); info.setdefault(kb, {"name": p[1], "sys": sb})
                union(ka, kb)

    comps: dict = {}
    for k in list(parent):
        r = find(k)
        c = comps.setdefault(r, {"systems": set(), "names": []})
        c["systems"].add(info[k]["sys"])
        c["names"].append(info[k]["name"])

    entries = [(sorted(c["names"], key=len)[0], len(c["systems"]))
               for c in comps.values()]
    # collapse display-name duplicates, keep the higher count
    best: dict = {}
    for name, cnt in entries:
        nk = name.strip().lower()
        if nk not in best or cnt > best[nk][1]:
            best[nk] = (name, cnt)
    return sorted(best.values(), key=lambda e: (-e[1], e[0].lower()))[:top_n]


def brand_visibility_html(systems: dict, matches: dict, brand: str,
                          *, top_n: int = 9, category: str = "this") -> str:
    """Bar chart: how many of the 3 systems surface each brand (0–3), brands
    clustered across systems via the judge's matched pairs. The tracked brand
    is pinned (red) if no system surfaced it. Always computable from a run."""
    lists = {
        "surrogate": systems.get("surrogate", {}).get("ranked") or [],
        "openai": systems.get("openai", {}).get("ranked") or [],
        "claude": systems.get("claude", {}).get("ranked") or [],
    }
    if not any(lists.values()):
        return ""

    entries = _brand_consensus(systems, matches, top_n=top_n)

    # tracked brand: present in any list?
    present = any(brand.lower() in str(it).lower()
                  for items in lists.values() for it in items)
    if not present and brand:
        entries.append((f"{brand} (you)", 0))

    label_w, gap, bar_max, val_w = 210, 10, 320, 46
    row_h, pad_t, pad_b = 34, 10, 8
    bar_x = label_w + gap
    width = bar_x + bar_max + val_w
    height = pad_t + len(entries) * row_h + pad_b

    def _short(d, lim=28):
        return d if len(d) <= lim else d[: lim - 1] + "…"

    color = {3: "#176874", 2: "#2DA5B6", 1: "#9AD0D9", 0: "#D43747"}
    bars = []
    for i, (name, cnt) in enumerate(entries):
        y = pad_t + i * row_h
        cy = y + row_h / 2
        bw = max(6, bar_max * (cnt / 3))
        is_you = cnt == 0
        lblcls = "svglbl you" if is_you else "svglbl"
        bars.append(
            f"<text x='{label_w}' y='{cy:.0f}' dominant-baseline='central' "
            f"text-anchor='end' class='{lblcls}'>{_esc(_short(name))}</text>"
            f"<rect x='{bar_x}' y='{y + 7}' width='{bw:.1f}' height='{row_h - 14}' "
            f"rx='5' fill='{color.get(cnt, '#2DA5B6')}'></rect>"
            f"<text x='{bar_x + bw + 8:.1f}' y='{cy:.0f}' dominant-baseline='central' "
            f"class='svgval'>{cnt}/3</text>"
        )
    svg = (f"<svg viewBox='0 0 {width} {height}' class='dchart' role='img' "
           f"preserveAspectRatio='xMinYMin meet'>{''.join(bars)}</svg>")
    sub = (f"How many of the three AI systems recommend each brand. "
           + (f"{_esc(brand)} appears in none."
              if not present and brand else
              f"Higher means stronger agreement."))
    return (
        "<div class='chart'>"
        "<div class='charthd'>Which brands the AI systems agree on</div>"
        f"<div class='chartsub'>{sub}</div>{svg}</div>"
    )


def counterfactual_html(record: dict) -> str:
    """Before/after 'what if <brand> were listed on <anchor>' panel. Returns ''
    when the run has no counterfactual block (it's an optional, GPU-produced
    extra). Clearly labelled as a projection."""
    cf = record.get("counterfactual")
    if not cf:
        return ""
    brand = _esc(cf.get("brand", "the brand"))
    anchor = _esc(cf.get("anchor", "a top source"))
    base = cf.get("baseline", {})
    sysd = cf.get("systems", {})
    labels = {"openai": "ChatGPT", "claude": "Claude", "surrogate": "Our surrogate"}

    def cell(ranked, hit, after):
        if not hit:
            return ("<span class='cf-no'>still absent</span>" if after
                    else "<span class='cf-no'>absent</span>")
        pos = next((i + 1 for i, x in enumerate(ranked or [])
                    if str(x).lower() == str(hit).lower()), None)
        where = f" (#{pos})" if pos else ""
        mark = "&#10003; " if after else ""
        cls = "cf-yes" if after else "cf-mid"
        return f"<span class='{cls}'>{mark}appears{where}</span>"

    rows = ["<tr><th>System</th><th>Before</th><th>After</th></tr>"]
    for key in ("openai", "claude", "surrogate"):
        b = base.get(key, {})
        a = sysd.get(key, {})
        rows.append(
            f"<tr><td>{labels[key]}</td>"
            f"<td>{cell(b.get('ranked'), b.get('hit'), False)}</td>"
            f"<td>{cell(a.get('ranked'), a.get('hit'), True)}</td></tr>"
        )
    return (
        "<div class='chart cf'>"
        f"<div class='charthd'>What if {brand} were listed on {anchor}?</div>"
        f"<div class='chartsub'>A projection: we re-asked each model assuming "
        f"{brand} appears on {anchor}, using {brand}'s real data, and to include "
        f"it only if it genuinely ranks.</div>"
        f"<table class='cftab'>{''.join(rows)}</table></div>"
    )


def _collapse(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _google_field_for(name: str, ranks: list[dict], field: str):
    """Best (lowest) value of `field` ('position' or 'mention') among
    classic_rank entries whose brand collapse-matches `name`. None if absent."""
    nc = _collapse(name)
    if not nc:
        return None
    found = [r[field] for r in ranks
             if (rc := _collapse(r.get("brand", "")))
             and (nc in rc or rc in nc) and r.get(field)]
    return min(found) if found else None


def classic_search_html(record: dict) -> str:
    """#6 — how each AI-recommended brand fares in classic Google search, two
    ways: where its OWN website ranks, and where it is first NAMED by any source
    (a listicle, review, etc.). Returns '' when no classic_rank block is present.
    Deterministic. Brands come from the same consensus clustering as the
    visibility chart, so the panels line up. The point: a brand a buyer reaches
    via Google only through third-party articles — if at all — versus how AI
    names it directly (shown in the visibility chart)."""
    cr = record.get("classic_rank")
    if not cr or not cr.get("ranks"):
        return ""
    systems = record.get("systems", {})
    matches = record.get("matches", {})
    brand = record.get("brand", "")
    label = cr.get("label", "Google")      # "Google" or "web search"
    depth = cr.get("results_count") or cr.get("depth", 100)
    bc_track = _collapse(brand)

    # Brand set = top consensus brands PLUS any AI-recommended brand that
    # actually shows up in Google (own site or mention), so the Google-visible
    # ones are never cut by the consensus cap. Consensus gives deduped names.
    consensus = _brand_consensus(systems, matches, top_n=50)
    rows = []
    for i, (name, _cnt) in enumerate(consensus):
        own = _google_field_for(name, cr["ranks"], "position")
        men = _google_field_for(name, cr["ranks"], "mention")
        if i < 12 or own or men:
            rows.append((name, own, men,
                         bool(bc_track) and bc_track in _collapse(name)))
    # tracked brand: ensure a row even if no system surfaced it
    if bc_track and not any(r[3] for r in rows):
        rows.append((brand,
                     _google_field_for(brand, cr["ranks"], "position"),
                     _google_field_for(brand, cr["ranks"], "mention"),
                     True))

    # sort by own-site rank, then first-mention rank; absent sinks to the bottom
    def sort_key(row):
        _, own, men, _t = row
        return (0 if own else 1, own or 10**6, 0 if men else 1, men or 10**6)
    rows.sort(key=sort_key)

    def cell(pos):
        return (f"<span class='cr-rank'>#{pos}</span>" if pos
                else f"<span class='cr-none'>— not in top {depth}</span>")

    trs = ["<tr><th>Brand</th><th>Own website</th><th>First mentioned</th></tr>"]
    for name, own, men, is_track in rows:
        nm = _esc(name) + (" <span class='cr-you'>(you)</span>" if is_track else "")
        cls = " class='cr-track'" if is_track else ""
        trs.append(f"<tr{cls}><td>{nm}</td><td>{cell(own)}</td><td>{cell(men)}</td></tr>")

    engine_note = ("real Google organic positions" if cr.get("engine") == "serper"
                   else "web-search ranking, Google-class index")
    sub = (f"Where each AI-recommended brand appears in {_esc(label)} for this "
           f"query ({engine_note}, top {depth}). <b>Own website</b>: where its "
           f"own site ranks. <b>First mentioned</b>: the first article that names "
           f"it. A dash means neither.")
    return (
        "<div class='chart'>"
        f"<div class='charthd'>Where these brands appear in classic {_esc(label)} search</div>"
        f"<div class='chartsub'>{sub}</div>"
        f"<table class='cftab crtab'>{''.join(trs)}</table></div>"
    )


def graph_panels(record: dict) -> list[str]:
    """Registry: return the HTML for every graph applicable to this run, in
    display order. Always-on graphs render for any question; conditional ones
    return '' when their data isn't rich enough."""
    sysd = record.get("systems", {})
    matches = record.get("matches", {})
    brand = record.get("brand", "")
    cat = f"“{record.get('question', 'this query')}”"
    panels = [
        brand_visibility_html(sysd, matches, brand, category=cat),
        domain_authority_html(sysd.get("openai", {}).get("urls"),
                              sysd.get("claude", {}).get("urls"), category=cat),
        classic_search_html(record),
    ]
    return [p for p in panels if p]


CHART_CSS = """
.charts-row { display:flex; gap:1.2rem; flex-wrap:wrap; align-items:flex-start; margin:1.4rem 0; }
.charts-row > .chart { flex:1 1 380px; min-width:300px; margin:0; }
.chart { margin:1.4rem 0; padding:1.1rem 1.3rem; background:var(--soft);
         border:1px solid var(--line); border-radius:12px; }
.charthd { font-family:'Epilogue',sans-serif; font-weight:600; font-size:1.18rem; margin-bottom:.25rem; }
.chartsub { color:#6B6B6B; font-size:.85rem; margin-bottom:1rem; max-width:660px; }
.dchart { width:100%; height:auto; display:block; }
.dchart .svglbl { fill:#444; font-size:13px; font-family:'Mulish',sans-serif; }
.dchart .svglbl.you { fill:#B02A38; font-weight:700; }
.dchart .svgval { fill:#176874; font-size:13px; font-weight:700; font-family:'Mulish',sans-serif; }
.cftab { width:100%; border-collapse:collapse; margin-top:.3rem; }
.cftab th { text-align:left; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em;
            color:#555; border-bottom:2px solid var(--teal); padding:.4rem .6rem; }
.cftab td { padding:.45rem .6rem; border-bottom:1px solid var(--line); font-size:.92rem; }
.cf-yes { color:#1B8090; font-weight:700; }
.cf-mid { color:#176874; }
.cf-no { color:#9a9a9a; }
.crtab .cr-rank { color:#176874; font-weight:700; }
.crtab .cr-none { color:#9a9a9a; }
.crtab .cr-ai { font-weight:700; color:#444; }
.crtab .cr-dot { display:inline-block; width:.62rem; height:.62rem; border-radius:50%;
                 margin-right:.4rem; vertical-align:middle; }
.crtab tr.cr-track td { background:rgba(212,55,71,.07); font-weight:600; }
.crtab .cr-you { color:#B02A38; font-weight:700; font-size:.8rem; }
"""


def estimate_height(traj: list[dict]) -> int:
    h = 40
    for s in traj or []:
        h += 90 + int(len(s.get("think") or "") / 80) * 22
    return min(max(h, 200), 6000)


def trajectory_component_html(traj: list[dict], *, viewport_height: int = 560) -> str:
    """Standalone doc for st.components.v1.html. A modest fixed iframe height
    with internal scroll keeps the modal (position:fixed) centered in the
    visible area rather than lost in a very tall iframe."""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Mulish:wght@400;600;700&display=swap' rel='stylesheet'>"
        "<style>html,body{margin:0;}"
        "body{font-family:'Mulish',sans-serif;color:#1A1A1A;padding:.2rem 1rem;}"
        f"{TRAJECTORY_CSS}</style></head><body>"
        f"<div class='traj'>{trajectory_inner_html(traj)}</div>"
        f"{OVERLAY_HTML}<script>{TRAJECTORY_JS}</script></body></html>"
    )
