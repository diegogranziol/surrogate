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


_VERBATIM_URL_RE = re.compile(r"https?://[^\s<>]+")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")


def linkify_verbatim(text: str) -> str:
    """Render a model's answer for display: keep its content faithful but show
    the light Markdown it actually uses (**bold**, _italic_) as formatting, turn
    bare URLs into clickable links, and keep '$20–$50' literal (NO LaTeX). Use
    with st.html() / raw HTML — NOT st.markdown (which would eat '$…$' as math).

    Deliberately handles only **bold** and _italic_ (what these answers use) so
    stray '*' / '_' elsewhere can't garble the text."""
    s = _esc(text or "")

    # stash URLs first so the emphasis passes can't touch their punctuation
    urls: list[str] = []

    def _stash(m):
        urls.append(m.group(0))
        return f"\x00U{len(urls) - 1}\x00"

    s = _VERBATIM_URL_RE.sub(_stash, s)
    s = _BOLD_RE.sub(r"<strong>\1</strong>", s)
    s = _ITALIC_RE.sub(r"<em>\1</em>", s)

    def _restore(m):
        url = urls[int(m.group(1))]
        trail = ""
        while url and url[-1] in ".,)]}":   # don't swallow trailing punctuation
            trail = url[-1] + trail
            url = url[:-1]
        return f"<a href='{url}' target='_blank' rel='noopener'>{url}</a>{trail}"

    s = re.sub(r"\x00U(\d+)\x00", _restore, s)
    return s.replace("\n", "<br>")


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


def trajectory_inner_html(traj: list[dict], id_prefix: str = "") -> str:
    """Timeline items (no wrapper, no <style>/<script>). Each step that pulled
    sources holds a hidden `.srcdata` div the modal reads on click.

    `id_prefix` namespaces the per-step element IDs (e.g. 'swiss-src0') so
    several trajectories can coexist in one document — needed when the static
    demo embeds multiple examples behind a selector."""
    if not traj:
        return ""
    items = []
    for s in traj:
        step = s.get("step", 0)
        pid = f"{id_prefix}src{step}"
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
.pop { position:fixed; display:none; z-index:9999; max-width:520px;
       max-height:72vh; overflow:auto;
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


def _cluster_brands(systems: dict, matches: dict) -> list[dict]:
    """Cluster brand mentions across the 3 systems via the judge's matched pairs
    (union-find). Returns a list of {names: [...], systems: set(...)} — one entry
    per distinct brand, with the set of systems that surfaced it. Shared by the
    consensus bar chart, the classic-rank panel, and the Venn diagram."""
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
    return list(comps.values())


def _brand_consensus(systems: dict, matches: dict, *, top_n: int = 9) -> list[tuple]:
    """Cluster brands across the 3 systems and count how many surface each (1–3).
    Returns [(display_name, count)] sorted by count desc then name, capped at
    top_n. Shared by the brand-visibility chart and the classic-rank panel."""
    entries = [(sorted(c["names"], key=len)[0], len(c["systems"]))
               for c in _cluster_brands(systems, matches)]
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


_CF_SYS_LABELS = {"openai": "ChatGPT", "claude": "Claude",
                  "surrogate": "Our surrogate"}


def _cf_cell(ranked, hit, after):
    """One before/after status cell for the counterfactual table."""
    if not hit:
        return ("<span class='cf-no'>still absent</span>" if after
                else "<span class='cf-no'>absent</span>")
    pos = next((i + 1 for i, x in enumerate(ranked or [])
                if str(x).lower() == str(hit).lower()), None)
    where = f" (#{pos})" if pos else ""
    mark = "&#10003; " if after else ""
    cls = "cf-yes" if after else "cf-mid"
    return f"<span class='{cls}'>{mark}appears{where}</span>"


def _cf_assumption(sid: str, brand: str, anchor: str) -> str:
    """Plain-language description of the assumption we fed the model for a
    scenario — shown in the click-through popover so the client sees exactly
    what we posed, not a black box."""
    if sid == "listing":
        return (f"Assume {brand} now appears on {anchor} (a third-party "
                f"directory/review platform for this category).")
    if sid == "ownsite":
        return (f"Assume {brand}'s own website now ranks on the first page of "
                f"search results for this query.")
    if sid == "wikipedia":
        return (f"Assume {brand} now has its own dedicated Wikipedia article.")
    return f"Assume {brand} were more discoverable for this query."


def _cf_traj_inline(traj: list[dict]) -> str:
    """Compact, read-only step-by-step view of the surrogate's reasoning for a
    counterfactual run, for inside the popover (no nested click-to-reveal — the
    sources are listed inline per step)."""
    parts = []
    for s in traj or []:
        n = (s.get("step", 0) or 0) + 1
        think = _esc((s.get("think") or "").strip()).replace("\n", " ")
        icon = _TOOL_ICON.get(s.get("tool"), "&#8226;")
        action = _esc(s.get("action") or "")
        urls = s.get("urls") or []
        links = "".join(
            f"<li><a href='{_esc(u)}' target='_blank' rel='noopener'>{_esc(u)}</a></li>"
            for u in urls)
        links = f"<ul class='cfstep-src'>{links}</ul>" if links else ""
        act = f"<div class='cfstep-act'>{icon} {action}</div>" if action else ""
        ev = _evidence_html(s)
        parts.append(
            f"<div class='cfstep'><span class='cfstep-n'>{n}</span>"
            f"<div class='cfstep-b'>{think}{act}{ev}{links}</div></div>"
        )
    return "".join(parts)


def counterfactual_html(record: dict, id_prefix: str = "") -> str:
    """Counterfactual 'what if <brand> were more discoverable' panel. Returns ''
    when the run has no counterfactual block. Always labelled as a projection.

    Each scenario cell (where the brand was absent at baseline) is clickable: it
    opens the same anchored popover as the reasoning timeline, showing the exact
    assumption we posed and the model's verbatim re-answer — so the client can
    see we re-ran the model, not invented the result. `id_prefix` namespaces the
    popover IDs so several examples can coexist (the static demo embeds many).

    Two shapes: multi-scenario (cf['scenarios']) and legacy single (cf['systems'])."""
    cf = record.get("counterfactual")
    if not cf:
        return ""
    brand = _esc(cf.get("brand", "the brand"))
    raw_brand = cf.get("brand", "the brand")
    raw_anchor = cf.get("anchor", "a top source")
    base = cf.get("baseline", {})
    scenarios = cf.get("scenarios")

    # hidden data divs (popover bodies), collected and appended after the table
    data_divs: list[str] = []

    def clickable(pid_tail, key, label, sysblock, sid, cell_inner):
        """Wrap a cell so clicking opens the popover with the model's actual
        output: the surrogate's step-by-step reasoning when we captured it,
        otherwise the verbatim answer. Plain cell when there's nothing stored."""
        sysblock = sysblock or {}
        ans = sysblock.get("answer") or ""
        traj = sysblock.get("trajectory") or []
        if not ans and not traj:
            return f"<td>{cell_inner}</td>"
        pid = f"{id_prefix}cf-{pid_tail}"
        assume = _cf_assumption(sid, raw_brand, raw_anchor)
        if key == "surrogate" and traj:
            body = ("<div class='cfpop-anshd'>Its step-by-step reasoning:</div>"
                    f"<div class='cfpop-traj'>{_cf_traj_inline(traj)}</div>")
        else:
            body = ("<div class='cfpop-anshd'>What it answered (verbatim):</div>"
                    f"<div class='cfpop-ans'>{linkify_verbatim(ans)}</div>")
        data_divs.append(
            f"<div class='srcdata' id='{pid}'>"
            f"<div class='cfpop-hd'>{_CF_SYS_LABELS[key]} &middot; {_esc(label)}</div>"
            f"<div class='cfpop-assume'><b>What we told the model:</b> {_esc(assume)} "
            f"We then re-asked the original question and told it to include "
            f"{brand} only if it genuinely belongs.</div>{body}</div>"
        )
        return (f"<td><span class='cf-click' onclick=\"openSrc('{pid}',event)\" "
                f"title='see what {_CF_SYS_LABELS[key]} actually did'>"
                f"{cell_inner} <span class='cf-i'>&#9432;</span></span></td>")

    if scenarios:
        # Gate: the counterfactual only means something where the brand is
        # currently ABSENT. Systems that already list it at baseline can't be
        # "moved in"; if ALL three already include it, skip the table.
        base_hit = {k: bool(base.get(k, {}).get("hit"))
                    for k in ("openai", "claude", "surrogate")}
        if all(base_hit.values()):
            return (
                "<div class='chart cf'>"
                f"<div class='charthd'>{brand} already appears for this query</div>"
                f"<div class='chartsub'>All three systems already include {brand} in "
                f"their answers here, so the &ldquo;what if it were more "
                f"discoverable&rdquo; scenarios don&rsquo;t apply &mdash; the "
                f"before/after projection is only meaningful when {brand} is "
                f"currently absent.</div></div>"
            )

        head = "<tr><th>System</th><th>Before</th>"
        for sc in scenarios:
            head += f"<th>{_esc(sc.get('label', sc.get('id', '')))}</th>"
        head += "</tr>"
        rows = [head]
        for key in ("openai", "claude", "surrogate"):
            b = base.get(key, {})
            cells = [f"<td>{_CF_SYS_LABELS[key]}</td>",
                     f"<td>{_cf_cell(b.get('ranked'), b.get('hit'), False)}</td>"]
            for si, sc in enumerate(scenarios):
                if base_hit[key]:
                    cells.append("<td><span class='cf-already'>already listed</span></td>")
                else:
                    a = sc.get("systems", {}).get(key, {})
                    inner = _cf_cell(a.get("ranked"), a.get("hit"), True)
                    cells.append(clickable(f"{key}-{si}", key,
                                           sc.get("label", sc.get("id", "")),
                                           a, sc.get("id", ""), inner))
            rows.append("<tr>" + "".join(cells) + "</tr>")
        sub = (f"A projection, not a measurement. For each what-if we re-asked all "
               f"three models — using {brand}'s real data — to rank the genuinely "
               f"best options and include {brand} only if it then belongs. "
               f"<b>Before</b> is today's answer; each column is one improvement "
               f"{brand} could make. <b>Click any result to see exactly what we "
               f"asked and what the model answered.</b>")
        return (
            "<div class='chart cf'>"
            f"<div class='charthd'>What would actually move the needle for {brand}?</div>"
            f"<div class='chartsub'>{sub}</div>"
            f"<table class='cftab'>{''.join(rows)}</table>{''.join(data_divs)}</div>"
        )

    # legacy single-scenario shape
    anchor = _esc(cf.get("anchor", "a top source"))
    sysd = cf.get("systems", {})
    rows = ["<tr><th>System</th><th>Before</th><th>After</th></tr>"]
    for key in ("openai", "claude", "surrogate"):
        b = base.get(key, {})
        a = sysd.get(key, {})
        rows.append(
            f"<tr><td>{_CF_SYS_LABELS[key]}</td>"
            f"<td>{_cf_cell(b.get('ranked'), b.get('hit'), False)}</td>"
            f"<td>{_cf_cell(a.get('ranked'), a.get('hit'), True)}</td></tr>"
        )
    return (
        "<div class='chart cf'>"
        f"<div class='charthd'>What if {brand} were listed on {anchor}?</div>"
        f"<div class='chartsub'>A projection: we re-asked each model assuming "
        f"{brand} appears on {anchor}, using {brand}'s real data, and to include "
        f"it only if it genuinely ranks.</div>"
        f"<table class='cftab'>{''.join(rows)}</table></div>"
    )


def counterfactual_component_html(record: dict) -> str:
    """Self-contained HTML doc for st.components.v1.html: the counterfactual
    panel with its click-through popover working. Streamlit's main page can't
    run the inline popover JS, so (like the reasoning timeline) we render it in
    an isolated iframe that carries its own CSS, overlay, and JS. Returns '' when
    there's no counterfactual block."""
    inner = counterfactual_html(record)
    if not inner:
        return ""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Epilogue:wght@500;600;700&family=Mulish:wght@400;600;700&display=swap' rel='stylesheet'>"
        "<style>html,body{margin:0;}"
        "body{font-family:'Mulish',sans-serif;color:#1A1A1A;padding:.2rem .4rem;}"
        f"{CHART_CSS}{TRAJECTORY_CSS}</style></head><body>"
        f"{inner}{OVERLAY_HTML}<script>{TRAJECTORY_JS}</script></body></html>"
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


def _google_best(name: str, ranks: list[dict], field: str, url_field: str):
    """(position, url) of the lowest-ranked classic_rank entry that collapse-
    matches `name` on `field`; (None, None) if absent. `url_field` is the URL
    that goes with that position (e.g. 'mention_url' for 'mention')."""
    nc = _collapse(name)
    if not nc:
        return (None, None)
    best = None
    for r in ranks:
        rc = _collapse(r.get("brand", ""))
        if rc and (nc in rc or rc in nc) and r.get(field):
            if best is None or r[field] < best[0]:
                best = (r[field], r.get(url_field))
    return best or (None, None)


def _short_domain(url: str) -> str:
    """Display host for a URL, e.g. 'https://www.healthline.com/x' -> 'healthline.com'."""
    m = re.search(r"https?://([^/]+)", url or "")
    d = (m.group(1) if m else (url or "")).lower()
    return d[4:] if d.startswith("www.") else d


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
        own = _google_best(name, cr["ranks"], "position", "url")
        men = _google_best(name, cr["ranks"], "mention", "mention_url")
        if i < 12 or own[0] or men[0]:
            rows.append((name, own, men,
                         bool(bc_track) and bc_track in _collapse(name)))
    # tracked brand: ensure a row even if no system surfaced it
    if bc_track and not any(r[3] for r in rows):
        rows.append((brand,
                     _google_best(brand, cr["ranks"], "position", "url"),
                     _google_best(brand, cr["ranks"], "mention", "mention_url"),
                     True))

    # sort by own-site rank, then first-mention rank; absent sinks to the bottom
    def sort_key(row):
        _, own, men, _t = row
        return (0 if own[0] else 1, own[0] or 10**6,
                0 if men[0] else 1, men[0] or 10**6)
    rows.sort(key=sort_key)

    def cell(pos, url=None):
        if not pos:
            return f"<span class='cr-none'>— not in top {depth}</span>"
        out = f"<span class='cr-rank'>#{pos}</span>"
        if url:
            out += (f" <a class='cr-src' href='{_esc(url)}' target='_blank' "
                    f"rel='noopener'>{_esc(_short_domain(url))}</a>")
        return out

    trs = ["<tr><th>Brand</th><th>Own website</th><th>First mentioned</th></tr>"]
    for name, own, men, is_track in rows:
        nm = _esc(name) + (" <span class='cr-you'>(you)</span>" if is_track else "")
        cls = " class='cr-track'" if is_track else ""
        trs.append(f"<tr{cls}><td>{nm}</td><td>{cell(*own)}</td>"
                   f"<td>{cell(*men)}</td></tr>")

    engine_note = ("real Google organic positions" if cr.get("engine") == "serper"
                   else "web-search ranking, Google-class index")
    sub = (f"Where each AI-recommended brand appears in {_esc(label)} for this "
           f"query ({engine_note}, top {depth}). <b>Own website</b>: where its "
           f"own site ranks. <b>First mentioned</b>: the first article that names "
           f"it (click the source to see where). A dash means neither.")
    return (
        "<div class='chart'>"
        f"<div class='charthd'>Where these brands appear in classic {_esc(label)} search</div>"
        f"<div class='chartsub'>{sub}</div>"
        f"<table class='cftab crtab'>{''.join(trs)}</table></div>"
    )


def _brand_pos(brand: str, ranked: list) -> int | None:
    """1-based position of `brand` in `ranked` (collapse-matched), else None."""
    bc = _collapse(brand)
    if not bc:
        return None
    for i, x in enumerate(ranked or []):
        xc = _collapse(x)
        if xc and (bc in xc or xc in bc):
            return i + 1
    return None


def agreement_venn_html(record: dict) -> str:
    """3-circle Venn of the brand picks: surrogate / ChatGPT / Claude, with the
    count of brands in each region (unique to one system, shared by two, or by
    all three). A visual companion to the consensus bar chart — the center is
    where all three agree. Deterministic; uses the same cross-system clustering."""
    systems = record.get("systems", {})
    matches = record.get("matches", {})
    clusters = _cluster_brands(systems, matches)
    if not clusters:
        return ""

    S, O, C = "surrogate", "openai", "claude"
    reg = {k: 0 for k in ("s", "o", "c", "so", "sc", "oc", "soc")}
    for cl in clusters:
        ss = cl["systems"]
        has_s, has_o, has_c = S in ss, O in ss, C in ss
        if has_s and has_o and has_c:
            reg["soc"] += 1
        elif has_s and has_o:
            reg["so"] += 1
        elif has_s and has_c:
            reg["sc"] += 1
        elif has_o and has_c:
            reg["oc"] += 1
        elif has_s:
            reg["s"] += 1
        elif has_o:
            reg["o"] += 1
        elif has_c:
            reg["c"] += 1

    SUR, OAI, CLA = "#2DA5B6", "#E0A33E", "#C2613F"
    circles = (
        f"<circle cx='140' cy='130' r='92' fill='{SUR}' fill-opacity='.42' "
        f"stroke='{SUR}' style='mix-blend-mode:multiply'/>"
        f"<circle cx='240' cy='130' r='92' fill='{OAI}' fill-opacity='.42' "
        f"stroke='{OAI}' style='mix-blend-mode:multiply'/>"
        f"<circle cx='190' cy='210' r='92' fill='{CLA}' fill-opacity='.42' "
        f"stroke='{CLA}' style='mix-blend-mode:multiply'/>"
    )

    def num(x, y, n):
        cls = "vnum" if n else "vnum z"
        return f"<text x='{x}' y='{y}' class='{cls}'>{n}</text>"

    nums = (
        num(102, 116, reg["s"]) + num(278, 116, reg["o"]) + num(190, 258, reg["c"])
        + num(190, 104, reg["so"]) + num(150, 188, reg["sc"]) + num(230, 188, reg["oc"])
        + num(190, 162, reg["soc"])
    )
    labels = (
        f"<text x='108' y='44' class='vlbl' style='fill:{SUR}'>Surrogate</text>"
        f"<text x='272' y='44' class='vlbl' style='fill:{OAI}'>ChatGPT</text>"
        f"<text x='190' y='314' class='vlbl' style='fill:{CLA}'>Claude</text>"
    )
    svg = (f"<svg viewBox='0 0 380 320' class='venn' role='img' "
           f"preserveAspectRatio='xMidYMid meet'>{circles}{nums}{labels}</svg>")
    sub = ("Each brand placed by which systems recommend it. The center is "
           "where all three agree; outer slices are brands only one system picks.")
    return (
        "<div class='chart'>"
        "<div class='charthd'>Where the three systems overlap</div>"
        f"<div class='chartsub'>{sub}</div>{svg}</div>"
    )


def fidelity_html(record: dict) -> str:
    """How closely the surrogate mirrors the frontiers for THIS query. Shows the
    surrogate↔ChatGPT and surrogate↔Claude pick overlap against the ChatGPT↔Claude
    'reference' (how much the frontiers agree with each other — the honest
    yardstick, since these rankings are subjective), an interpretive verdict, and
    where the tracked brand lands across the three. Deterministic (uses the
    judge's match counts already in the record)."""
    m = record.get("matches", {}) or {}
    sysd = record.get("systems", {}) or {}
    brand = record.get("brand", "")
    so, sc, oc = (m.get("sur_openai") or {}, m.get("sur_claude") or {},
                  m.get("openai_claude") or {})
    if not (so and sc):
        return ""
    so_ov, sc_ov, oc_ov = (so.get("overlap") or 0, sc.get("overlap") or 0,
                           oc.get("overlap") or 0)
    oc_n = len(oc.get("a") or []) or 0

    best_sf = max(so_ov, sc_ov)
    if oc_ov == 0:
        verdict = (f"Even ChatGPT and Claude share no picks for this query, so "
                   f"there is no single &ldquo;correct&rdquo; ranking to match. "
                   f"The surrogate overlaps {so_ov} with ChatGPT and {sc_ov} with "
                   f"Claude.")
    elif best_sf >= oc_ov:
        verdict = (f"These rankings are subjective &mdash; ChatGPT and Claude "
                   f"themselves agree on only {oc_ov}. The surrogate matches at "
                   f"least one frontier as closely as the frontiers match each "
                   f"other ({so_ov} with ChatGPT, {sc_ov} with Claude).")
    elif best_sf == 0:
        verdict = (f"The surrogate picked a different set here &mdash; no overlap "
                   f"with either frontier (though the frontiers themselves share "
                   f"only {oc_ov} of {oc_n}). A candidate for tuning on this "
                   f"category.")
    else:
        verdict = (f"These rankings are subjective &mdash; even ChatGPT and Claude "
                   f"share only {oc_ov}. The surrogate is close but a bit lower "
                   f"({so_ov} with ChatGPT, {sc_ov} with Claude).")

    pos = {k: _brand_pos(brand, sysd.get(k, {}).get("ranked"))
           for k in ("surrogate", "openai", "claude")}
    lab = {"surrogate": "surrogate", "openai": "ChatGPT", "claude": "Claude"}
    present = [k for k in ("surrogate", "openai", "claude") if pos[k]]
    bn = ""
    if brand and len(present) == 3:
        bn = f"<b>All three rank {_esc(brand)} in their top {max(pos.values())}.</b>"
    elif brand and present:
        bits = ", ".join(f"{lab[k]} #{pos[k]}" for k in present)
        bn = f"{_esc(brand)} appears in {bits}."
    elif brand:
        bn = f"No system surfaced {_esc(brand)} for this query."
    bn_html = f"<div class='fid-brand'>{bn}</div>" if bn else ""

    return (
        "<div class='chart chart-wide'>"
        "<div class='charthd'>How closely the surrogate mirrors the frontiers</div>"
        f"<div class='chartsub'>{verdict}</div>{bn_html}</div>"
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
        # fidelity_html(record),  # methodology/credibility, not client-facing —
        # kept in code for a possible internal "is the surrogate faithful?" view.
        brand_visibility_html(sysd, matches, brand, category=cat),
        agreement_venn_html(record),
        domain_authority_html(sysd.get("openai", {}).get("urls"),
                              sysd.get("claude", {}).get("urls"), category=cat),
        classic_search_html(record),
    ]
    return [p for p in panels if p]


CHART_CSS = """
.charts-row { display:flex; gap:1.2rem; flex-wrap:wrap; align-items:flex-start; margin:1.4rem 0; }
.charts-row > .chart { flex:1 1 380px; min-width:300px; margin:0; }
.charts-row > .chart-wide { flex:1 1 100%; }
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
.cf-already { color:#9a9a9a; font-style:italic; }
.cf-click { cursor:pointer; border-bottom:1px dashed var(--teal); padding-bottom:1px; }
.cf-click:hover { background:rgba(45,165,182,.12); }
.cf-i { color:var(--teal); font-size:.85em; }
.cfpop-hd { font-family:'Epilogue',sans-serif; font-weight:700; color:#176874;
            margin-bottom:.45rem; }
.cfpop-assume { font-size:.84rem; color:#444; background:var(--soft);
                border-radius:6px; padding:.45rem .6rem; margin-bottom:.6rem; }
.cfpop-anshd { font-size:.7rem; text-transform:uppercase; letter-spacing:.05em;
               color:#888; margin-bottom:.25rem; }
.cfpop-ans { font-size:.85rem; line-height:1.45; color:#1f1f1f; }
.cfpop-traj { font-size:.82rem; }
.cfstep { display:flex; gap:.5rem; padding:.4rem 0; border-bottom:1px solid var(--line); }
.cfstep-n { flex:0 0 1.3rem; height:1.3rem; border-radius:999px; background:var(--teal);
            color:#fff; font-size:.7rem; font-weight:700; display:flex;
            align-items:center; justify-content:center; }
.cfstep-b { flex:1; min-width:0; line-height:1.4; color:#333; }
.cfstep-act { color:#6B6B6B; font-weight:600; margin-top:.2rem; }
.cfstep-src { margin:.25rem 0 0; padding-left:1rem; }
.cfstep-src li { font-size:.79rem; margin-bottom:.15rem; }
.fid-brand { margin-top:.7rem; padding:.55rem .8rem; background:rgba(45,165,182,.10);
             border-radius:8px; font-size:.9rem; color:#176874; }
.venn { width:100%; max-width:360px; height:auto; display:block; margin:.2rem auto 0; }
.venn .vnum { font-family:'Mulish',sans-serif; font-size:20px; font-weight:700;
              fill:#2A2A2A; text-anchor:middle; dominant-baseline:central; }
.venn .vnum.z { fill:#B9B9B9; font-weight:600; }
.venn .vlbl { font-family:'Epilogue',sans-serif; font-size:13px; font-weight:700;
              text-anchor:middle; }
.crtab .cr-rank { color:#176874; font-weight:700; }
.crtab .cr-src { font-size:.82rem; color:var(--teal); margin-left:.4rem;
                 text-decoration:none; border-bottom:1px dotted var(--teal); }
.crtab .cr-src:hover { color:#176874; }
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
