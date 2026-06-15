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
            return (f"<div class='vf {cls}'>{mark} verified: "
                    f"&ldquo;{claim}&rdquo; — {label}</div>")
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
