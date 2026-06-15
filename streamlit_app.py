"""Streamlit UI for the surrogate deep-research agent.

Demo-first layout: the app opens directly on the Ask page (sidebar collapsed).
Operator pages (Ingest, Documents, Compare two URLs, Settings) are hidden
behind the "Operator tools" toggle in the sidebar so a client demo only ever
sees Ask.

Ask modes:
  - Compare (default, the demo): one question through the surrogate + ChatGPT
    + Claude in parallel; ranked-picks table with brand-level match ticks,
    "what should <brand> do" suggestions, sources consulted.
  - Surrogate: single-stage ReAct loop, 7 tools, visible <think> reasoning.
    Optional side-by-side vs GLM-4.6 on the SAME evidence.
  - Retrieve only: cosine search over the local user-RAG store.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import textwrap
import time
import traceback

import streamlit as st

# NOTE: surrogate.rag pulls sentence-transformers/torch (heavy). It's only
# needed by the operator pages (Ingest/Documents) and Retrieve-only mode, so
# we import it lazily inside those blocks. This keeps the Compare/Test-mode
# demo — and a lightweight cloud deploy — from loading torch at startup.

def _need_rag():
    """Lazy-import the RAG module; show a friendly note on torch-free deploys."""
    try:
        import surrogate.rag as _rag
        return _rag
    except Exception as e:
        st.info(
            "This page needs the embedding extras (sentence-transformers), "
            "which aren't installed on the lightweight demo build. Run "
            "`pip install -r requirements-rag.txt` locally to enable it.",
            icon="ℹ️",
        )
        st.caption(f"import detail: {e!r}")
        st.stop()

# On Streamlit Community Cloud, API keys are set in the dashboard's Secrets.
# Our frontier clients read os.environ, so bridge any secrets across. No-op
# locally (.env handles it) and in Test mode (no keys needed).
try:
    for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY",
               "ZAI_API_KEY", "FRONTIER_OPENAI_MODEL", "FRONTIER_CLAUDE_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

st.set_page_config(
    page_title="AVEA · AI Visibility",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Avea-life brand layer (palette + type pulled from avea-life.com) -------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Epilogue:wght@500;600;700&family=Mulish:wght@400;500;600;700&display=swap');

html, body, .stMarkdown, p, li, label, input, textarea, button {
    font-family: 'Mulish', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Epilogue', sans-serif !important; letter-spacing: -0.01em; }

/* keep Streamlit's Material Symbols icons on their icon font (otherwise the
   ligature name renders as literal text, e.g. "keyboard_double_arrow_right") */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
span[translate="no"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* hide default streamlit chrome for a clean demo */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stAppDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }

/* readable content width, centered — like avea-life.com's page container */
.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1080px;
    margin: 0 auto;
    padding-top: 4.2rem;   /* clear streamlit's fixed top bar (~3.75rem) */
    padding-left: 2rem;
    padding-right: 2rem;
}

/* widget labels ("Your question", "Mode", "Brand to track", …) */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 1.12rem !important;
    font-weight: 600;
    color: #222;
}

.avea-header {
    display: flex; flex-direction: column; align-items: center;
    gap: 1.2rem; margin: 0 0 2.4rem 0;
    font-family: 'Epilogue', sans-serif; font-weight: 600;
    font-size: 1.45rem; letter-spacing: .04em;
}
.avea-header img { height: 2.4rem; display: block; }
.avea-header .accent { color: #2DA5B6; }

/* section rhythm: larger titles, more air between blocks */
[data-testid="stMainBlockContainer"] h2 {
    font-size: 1.7rem; margin: 2.2rem 0 .9rem 0;
}
[data-testid="stMainBlockContainer"] h3 {
    font-size: 1.45rem; margin: 2.0rem 0 .8rem 0;
}
[data-testid="stMainBlockContainer"] h4 {
    font-size: 1.18rem; margin: 1.6rem 0 .6rem 0;
}
[data-testid="stExpander"] { margin-top: 1.2rem; }
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    font-size: 1.05rem; font-weight: 600;
}

/* primary button -> avea pill (covers normal + form-submit buttons) */
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button {
    background: #2DA5B6; border: none; border-radius: 999px;
    padding: .5rem 1.8rem; font-weight: 700; letter-spacing: .02em;
    color: #fff;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button:hover { background: #238D9C; }

/* markdown tables -> branded */
[data-testid="stMarkdownContainer"] table { width: 100%; border-collapse: collapse; }
[data-testid="stMarkdownContainer"] th {
    text-align: left; font-family: 'Epilogue', sans-serif; font-size: .84rem;
    letter-spacing: .05em; text-transform: uppercase; color: #333;
    border-bottom: 2px solid #2DA5B6; padding: .5rem .65rem;
}
[data-testid="stMarkdownContainer"] td {
    border-bottom: 1px solid #ECE9E4; padding: .48rem .65rem; font-size: .93rem;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: #FAF9F7; }

/* suggestions card */
.avea-card {
    background: #F8F7F5; border-left: 4px solid #2DA5B6;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: .6rem 0 1rem 0;
}
.avea-card { margin-top: 1.6rem; }
.avea-card h3 { margin: 0 0 .6rem 0 !important; font-size: 1.4rem; }
.avea-card p { margin: 0 0 .6rem 0; }
.avea-card ul { margin: 0; padding-left: 1.2rem; }
.avea-card li { margin-bottom: .45rem; }

/* status pills */
.pill {
    display: inline-block; border-radius: 999px; padding: .2rem .85rem;
    font-size: .8rem; font-weight: 700; background: #F0EEEA; color: #6B6B6B;
}
.pill.done { background: rgba(45,165,182,.13); color: #1B7F8C; }
.pill.err  { background: rgba(212,55,71,.12);  color: #B02A38; }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def _logo_b64() -> str:
    import base64
    from pathlib import Path
    p = Path(__file__).parent / "static" / "avea_logo.png"
    return base64.b64encode(p.read_bytes()).decode()


try:
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64()}" alt="AVEA"/>'
except Exception:
    _logo_html = "AVEA"  # fall back to text if the asset is missing
st.markdown(
    f'<div class="avea-header">{_logo_html}'
    '<span class="accent">AI Visibility Analyzer</span></div>',
    unsafe_allow_html=True,
)

# Demo-first navigation: the app opens on Ask with the sidebar collapsed;
# the other pages are one click away in the sidebar.
with st.sidebar:
    page = st.radio(
        "Section",
        ["Ask", "Settings", "Ingest", "Documents", "Compare two URLs"],
    )
    mode = "Compare"
    if page == "Ask":
        mode = st.selectbox(
            "Mode",
            [
                "Compare — surrogate vs ChatGPT + Claude",
                "Surrogate — 7-tool deep-research agent",
                "Retrieve only — search ingested docs",
            ],
            help=(
                "Compare: the brand-visibility demo (3 systems in parallel). "
                "Surrogate: just our agent, with visible reasoning. "
                "Retrieve only: local doc search, no model call."
            ),
        )
    st.divider()
    test_mode = st.toggle(
        "Test mode",
        value=True,
        help=(
            "ON by default: renders the Compare results from canned data (a "
            "real past run), no model calls — ideal for demos and when the GPU "
            "box is down. Turn OFF to run the question live."
        ),
    )

# ----- INGEST ----------------------------------------------------------------

if page == "Ingest":
    _rag = _need_rag()
    ingest_text, ingest_url = _rag.ingest_text, _rag.ingest_url
    st.subheader("Ingest URLs")
    urls_text = st.text_area(
        "Paste one URL per line",
        height=160,
        placeholder="https://example.com/page-1\nhttps://example.com/page-2",
    )
    overwrite = st.checkbox(
        "Overwrite if URL already in store", value=False, key="ov_url"
    )
    if st.button("Ingest URLs", type="primary"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.warning("No URLs to ingest.")
        else:
            for u in urls:
                with st.spinner(f"Fetching {u} ..."):
                    try:
                        r = ingest_url(u, overwrite=overwrite)
                        st.write(r)
                    except Exception as e:
                        st.error(f"{u}: {e!r}")

    st.divider()
    st.subheader("Ingest raw text")
    raw = st.text_area("Paste text", height=180)
    label = st.text_input("Source label", value="pasted text")
    if st.button("Ingest text"):
        if not raw.strip():
            st.warning("No text to ingest.")
        else:
            with st.spinner("Embedding ..."):
                r = ingest_text(raw, source_label=label, overwrite=True)
                st.write(r)

# ----- DOCUMENTS --------------------------------------------------------------

if page == "Documents":
    _rag = _need_rag()
    delete_doc, list_docs = _rag.delete_doc, _rag.list_docs
    rows = list_docs()
    if not rows:
        st.info("Store is empty. Ingest something on the Ingest tab.")
    else:
        st.write(f"**{len(rows)} document(s) in the store**")
        for r in rows:
            with st.expander(
                f"[{r['id']}] {r['title']}  "
                f"({r['n_chunks']} chunks · {r['raw_chars']} chars · {r['source_type']})"
            ):
                st.markdown(f"**URL/label:** {r['url']}")
                st.markdown(f"**Fetched at:** {r['fetched_at']}")
                if st.button(f"Delete doc {r['id']}", key=f"del_{r['id']}"):
                    if delete_doc(r["id"]):
                        st.success(f"Deleted doc {r['id']}; refresh tab to see it gone.")

# ----- ASK --------------------------------------------------------------------

if page == "Ask":
    # `mode` comes from the sidebar dropdown. The form makes Enter in any
    # text field submit the run; mode-specific controls swap live because
    # the selectbox lives outside the form.
    brand, k, compare_glm = "Avea", 5, False
    with st.form("ask_form", border=False):
        q = st.text_input(
            "Your question",
            value="What are the top 10 Swiss supplement brands?" if test_mode else "",
            placeholder="e.g. which restaurant is the best for italian food in Tashkent",
        )
        if mode.startswith("Compare"):
            brand = st.text_input(
                "Brand to track",
                value="Avea",
                help=(
                    "Compare mode checks whether this brand appears in each "
                    "system's ranked answer and generates the action plan for it."
                ),
            )
        elif mode.startswith("Surrogate"):
            compare_glm = st.checkbox(
                "Also call GLM reference (same evidence, side-by-side)",
                value=False,
                help=(
                    "After the surrogate runs, send the SAME evidence pack the "
                    "surrogate gathered (tool outputs from `search`, `fetch_url`, "
                    "`extract_entity`) to the GLM reference (z.ai) and show both "
                    "answers side-by-side. Apples-to-apples fidelity comparison."
                ),
            )
        else:
            k = st.slider(
                "Top-k retrieved chunks", 1, 10, 5,
                help="Number of chunks returned from the user-RAG store.",
            )
        submitted = st.form_submit_button("Run", type="primary")

    if submitted and not q.strip():
        st.warning("Type a question first.")
    if submitted and q.strip():
        if mode.startswith("Retrieve only"):
            retrieve = _need_rag().retrieve
            with st.spinner("Embedding question + searching store ..."):
                hits = retrieve(q, k=k)
            if not hits:
                st.warning("No matches above the score threshold. Try a different question or ingest more sources.")
            else:
                for i, h in enumerate(hits, 1):
                    with st.expander(
                        f"#{i}  score={h['score']:.3f}  · {h['title']}  ·  {h['url']}"
                    ):
                        st.text(h["text"])
        elif mode.startswith("Compare"):
            try:
                from surrogate.compare import compare_run
            except Exception as e:
                st.error(f"could not import compare module: {e!r}")
                st.stop()
            if test_mode:
                st.info(
                    "Test mode — rendering canned results from a real past "
                    "run. No model is called."
                )
            else:
                st.info(
                    "Running the same question through three systems in parallel — "
                    "the surrogate (needs vLLM at localhost:8000), ChatGPT (gpt-5 + "
                    "web_search) and Claude (web_search + extended thinking). "
                    "Typically 3–5 minutes."
                )

            # --- live status strip --------------------------------------------
            labels = {"surrogate": "Surrogate", "openai": "ChatGPT",
                      "claude": "Claude", "judge": "Match scoring",
                      "deep": "Deep analysis"}
            def _pill(label, state):
                cls = {"done": " done", "error": " err"}.get(state, "")
                text = {"done": "done", "running": "running…",
                        "waiting": "waiting"}.get(state, state)
                return (f'<span class="pill{cls}">{label} · {text}</span>')

            cols = st.columns(5)
            slots = {}
            for col, key in zip(cols, ["surrogate", "openai", "claude",
                                       "judge", "deep"]):
                slots[key] = col.empty()
                if test_mode:
                    slots[key].markdown(_pill(labels[key], "done"),
                                        unsafe_allow_html=True)
                else:
                    init = "running" if key in ("surrogate", "openai", "claude") else "waiting"
                    slots[key].markdown(_pill(labels[key], init),
                                        unsafe_allow_html=True)

            def _status(system, state):
                label = labels.get(system, system)
                if state not in ("done", "running", "waiting"):
                    state = "error"
                slots[system].markdown(_pill(label, state),
                                       unsafe_allow_html=True)

            if test_mode:
                import json as _json
                try:
                    with open("data/demo_fixture.json") as f:
                        rec = _json.load(f)
                except FileNotFoundError:
                    st.error("data/demo_fixture.json missing — run "
                             "`python scripts/build_demo_fixture.py` once to create it.")
                    st.stop()
            else:
                with st.spinner("Comparing — surrogate + 2 frontiers in parallel…"):
                    try:
                        rec = compare_run(q, brand=brand.strip() or "Avea",
                                          status_cb=_status)
                    except Exception as e:
                        st.error(f"compare run failed: {e!r}")
                        st.text(traceback.format_exc())
                        st.stop()

            sys_ = rec["systems"]
            m = rec["matches"]
            if rec["errors"]:
                st.warning("Some systems failed: "
                           + " · ".join(f"{labels.get(k, k)} ({v})"
                                        for k, v in rec["errors"].items()))

            # --- ranked-picks table (✓ + bold only in frontier columns) -------
            sur_list = sys_["surrogate"]["ranked"]
            oai_list = sys_["openai"]["ranked"]
            cla_list = sys_["claude"]["ranked"]
            oai_matched = {str(p[1]).lower()
                           for p in (m["sur_openai"].get("matched_pairs") or [])
                           if len(p) >= 2}
            cla_matched = {str(p[1]).lower()
                           for p in (m["sur_claude"].get("matched_pairs") or [])
                           if len(p) >= 2}

            def _fr_cell(pick, matched):
                if not pick:
                    return ""
                if pick.lower() in matched:
                    return f"**{pick}** ✓"
                return pick

            n_rows = max(len(sur_list), len(oai_list), len(cla_list))
            lines = ["| # | Surrogate | ChatGPT | Claude |",
                     "|---|-----------|---------|--------|"]
            for i in range(n_rows):
                s_c = sur_list[i] if i < len(sur_list) else ""
                o_c = _fr_cell(oai_list[i] if i < len(oai_list) else "", oai_matched)
                c_c = _fr_cell(cla_list[i] if i < len(cla_list) else "", cla_matched)
                lines.append(f"| {i + 1} | {s_c} | {o_c} | {c_c} |")
            st.markdown("\n".join(lines))
            st.caption(
                f"✓ = brand-level match with the surrogate's list · "
                f"Surrogate↔ChatGPT {m['sur_openai']['overlap']}/{len(m['sur_openai']['a'])} · "
                f"Surrogate↔Claude {m['sur_claude']['overlap']}/{len(m['sur_claude']['a'])} · "
                f"ChatGPT↔Claude {m['openai_claude']['overlap']}/{len(m['openai_claude']['a'])}"
            )

            # --- suggestions panel (the headline feature) ----------------------
            import html as _html
            sug = rec["suggestions"]
            card = [
                '<div class="avea-card">',
                f"<h3>What {_html.escape(rec['brand'])} should do</h3>",
                f"<p>{_html.escape(sug['why'])}</p>",
                "<ul>",
            ]
            for a in sug["actions"]:
                card.append(f"<li>{_html.escape(a)}</li>")
            card.append("</ul></div>")
            st.markdown("".join(card), unsafe_allow_html=True)

            # --- transparency timeline (how the surrogate reached this) --------
            # Rendered as an isolated HTML component so the click-to-reveal
            # interaction works (st.markdown strips the onclick JS). Shares the
            # exact renderer with the static demo via surrogate.demo_render.
            traj = rec["systems"]["surrogate"].get("trajectory") or []
            if traj:
                import streamlit.components.v1 as _components
                from surrogate.demo_render import (
                    trajectory_component_html, estimate_height,
                )
                st.markdown("### How our surrogate reached this")
                st.caption(
                    "Every step the model took — its live reasoning bound to the "
                    "action it triggered. Click any highlighted phrase to reveal "
                    "the exact sources the model pulled. The frontier models "
                    "don't expose this."
                )
                _components.html(
                    trajectory_component_html(traj),
                    height=estimate_height(traj),
                    scrolling=True,
                )

            # --- deeper suggestions (rubric-driven analyst output) -------------
            deep = rec.get("deep")
            if deep:
                with st.expander("Deeper suggestions — competitive gaps, "
                                 "priority plan, rival deep-dive", expanded=False):
                    st.markdown(deep.get("summary", ""))

                    gaps = deep.get("competitive_gaps") or []
                    if gaps:
                        st.markdown("#### What winning brands have that you don't")
                        rows = ["| Asset | Who has it | Why it wins AI visibility | Your gap |",
                                "|---|---|---|---|"]
                        for g in gaps:
                            who = ", ".join(g.get("brands_with_it") or [])
                            rows.append(
                                f"| **{g.get('asset', '')}** | {who} | "
                                f"{g.get('why_it_matters', '')} | "
                                f"{g.get('gap_for_brand', '')} |"
                            )
                        st.markdown("\n".join(rows))

                    plan = deep.get("priority_plan") or []
                    if plan:
                        st.markdown("#### Do this first — priority order")
                        for p in sorted(plan, key=lambda x: x.get("rank", 99)):
                            st.markdown(
                                f"**{p.get('rank', '?')}.** {p.get('action', '')}  \n"
                                f"&nbsp;&nbsp;&nbsp;*{p.get('horizon', '')} · "
                                f"{p.get('effort', '')} effort · "
                                f"{p.get('impact', '')}*"
                            )

                    rivals = deep.get("rival_deep_dive") or []
                    if rivals:
                        st.markdown("#### Rival deep-dive")
                        for r in rivals:
                            st.markdown(
                                f"**{r.get('brand', '')}**  \n"
                                f"*Why AI ranks them:* {r.get('why_ai_ranks_them', '')}  \n"
                                f"*Their visible assets:* {r.get('their_visible_assets', '')}  \n"
                                f"*How to compete:* {r.get('how_to_compete', '')}"
                            )
            elif not test_mode:
                st.caption("Deeper analysis unavailable for this run.")

            # --- collapsed detail ----------------------------------------------
            with st.expander("Sources each model consulted"):
                o_urls = sorted(set(sys_["openai"].get("urls") or []))
                c_urls = sorted(set(sys_["claude"].get("urls") or []))
                st.markdown(f"**ChatGPT cited {len(o_urls)} URL(s):**")
                for u in o_urls:
                    st.markdown(f"- {u}")
                st.markdown(f"**Claude consulted {len(c_urls)} URL(s):**")
                for u in c_urls:
                    st.markdown(f"- {u}")
                if sys_["surrogate"].get("bundle"):
                    st.markdown(f"**Surrogate trace bundle:** "
                                f"`{sys_['surrogate']['bundle']}`")

            with st.expander("Full answers & reasoning (verbatim)"):
                for key in ("surrogate", "openai", "claude"):
                    s = sys_[key]
                    st.markdown(f"### {labels[key]} · {s.get('model', '')}")
                    if s.get("error"):
                        st.error(s["error"])
                        continue
                    st.write(s.get("answer") or "(empty)")
                    if s.get("thinking"):
                        # Surrogate reasoning is shown step-by-step (with its
                        # sources) in the timeline above — don't dump it again.
                        if key == "surrogate" and traj:
                            st.caption("↑ Step-by-step reasoning with sources is "
                                       "in the timeline above.")
                        else:
                            st.markdown("**Reasoning (verbatim):**")
                            st.text(s["thinking"])
        else:
            # New single-stage ReAct loop with the 7-tool engineered workflow.
            try:
                from surrogate.loop import run as loop_run
                from surrogate.loop_tools import default_tools
                from surrogate.backtest import (
                    _evidence_pack_from_bundle, _concat_thinking,
                )
            except Exception as e:
                st.error(f"could not import loop modules: {e!r}")
                st.stop()
            st.info(
                "Running the deep-research agent: a single ReAct loop with "
                "`search` → `fetch_url`/`extract_entity` → `verify_fact` → "
                "`check_missing_fields` → `think` → `stop_and_answer`. "
                "Needs the vLLM endpoint at localhost:8000 (keep_tunnel.sh)."
            )
            # --- 1. try the surrogate run; capture (not raise) any failure ---
            res = None
            surrogate_error = None
            surrogate_tb = None
            try:
                with st.spinner("Running surrogate agent (may take 1–3 min)…"):
                    res = loop_run(q, tools=default_tools())
            except Exception as e:
                surrogate_error = e
                surrogate_tb = traceback.format_exc()

            if res is not None:
                st.success(
                    f"Done — termination={res.termination}, "
                    f"steps={res.steps}, bundle: {res.bundle_dir}"
                )

            # --- 2. if requested, call GLM with the surrogate's evidence ---
            ref = None
            ref_error = None
            ref_used_full_evidence = False
            if compare_glm:
                from surrogate.reference import ask_reference
                from surrogate.two_stage import STAGE2_SYSTEM

                if res is not None:
                    # SURROGATE OK -> apples-to-apples: reconstruct the
                    # evidence pack from the loop's trace and feed it to GLM
                    # with the strict "use only this evidence" system prompt.
                    stage2_user = _evidence_pack_from_bundle(q, res.bundle_dir)
                    ref_used_full_evidence = True
                    ref_sys = STAGE2_SYSTEM
                    ref_spinner = "Calling GLM reference (same evidence)…"
                else:
                    # SURROGATE FAILED -> let GLM answer from memory alone
                    # (no system prompt that mandates evidence-only, since
                    # there is none — would force a refusal otherwise).
                    stage2_user = q
                    ref_sys = None
                    ref_spinner = ("Calling GLM in bare mode (no surrogate "
                                   "evidence; GLM answers from memory)…")

                with st.spinner(ref_spinner):
                    try:
                        ref = ask_reference(question=stage2_user, system=ref_sys)
                    except Exception as e:
                        ref_error = e

            # --- 3. render ---
            def _show_friendly_surrogate_error():
                name = type(surrogate_error).__name__
                low = str(surrogate_error).lower()
                if ("connection" in low or "refused" in low
                        or "apiconnection" in name.lower()
                        or "timeout" in name.lower()):
                    st.warning(
                        "Surrogate not reachable — the GPU box is probably "
                        "paused, the SSH tunnel is down, or vLLM isn't "
                        "serving. Open **Settings** to check status or "
                        "spin the box up.",
                        icon="🔌",
                    )
                else:
                    st.error(f"Surrogate failed: {surrogate_error!r}", icon="🔴")
                with st.expander("debug detail"):
                    st.text(surrogate_tb or "(no traceback)")

            surrogate_model = os.environ.get("STAGE2_MODEL", "qwen3-8b")
            if compare_glm:
                col_s, col_r = st.columns(2)
                with col_s:
                    if res is not None:
                        st.subheader(f"Surrogate · {surrogate_model}")
                        st.write(res.final_answer or "(empty)")
                        thinking = _concat_thinking(res.messages)
                        if thinking:
                            with st.expander("thinking (all <think> blocks, verbatim)"):
                                st.text(thinking)
                    else:
                        st.subheader("Surrogate · not run")
                        _show_friendly_surrogate_error()
                with col_r:
                    if ref_error is not None:
                        st.subheader("Reference · call failed")
                        st.error(f"GLM call failed: {ref_error!r}", icon="🔴")
                    elif ref is None:
                        st.subheader("Reference · skipped")
                    else:
                        title = f"Reference · {ref['model']}"
                        if not ref_used_full_evidence:
                            title += "  (no surrogate evidence — bare GLM)"
                        st.subheader(title)
                        st.write(ref["answer"] or "(empty)")
                        if ref.get("thinking"):
                            with st.expander("thinking (verbatim)"):
                                st.text(ref["thinking"])
            else:
                if res is not None:
                    st.subheader(f"Surrogate answer · {surrogate_model}")
                    st.write(res.final_answer or "(empty)")
                    thinking = _concat_thinking(res.messages)
                    if thinking:
                        with st.expander("thinking (all <think> blocks, verbatim)"):
                            st.text(thinking)
                else:
                    _show_friendly_surrogate_error()


# ----- COMPARE TWO URLs (the DOM-crawler presentation flow) -----------------

if page == "Compare two URLs":
    st.subheader(
        "Compare two URLs",
        help=(
            "Paste two web pages. We crawl each one's full DOM (richer than "
            "the agent's `fetch_url`: headings, lists, tables, ratings/prices, "
            "top links), pack both into the model's evidence, and Stage 2 "
            "answers + thinks over them. No tools are called — the surrogate "
            "reasons over exactly the two pages you chose."
        ),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        url_a = st.text_input("Website A", placeholder="https://...")
    with col_b:
        url_b = st.text_input("Website B", placeholder="https://...")
    dom_q = st.text_input(
        "Question",
        placeholder="e.g. which restaurant is the best for italian food in Tashkent",
        key="dom_q",
    )
    if st.button("Run DOM-pair surrogate", type="primary",
                 disabled=not (url_a.strip() and url_b.strip() and dom_q.strip())):
        try:
            from surrogate.two_stage import run_with_dom_pair
            from surrogate.tools.dom import crawl_dom
        except Exception as e:
            st.error(f"import failure: {e!r}"); st.stop()

        st.info(
            "Crawling both DOMs, then running Stage 2 over both. Needs the "
            "vLLM endpoint at localhost:8000 (keep_tunnel.sh)."
        )
        try:
            with st.spinner("Crawling + running Stage 2 ..."):
                res = run_with_dom_pair(dom_q, url_a, url_b)
            st.success(f"Done — bundle: {res['bundle_dir']}")

            st.subheader("Surrogate answer")
            st.write(res["answer"] or "(empty)")
            if res.get("thinking"):
                with st.expander("Stage 2 thinking (verbatim, sample 0 greedy)"):
                    st.text(res["thinking"])

            st.subheader(f"DOM extracts ({len(res['samples'])} stage-2 samples ran)")
            c1, c2 = st.columns(2)
            with c1:
                pair = res["dom_pair"]["a"]
                st.markdown(f"**A — {pair['title'] or '(no title)'}**")
                st.markdown(f"`{pair['url']}` · ok={pair['ok']}")
                with st.expander("Extracted structure (verbatim)"):
                    st.text(pair["as_text"])
            with c2:
                pair = res["dom_pair"]["b"]
                st.markdown(f"**B — {pair['title'] or '(no title)'}**")
                st.markdown(f"`{pair['url']}` · ok={pair['ok']}")
                with st.expander("Extracted structure (verbatim)"):
                    st.text(pair["as_text"])

            with st.expander(f"All {len(res['samples'])} stage-2 samples (verbatim)"):
                for s in res["samples"]:
                    st.markdown(f"**sample {s['sample_index']}  T={s['temperature']}  {s['duration_s']:.1f}s**")
                    if s.get("reasoning"):
                        st.text("--- thinking ---\n" + s["reasoning"])
                    st.text("--- answer ---\n" + (s["content"] or ""))
                    st.markdown("---")
        except Exception as e:
            st.error(f"DOM-pair run failed: {e!r}")
            st.text(traceback.format_exc())


# ----- SETTINGS (box config + tunnel control) -------------------------------

if page == "Settings":
    from surrogate import box

    # Push any persisted settings into os.environ on every page load so the
    # running process always reflects the JSON (e.g. REFERENCE_MODEL).
    box.apply_to_env()

    st.subheader(
        "Remote GPU box & tunnel",
        help=(
            "Where the surrogate's vLLM endpoint lives. Saved to "
            "`box_config.json` (gitignored) so 'last used' values persist "
            "between sessions. The tunnel keeper (`scripts/keep_tunnel.sh`) "
            "is restarted with these values when you click Save & Restart. "
            "The reference-model field controls which GLM model surrogate."
            "reference and the backtest harness use; it takes effect on the "
            "next call, no restart."
        ),
    )

    status = box.status_summary()
    s = status["settings"]
    last = s.get("last_used", "never saved")

    # Compact status row — short labels, tooltips for detail.
    c1, c2, c3 = st.columns(3)
    with c1:
        if status["keeper_running"]:
            st.success("Tunnel: running",
                       icon="🟢")
        else:
            st.warning("Tunnel: stopped", icon="⚪")
    with c2:
        ep = status["endpoint"]
        if ep["ok"]:
            st.success(f"Endpoint: {ep['model']}", icon="🟢")
        else:
            st.error("Endpoint: unreachable", icon="🔴")
    with c3:
        st.info(f"Last saved: {last}", icon="💾")

    st.divider()

    ADD_NEW = "+ Add new preset…"
    all_presets = box.all_presets()
    preset_labels = [
        n if box.is_factory_preset(n) else f"{n}  · custom"
        for n in all_presets.keys()
    ] + [ADD_NEW]
    label_to_name = {lbl: name for lbl, name in zip(preset_labels, all_presets.keys())}

    cp1, cp2 = st.columns([3, 1], vertical_alignment="bottom")
    with cp1:
        chosen_label = st.selectbox(
            "Quick preset",
            preset_labels,
            index=0,
            help=("Pick a preset to fill the form below. "
                  "Pick '+ Add new preset…' to create one — Save / Save & "
                  "Connect will then also persist it under the name you give. "
                  "Custom presets can be deleted from here."),
        )
    creating_new = (chosen_label == ADD_NEW)
    selected_name = None if creating_new else label_to_name[chosen_label]

    with cp2:
        if not creating_new:
            if st.button("Load into form", use_container_width=True):
                preset = all_presets[selected_name].copy()
                for k, v in preset.items():
                    st.session_state[f"box_{k}"] = v
                st.rerun()
        # In "+ Add new" mode the right column stays empty — the form below
        # with its "Name for the new preset" field makes the next step obvious.

    # When a custom preset is selected, offer a one-click delete here too.
    if selected_name is not None and not box.is_factory_preset(selected_name):
        if st.button(f"🗑 Delete preset '{selected_name}'"):
            if box.delete_user_preset(selected_name):
                st.success(f"Deleted '{selected_name}'.")
                st.rerun()

    def _sv(key: str, default):
        return st.session_state.get(f"box_{key}", s.get(key, default))

    with st.form("box_form"):
        if creating_new:
            new_preset_name = st.text_input(
                "Name for the new preset",
                key="new_preset_name",
                placeholder="e.g. vast-4090-may, lab-h100, mithril-backup",
                help="When you click Save or Save & Connect, the form values "
                     "below will also be persisted under this name and show "
                     "up in the dropdown next time.",
            )
        else:
            new_preset_name = ""
        ca, cb = st.columns(2)
        with ca:
            host = st.text_input(
                "Host", value=_sv("host", ""),
                key="box_host", placeholder="44.250.249.199",
                help="The remote box's IP or hostname.",
            )
            user = st.text_input(
                "SSH user", value=_sv("user", "ubuntu"),
                key="box_user",
                help="Mithril → ubuntu. vast.ai → root. Most clouds → ubuntu.",
            )
            port = st.number_input(
                "SSH port", min_value=1, max_value=65535,
                value=int(_sv("port", 22)), step=1, key="box_port",
                help="Mithril → 22. vast.ai → look at the provider's SSH command.",
            )
        with cb:
            key = st.text_input(
                "SSH key", value=_sv("key", ""), key="box_key",
                placeholder="/Users/you/ulusha-key.pem",
                help=(
                    "Path to your private key .pem. Leave empty to fall back "
                    "to ~/.ssh/config or the ssh-agent."
                ),
            )
            local_port = st.number_input(
                "Local port", min_value=1, max_value=65535,
                value=int(_sv("local_port", 8000)), step=1,
                key="box_local_port",
                help=(
                    "Mac side of the tunnel. The rest of the code expects "
                    "8000 — only change this if you know why."
                ),
            )
        st.markdown("")  # small spacer
        reference_model = st.text_input(
            "Reference model (GLM)",
            value=_sv("reference_model", "glm-4.6"),
            key="box_reference_model",
            placeholder="glm-4.6",
            help=(
                "GLM model used by surrogate.reference and the backtest "
                "harness (the 'frontier' comparator). Common values: "
                + ", ".join(box.REFERENCE_MODEL_SUGGESTIONS)
                + ". Changing this takes effect on the next call — no tunnel "
                "restart needed."
            ),
        )
        cs1, cs2, cs3 = st.columns(3)
        test_only = cs1.form_submit_button("🔌 Test endpoint")
        save_only = cs2.form_submit_button("💾 Save")
        save_restart = cs3.form_submit_button(
            "🚀 Save & Connect", type="primary",
            help=(
                "Saves, then makes the box ready for use: install vLLM + "
                "ninja if missing, launch Qwen3-8B, restart the tunnel, "
                "wait until the endpoint is alive. Idempotent — skips "
                "anything already done. Takes up to ~10 min on a fresh box."
            ),
        )

    if save_only or save_restart:
        new = {
            "host": host.strip(), "user": user.strip() or "ubuntu",
            "port": int(port), "key": key.strip(),
            "local_port": int(local_port),
            "reference_model": reference_model.strip() or "glm-4.6",
        }
        saved = box.save_settings(new)
        # If we're in "Add new" mode, ALSO persist these values as a named
        # preset so they show up in the dropdown next time.
        if creating_new:
            name = (new_preset_name or "").strip()
            if not name:
                st.warning("Pick a name in 'Name for the new preset' to also "
                           "save these values as a preset (active config still "
                           "saved).")
            elif box.is_factory_preset(name):
                st.warning(f"'{name}' clashes with a factory preset name. "
                           "Active config saved; preset not created.")
            else:
                box.save_user_preset(name, new)
                st.success(f"Preset '{name}' saved · also active config "
                           f"({saved['last_used']})", icon="💾")
        else:
            st.success(
                f"Saved · {saved['last_used']} · reference = {saved['reference_model']}",
                icon="💾",
            )
        if save_restart:
            # Provision the box (idempotent), restart the tunnel, verify.
            # One row per stage that updates IN PLACE — no scrollback wall.
            with st.status(
                f"connecting to {saved['host']}…",
                expanded=True,
            ) as status:
                slots: dict[str, "object"] = {}

                def on_progress(stage, msg, ok=True):
                    if stage not in slots:
                        slots[stage] = status.empty()
                    mark = "" if ok else " ⚠"
                    slots[stage].markdown(f"`{stage}`  {msg}{mark}")
                    short = msg if len(msg) <= 70 else msg[:67] + "…"
                    status.update(label=f"{stage}: {short}")

                prov = box.provision_remote(saved, on_progress=on_progress)

                if prov.get("ok"):
                    on_progress("tunnel", "restarting…")
                    box.restart_tunnel(saved)
                    time.sleep(2)
                    ep = box.is_endpoint_alive(local_port=saved["local_port"])
                    if ep["ok"]:
                        on_progress("tunnel", f"connected · {ep['model']}")
                        status.update(
                            label=f"connected · {ep['model']} on {saved['host']}",
                            state="complete", expanded=False,
                        )
                    else:
                        on_progress("tunnel",
                                    f"endpoint not answering yet: {ep['error']}", ok=False)
                        status.update(
                            label="provisioned but endpoint not verified",
                            state="error", expanded=True,
                        )
                else:
                    err = (prov.get("error") or "")[:200]
                    status.update(
                        label=f"failed at {prov.get('stage')}: {err}",
                        state="error", expanded=True,
                    )
    elif test_only:
        # Probe the values the user TYPED (not the saved/active tunnel).
        # SSHes straight to the entered host and asks the box whether vLLM
        # is up — so the user can validate a config before persisting it.
        candidate = {
            "host": host.strip(), "user": user.strip() or "ubuntu",
            "port": int(port), "key": key.strip(),
            "local_port": int(local_port),
        }
        with st.spinner(f"SSHing to {candidate['host']}:{candidate['port']} …"):
            res = box.probe_remote(candidate)

        # Compact two-line status — SSH layer + vLLM layer
        if res["ssh_ok"] and res["vllm_ok"]:
            st.success(
                f"✅ SSH reached **{candidate['host']}**  ·  "
                f"✅ vLLM responding (`{res['model']}`)",
                icon="🟢",
            )
            st.info(
                "Looks good — click **💾 Save** (or **🔄 Save & Restart**) "
                "above to keep these values and point the tunnel at this box.",
                icon="💡",
            )
        elif res["ssh_ok"] and not res["vllm_ok"]:
            st.warning(
                f"✅ SSH reached **{candidate['host']}**  ·  "
                "⚠️ vLLM not running on the box yet",
                icon="🟡",
            )
            with st.expander("What to do"):
                st.write(
                    "The box is reachable but no vLLM is serving on port "
                    "8000 there. Two routes:\n\n"
                    "- **Provision the box** — install vLLM and launch "
                    "Qwen3-8B on it. From the Mac terminal: SSH in and run "
                    "the commands in `commands.txt` *Phase 0 — On the GPU "
                    "box*, or ask the assistant to drive the provisioning "
                    "for you over SSH.\n"
                    "- If vLLM IS running but on a different port, change "
                    "the launch command to use `--port 8000` (the rest of "
                    "the code expects that)."
                )
        else:
            st.error(f"❌ Can't reach the box over SSH", icon="🔴")
            with st.expander("Error detail / common causes"):
                st.code(res["error"])
                st.write(
                    "Common causes:\n"
                    "- **Host / port wrong** — double-check the provider's "
                    "SSH command.\n"
                    "- **Key wrong** — leave empty if the box uses your "
                    "ssh-agent / `~/.ssh/config` (vast.ai), or set the full "
                    "path to the `.pem` if the provider issued one "
                    "(Mithril).\n"
                    "- **Box paused / destroyed** — check the provider "
                    "dashboard.\n"
                    "- **First-time host key prompt** — we accept "
                    "new keys automatically; if you see a different "
                    "host-key error, that's a man-in-the-middle warning "
                    "and worth investigating."
                )

    st.divider()
    with st.expander("Tunnel keeper log (last 40 lines)"):
        try:
            lines = box.KEEPER_LOG.read_text().splitlines()[-40:]
            st.code("\n".join(lines) or "(empty)")
        except FileNotFoundError:
            st.write("(no log yet)")
