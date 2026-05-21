"""Tiny Streamlit UI for the user-RAG + surrogate.

Three things only — paste URLs/text to ingest, list/delete docs, ask a question.
Questions can be answered RAG-only (just retrieved chunks, no model run) or
through the full surrogate two-stage pipeline with RAG evidence appended to
Stage 2.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import textwrap
import traceback

import streamlit as st

from surrogate.rag import (
    delete_doc, ingest_text, ingest_url, list_docs, retrieve,
)

st.set_page_config(page_title="Surrogate + User-RAG", layout="wide")
st.title("Surrogate + User-RAG")
st.caption(
    "Bring your own web links into the surrogate's evidence. Stage 1's "
    "web search still runs; user chunks are appended to Stage 2's evidence "
    "pack so the answer is grounded in BOTH the live web and your own sources."
)

tab_ingest, tab_docs, tab_ask, tab_dom = st.tabs(
    ["Ingest", "Documents", "Ask", "Compare two URLs"]
)

# ----- INGEST ----------------------------------------------------------------

with tab_ingest:
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

with tab_docs:
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

with tab_ask:
    q = st.text_input(
        "Your question",
        placeholder="e.g. which restaurant is the best for italian food in Tashkent",
    )
    mode = st.radio(
        "Mode",
        ["Retrieve only (no model call)",
         "Surrogate + RAG (full two-stage, hits remote GPU via tunnel)"],
        horizontal=False,
    )
    k = st.slider("Top-k retrieved chunks", 1, 10, 5)

    if st.button("Run", type="primary", disabled=not q.strip()):
        if mode.startswith("Retrieve only"):
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
        else:
            # Full surrogate run (requires the vLLM endpoint / SSH tunnel up).
            try:
                from surrogate.two_stage import run_two_stage
            except Exception as e:
                st.error(f"could not import two_stage: {e!r}")
                st.stop()
            st.info(
                "Running Stage 1 (tools) + Stage 2 (reasoning with RAG-augmented "
                "evidence). Needs the vLLM endpoint at localhost:8000 to be up "
                "(keep_tunnel.sh)."
            )
            try:
                with st.spinner("Running surrogate two-stage ..."):
                    res = run_two_stage(q, use_rag=True, rag_k=k)
                st.success(f"Done — bundle: {res.bundle_dir}")
                st.subheader("Stage 2 answer")
                st.write(res.stage2.answer or "(empty)")
                if res.stage2.reasoning:
                    with st.expander("Stage 2 thinking (verbatim)"):
                        st.text(res.stage2.reasoning)
                with st.expander("Stage 1 answer (for reference)"):
                    st.write(res.stage1.answer or "(empty)")
            except Exception as e:
                st.error(f"surrogate run failed: {e!r}")
                st.text(traceback.format_exc())


# ----- COMPARE TWO URLs (the DOM-crawler presentation flow) -----------------

with tab_dom:
    st.subheader("Compare two URLs")
    st.caption(
        "Paste two web pages. We crawl each one's full DOM (richer than the "
        "agent's `fetch_url`: headings, lists, tables, ratings/prices, top "
        "links), pack both into the model's evidence, and Stage 2 answers + "
        "thinks over them. No tools are called — the surrogate reasons over "
        "exactly the two pages you chose."
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
