"""Streamlit UI for the surrogate deep-research agent.

Tabs:
  • Ingest / Documents — manage a local user-RAG store (sentence-transformers
    over SQLite). Not used by the deep-research agent below; kept available
    for retrieve-only queries and future composition.
  • Ask — ask a question. Two modes:
      - Retrieve only: cosine search over the user-RAG store (no model call).
      - Surrogate: single-stage ReAct loop, 7 tools (search, fetch_url,
        extract_entity, verify_fact, check_missing_fields, think,
        stop_and_answer). Visible <think> reasoning between tool calls.
        Optional side-by-side comparison vs GLM-4.6 on the SAME evidence.
  • Compare two URLs — DOM-pair flow (Stage 2 bypassed).
  • Settings — box/tunnel/reference-model config.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import textwrap
import time
import traceback

import streamlit as st

from surrogate.rag import (
    delete_doc, ingest_text, ingest_url, list_docs, retrieve,
)

st.set_page_config(page_title="Surrogate · Deep-Research Agent", layout="wide")
st.title(
    "Surrogate · Deep-Research Agent",
    help=(
        "Open-source surrogate of a frontier deep-research assistant. Single "
        "ReAct loop on Qwen3-8B with 7 tools (search, fetch_url, "
        "extract_entity, verify_fact, check_missing_fields, think, "
        "stop_and_answer). Visible <think> reasoning between every tool call. "
        "Compare side-by-side against GLM-4.6 on the same evidence to "
        "measure reasoning fidelity."
    ),
)

tab_ingest, tab_docs, tab_ask, tab_dom, tab_settings = st.tabs(
    ["Ingest", "Documents", "Ask", "Compare two URLs", "Settings"]
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
        [
            "Retrieve only — search ingested docs (no model call)",
            "Surrogate — deep-research agent (7-tool ReAct loop)",
        ],
        horizontal=False,
    )
    k = st.slider(
        "Top-k retrieved chunks (Retrieve-only mode)", 1, 10, 5,
        help="Only used in 'Retrieve only' mode — number of chunks from the user-RAG store.",
    )
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

with tab_dom:
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

with tab_settings:
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
