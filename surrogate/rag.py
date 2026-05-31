"""User-RAG ingestion + retrieval.

Lets a user bring their own web links (or pasted text) into a local retrieval
store. At query time, the top-k matching chunks are returned and injected into
the surrogate's Stage 2 evidence pack alongside whatever Stage 1's tools
gathered — so the answer is grounded in BOTH the live web evidence AND the
user's own provided sources.

Local-only by design: no cloud calls in the RAG path. Storage is a single
SQLite file under `userlinks/` (gitignored). Embeddings come from
sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) running on CPU/MPS — small,
fast, no API key.

Schema:
    docs(   id PK, url, title, source_type, fetched_at, n_chunks, raw_chars )
    chunks( id PK, doc_id FK, idx, text, embedding BLOB )

Embeddings are float32 numpy bytes. Retrieval is brute-force cosine over the
chunk matrix — fine up to ~10k chunks (covers any realistic v0 use).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

# Lazy imports inside functions where heavy (sentence_transformers, etc.).

USERLINKS_DIR = Path("userlinks")
DB_PATH = USERLINKS_DIR / "store.sqlite"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# Chunk sizing is in CHARACTERS (cheap and predictable). ~1k chars ≈ ~250 tok.
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150

_embed_lock = threading.Lock()
_embed_model = None  # lazy-loaded SentenceTransformer


def _get_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            from sentence_transformers import SentenceTransformer  # heavy, lazy
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        return _embed_model


# ---- DB ---------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    USERLINKS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS docs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        url         TEXT,
        title       TEXT,
        source_type TEXT,            -- 'url' or 'text'
        fetched_at  TEXT NOT NULL,
        n_chunks    INTEGER NOT NULL,
        raw_chars   INTEGER NOT NULL,
        full_text   TEXT
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id    INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
        idx       INTEGER NOT NULL,
        text      TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
    """)
    return con


# ---- chunking ---------------------------------------------------------------

def _split_into_chunks(text: str, size: int = CHUNK_CHARS,
                       overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Paragraph-aware char-window chunker. Keeps semantically close lines
    together when possible; falls back to hard slicing if a paragraph is huge.
    """
    text = text.strip()
    if not text:
        return []
    paras = re.split(r"\n\s*\n+", text)
    chunks, buf = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # If a single paragraph blows the budget, hard-slice it.
        if len(p) > size:
            if buf:
                chunks.append(buf.strip()); buf = ""
            for i in range(0, len(p), size - overlap):
                chunks.append(p[i : i + size])
            continue
        if len(buf) + len(p) + 2 > size:
            chunks.append(buf.strip())
            # carry overlap from end of previous chunk for continuity
            tail = buf[-overlap:] if overlap and buf else ""
            buf = (tail + "\n\n" + p) if tail else p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in (s.strip() for s in chunks) if c]


# ---- embeddings -------------------------------------------------------------

def _embed(texts: list[str]) -> np.ndarray:
    """L2-normalized embeddings, shape (N, EMBED_DIM), float32."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    m = _get_model()
    vecs = m.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False)
    return vecs.astype(np.float32, copy=False)


def _vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32, copy=False).tobytes()


def _blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


# ---- ingestion --------------------------------------------------------------

def _guess_title(text: str, fallback: str = "") -> str:
    """First non-empty short line as title — cheap and usually right."""
    for line in text.splitlines():
        line = line.strip()
        if 5 <= len(line) <= 140:
            return line
    return fallback or "(untitled)"


def ingest_url(url: str, *, overwrite: bool = False) -> dict:
    """Fetch a URL (FULL text, no cap), chunk, embed, store. Idempotent on URL
    by default (re-ingesting the same URL skips); pass overwrite=True to replace.
    """
    from surrogate.tools.fetch import fetch_url, fetch_html
    text_blob = fetch_url(url, max_chars=None)
    body = re.sub(r"^URL:[^\n]*\n+", "", text_blob, count=1).strip()
    if not body or body.startswith("(fetch error"):
        return {"status": "fetch_failed", "url": url, "detail": body[:200]}
    # Pull the real <title> so list_docs shows something useful instead of
    # whatever short line happens to appear first (e.g. SPA cookie banners).
    title_hint = ""
    try:
        from bs4 import BeautifulSoup
        html = fetch_html(url)
        if not html.startswith("(fetch error"):
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                title_hint = re.sub(r"\s+", " ", soup.title.string).strip()
    except Exception:
        pass
    return _ingest(body, source_type="url", url=url, overwrite=overwrite,
                   title_hint=title_hint)


def ingest_text(text: str, *, source_label: str = "pasted text",
                overwrite: bool = False) -> dict:
    """Ingest a chunk of raw text (e.g. a notes paste). `source_label` is
    stored in the `url` column for traceability."""
    return _ingest(text.strip(), source_type="text", url=source_label,
                   overwrite=overwrite, title_hint="")


def _ingest(body: str, *, source_type: str, url: str, overwrite: bool,
            title_hint: str = "") -> dict:
    con = _connect()
    try:
        existing = con.execute(
            "SELECT id FROM docs WHERE url = ?", (url,)
        ).fetchone()
        if existing and not overwrite:
            return {"status": "exists", "url": url, "doc_id": existing[0]}
        if existing and overwrite:
            con.execute("DELETE FROM docs WHERE id = ?", (existing[0],))

        chunks = _split_into_chunks(body)
        if not chunks:
            return {"status": "empty", "url": url}
        vecs = _embed(chunks)
        # Prefer the real <title> when ingest_url provided one; fall back to
        # a first-line heuristic for raw-text ingests.
        title = title_hint or _guess_title(body, fallback=url)
        cur = con.execute(
            "INSERT INTO docs(url, title, source_type, fetched_at, n_chunks, "
            "raw_chars, full_text) VALUES (?,?,?,?,?,?,?)",
            (url, title, source_type, datetime.now().isoformat(timespec="seconds"),
             len(chunks), len(body), body),
        )
        doc_id = cur.lastrowid
        con.executemany(
            "INSERT INTO chunks(doc_id, idx, text, embedding) VALUES (?,?,?,?)",
            [(doc_id, i, c, _vec_to_blob(v)) for i, (c, v) in enumerate(zip(chunks, vecs))],
        )
        con.commit()
        return {"status": "ok", "url": url, "doc_id": doc_id,
                "title": title, "n_chunks": len(chunks), "chars": len(body)}
    finally:
        con.close()


# ---- listing / deletion -----------------------------------------------------

def list_docs() -> list[dict]:
    con = _connect()
    try:
        rows = con.execute(
            "SELECT id, url, title, source_type, fetched_at, n_chunks, raw_chars "
            "FROM docs ORDER BY id DESC"
        ).fetchall()
        return [
            {"id": r[0], "url": r[1], "title": r[2], "source_type": r[3],
             "fetched_at": r[4], "n_chunks": r[5], "raw_chars": r[6]}
            for r in rows
        ]
    finally:
        con.close()


def delete_doc(doc_id: int) -> bool:
    con = _connect()
    try:
        cur = con.execute("DELETE FROM docs WHERE id = ?", (doc_id,))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


# ---- retrieval --------------------------------------------------------------

def retrieve(query: str, k: int = 5,
             min_score: float = 0.20) -> list[dict]:
    """Return top-k chunks with cosine score >= min_score, ordered by score.

    Brute-force cosine over all chunks (embeddings are L2-normalized so dot
    product = cosine). Fine up to ~10k chunks.
    """
    con = _connect()
    try:
        rows = con.execute(
            "SELECT c.id, c.doc_id, c.idx, c.text, c.embedding, d.url, d.title "
            "FROM chunks c JOIN docs d ON d.id = c.doc_id"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return []
    mat = np.stack([_blob_to_vec(r[4]) for r in rows], axis=0)
    qv = _embed([query])[0]
    scores = mat @ qv  # cosine since both unit-normalized
    order = np.argsort(-scores)[:k]
    out = []
    for j in order:
        s = float(scores[j])
        if s < min_score:
            continue
        r = rows[int(j)]
        out.append({
            "chunk_id": r[0], "doc_id": r[1], "chunk_idx": r[2],
            "text": r[3], "url": r[5], "title": r[6], "score": s,
        })
    return out


# ---- evidence-pack formatter (consumed by two_stage) ------------------------

def build_rag_evidence_block(query: str, k: int = 5) -> tuple[str, list[dict]]:
    """Return (text_block, hits). The block is in the same shape as the
    Stage 2 tool-evidence sources so it slots in cleanly.

    Empty hits -> ("", []) so the caller can skip appending anything.
    """
    hits = retrieve(query, k=k)
    if not hits:
        return "", []
    lines = ["", "ADDITIONAL EVIDENCE FROM USER-PROVIDED SOURCES (RAG):"]
    for i, h in enumerate(hits, 1):
        url = h.get("url") or "(no url)"
        title = h.get("title") or ""
        lines.append("")
        lines.append(f"---- User-source {i} (score={h['score']:.2f}): {title} — {url} ----")
        lines.append(h["text"])
    return "\n".join(lines), hits
