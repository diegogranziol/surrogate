"""CLI: ingest URLs (or a local text file) into the user-RAG store.

  python ingest_links.py https://example.com/page1 https://example.com/page2
  python ingest_links.py -f links.txt              # one URL per line
  python ingest_links.py --text "free-form notes"  # raw text paste
  python ingest_links.py --list                    # show docs in the store
  python ingest_links.py --delete 7                # delete doc id 7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from surrogate.rag import (
    DB_PATH, delete_doc, ingest_text, ingest_url, list_docs,
)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urls", nargs="*", help="One or more URLs to ingest.")
    p.add_argument("-f", "--file", help="A file with one URL per line.")
    p.add_argument("--text", help="Ingest a raw text string (no fetch).")
    p.add_argument("--label", default="pasted text", help="Source label for --text.")
    p.add_argument("--list", action="store_true", help="Show stored docs and exit.")
    p.add_argument("--delete", type=int, help="Delete a doc by id and exit.")
    p.add_argument("--overwrite", action="store_true", help="Re-ingest if URL already exists.")
    args = p.parse_args(argv[1:])

    if args.list:
        rows = list_docs()
        if not rows:
            print(f"(store empty: {DB_PATH})")
            return 0
        print(f"{len(rows)} document(s) in {DB_PATH}:")
        for r in rows:
            print(f"  [{r['id']:>3}] {r['title']!r}  ({r['n_chunks']} chunks, "
                  f"{r['raw_chars']} chars, {r['source_type']})  -> {r['url']}")
        return 0

    if args.delete is not None:
        ok = delete_doc(args.delete)
        print(f"deleted doc {args.delete}: {ok}")
        return 0 if ok else 1

    if args.text:
        r = ingest_text(args.text, source_label=args.label, overwrite=args.overwrite)
        print(r)
        return 0

    urls: list[str] = list(args.urls)
    if args.file:
        urls += [l.strip() for l in Path(args.file).read_text().splitlines() if l.strip()]
    if not urls:
        p.print_help()
        return 2

    for u in urls:
        r = ingest_url(u, overwrite=args.overwrite)
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
