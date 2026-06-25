"""Compare Claude and OpenAI ranked lists directly via soft_match_topN."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from surrogate.head_to_head import soft_match_topN

qs = [l.strip() for l in (ROOT/'data/h2h-10q.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
entries = [json.loads(l) for l in (ROOT/'backtests/h2h-store.jsonl').read_text().splitlines()]

def kind(e):
    m = e['frontier'].get('model','') or ''
    if 'gpt' in m: return 'openai'
    if 'claude' in m: return 'claude'
    return 'other'

def latest(k, q):
    matches = [e for e in entries if e.get('question','').strip()==q.strip() and kind(e)==k]
    return max(matches, key=lambda e: e.get('ts','')) if matches else None

print('Computing Claude ↔ OpenAI soft matches…', flush=True)
rows = []
for i, q in enumerate(qs, 1):
    c = latest('claude', q); o = latest('openai', q)
    if not c or not o:
        rows.append({'i':i,'q':q,'overlap':None}); continue
    a = c['match']['b']  # Claude's ranked
    b = o['match']['b']  # OpenAI's ranked
    k = c.get('k', 10)
    m = soft_match_topN(a, b, k=k)
    rows.append({'i':i,'q':q,'overlap':m['overlap'],'a':a,'b':b,'pairs':m['matched_pairs'],'k':k})
    print(f'  [{i}/10] {q[:55]!r}... -> overlap={m["overlap"]}/{len(a)}', flush=True)

(Path('/tmp/cla_vs_oai.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print('Saved /tmp/cla_vs_oai.json')
