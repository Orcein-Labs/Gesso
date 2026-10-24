"""Time one streamed chat completion per prompt length against any
OpenAI-compatible server, so gesso-serve and llama-server are measured by the
same client over the same HTTP path.

    python -I benchmark/stream_client.py --url http://127.0.0.1:8080 \
        --label gesso-cuda --text benchmark/corpus-pride-and-prejudice.txt --chars 2000,32000 \
        --max-tokens 64 --out results.tsv

A prompt of N characters is the first N characters of --text, prefixed with a
run id so neither server can serve it from a prompt cache. Each prompt runs
--reps times and the last run is kept. TTFT is request sent to first content
byte; inter-token gaps are between streamed chunks that carry content. Both
servers stream one token per chunk, which is checked: a run whose chunk count
differs from completion_tokens by more than one is flagged.

Standard library only.
"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid


def stream_once(url, prompt, max_tokens):
    body = json.dumps({
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        # llama-server: no prompt cache; gesso-serve ignores unknown fields
        "cache_prompt": False,
        # the timing must not stop at an early end-of-turn
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    stamps, text, usage = [], [], None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    stamps.append(time.perf_counter() - t0)
                    text.append(piece)
    return stamps, "".join(text), usage


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--chars", default="2000,32000,64000")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    corpus = open(a.text, encoding="utf-8").read()
    rows = []
    for n in (int(x) for x in a.chars.split(",")):
        while len(corpus) < n:
            corpus += "\n\n" + corpus
        for _ in range(a.reps):
            prompt = f"[run {uuid.uuid4().hex}]\n" + corpus[:n] + "\n\nSummarize the above."
            stamps, text, usage = stream_once(a.url, prompt, a.max_tokens)
        gaps = [(b - c) * 1e3 for c, b in zip(stamps, stamps[1:])]
        ptok = usage.get("prompt_tokens") if usage else None
        ctok = usage.get("completion_tokens") if usage else None
        row = {
            "label": a.label, "chars": n, "prompt_tokens": ptok, "completion_tokens": ctok,
            "chunks": len(stamps), "ttft_s": round(stamps[0], 3),
            "decode_tok_s": round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2),
            "itl_p50_ms": round(statistics.median(gaps), 2), "itl_p90_ms": round(pct(gaps, 0.9), 2),
            "itl_p99_ms": round(pct(gaps, 0.99), 2), "itl_max_ms": round(max(gaps), 2),
            "itl_stdev_ms": round(statistics.stdev(gaps), 2),
            "chunk_check": "ok" if ctok is None or abs(ctok - len(stamps)) <= 1 else "MISMATCH",
        }
        rows.append(row)
        print("RECEIPT stream_client " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write("\t".join(f"{k}={v}" for k, v in r.items()) + "\n")


if __name__ == "__main__":
    main()
