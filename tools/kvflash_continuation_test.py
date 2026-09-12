#!/usr/bin/env python3
"""Regression test for the KVFlash live-pool continuation fix (Bug 7).

Verifies that once a session's context exceeds the KVFlash resident pool,
subsequent turns reuse the live pool (fast, ~0.2s) instead of paying a full
pooled/evicting reprefill (~24-34s) every round-trip, AND that the answers
produced are correct (not just fast).

Usage: python3 kvflash_continuation_test.py [base_url]
Exits non-zero if any answer is wrong or the speedup doesn't materialize.
"""
import json
import sys
import time
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18081/v1/chat/completions"


def chat(messages, max_tokens=8):
    body = json.dumps({
        "model": "dflash", "messages": messages,
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(BASE_URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"], time.time() - t0


def main():
    filler = "The quick brown fox jumps over the lazy dog. " * 400
    messages = [{"role": "system", "content": "You are a terse assistant. Reply with ONLY the final number, no words."}]
    big_user = "Reference text, just say OK.\n" + filler * 6
    messages.append({"role": "user", "content": big_user})

    reply, dt = chat(messages, max_tokens=4)
    messages.append({"role": "assistant", "content": reply})
    print(f"SEED (forces past the pool boundary) dt={dt:.1f}s reply={reply!r}")
    if dt < 10:
        print("WARNING: seed prefill was suspiciously fast; pool boundary may "
              "not have been genuinely crossed -- results below are not a "
              "meaningful test of the continuation path.")

    pairs = [(3, 4), (10, 11), (100, 1), (7, 8), (20, 22), (5, 5), (9, 10), (50, 50)]
    ok = True
    fast_count = 0
    for i, (a, b) in enumerate(pairs):
        messages.append({"role": "user", "content": f"What is {a}+{b}? Reply with just the number."})
        reply, dt = chat(messages, max_tokens=6)
        messages.append({"role": "assistant", "content": reply})
        expect = str(a + b)
        got_ok = expect in reply
        ok = ok and got_ok
        if dt < 5:
            fast_count += 1
        print(f"T{i+1} dt={dt:.2f}s {a}+{b}={expect} reply={reply!r} "
              f"{'OK' if got_ok else 'MISMATCH'}")

    print(f"\ncorrectness: {'ALL_OK' if ok else 'SOME_MISMATCH'}")
    print(f"continuation engaged (fast, <5s) on {fast_count}/{len(pairs)} turns")
    if not ok or fast_count < len(pairs) - 1:
        sys.exit(1)


if __name__ == "__main__":
    main()
