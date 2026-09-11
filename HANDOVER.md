# Local LLM Deployment — Qwen3.8-27B-Uncensored via Lucebox

> **交接文档 / HANDOVER DOCUMENT** — this records the final, locked production
> setup for this GPU instance as of this session. Read this before touching
> the `llama-server` supervisor service, the model files under
> `/workspace/models/`, or the Codex integration.

Locked production configuration for this GPU instance. **Context length and KV
precision below are fixed invariants — never reduce them to gain speed.**

## Hardware

- GPU: NVIDIA RTX PRO 4000 Blackwell, 24GB GDDR7, 8960 CUDA cores, 672 GB/s bandwidth
- System RAM: 98GB (persistence mode ON, ECC OFF, power limit maxed at 145W — no headroom there)

## Model

- Target: `Qwen3.8-27B-Uncensored` (Q4_K_M GGUF) — `/workspace/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf`
- Architecture: `qwen35` hybrid (64 layers: 16 full-attention + 48 linear-attention/SSM), native 262144 max context
- Draft (DFlash2 speculative decode): `/workspace/models/qwen38-dflash2-q8_0.gguf` (converted from `incoai/Qwen3.8-27B-DFlash2` via Lucebox's own `convert_dflash_to_gguf.py` + `quantize_dflash_draft.py` — the llama.cpp-format z-lab GGUF is not tag-compatible with this loader)
- KVFlash relevance drafter: `/workspace/models/drafter/Qwen3-0.6B-Q8_0.gguf`

## Engine

[Lucebox](https://github.com/Luce-Org/lucebox) `dflash_server` — hand-tuned CUDA inference engine (fork of ggml/llama.cpp) with custom DFlash2 block-diffusion speculative decoding + DDTree verification + KVFlash bounded-residency KV cache, purpose-built for this `qwen35` model family. Built from source at `/workspace/lucebox/server/build/dflash_server` (`cmake -DCMAKE_CUDA_ARCHITECTURES=120`).

Why Lucebox over llama.cpp: llama.cpp's own DFlash2 support (recently merged upstream) measured ~4 tok/s on this card — an order of magnitude slower than Lucebox's hand-written CUDA kernels for the same drafter/target pair.

## Final locked configuration

Supervisor service: `llama-server` (managed by supervisord, config at `/etc/supervisor/conf.d/llama-server.conf`, wrapper script at `/opt/supervisor-scripts/llama-server.sh`).

```
./build/dflash_server /workspace/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf \
  --draft /workspace/models/qwen38-dflash2-q8_0.gguf \
  --target-device cuda:0 --draft-device cuda:0 \
  --draft-block-size 8 \
  --max-ctx 262144 \
  --think-max-tokens 14336 \
  --kvflash auto \
  --prefill-drafter /workspace/models/drafter/Qwen3-0.6B-Q8_0.gguf \
  --ddtree --ddtree-budget 22 \
  --cache-type-k f16 --cache-type-v f16 \
  --prefix-cache-slots 32 \
  --prefill-cache-slots 16 \
  --host 127.0.0.1 --port 18081
```

**`--agent-turn-cache` is deliberately *not* used — see the "multi-turn crash" postmortem below before re-adding it.**

The wrapper script also sets a `trap ... EXIT TERM INT` on the backgrounded `dflash_server` PID. Bash does not forward signals to a backgrounded child by default, so without this a `supervisorctl restart` would leave the old process running as an orphan holding VRAM, and the *next* start would OOM against that ghost instance (this actually happened once while wiring this up — fixed, don't remove the trap).

### Reasoning effort / thinking budget

Codex never sends an explicit `reasoning.effort` tier (it doesn't recognize the custom `dflash` model, so it has no basis to pick one) — every real request logs `reasoning_effort=default`, which without an override falls back to `think_max_tokens`. This model's family-fallback tier table (no dedicated model-card sidecar exists for it) is:

| Tier | Tokens |
|---|---|
| low | 3584 |
| **medium** | **14336** |
| high (= old default) | 28672 |
| x-high | 28672 |
| max | 28672 |

Left at its own default, the server was silently running every request at the **`high`** budget (28672 — `think_max_tokens` with no CLI override equals `default_max_tokens(32768) − hard_limit_reply_budget(4096) = 28672`, which happens to equal `high` for this model since there's no `complex_problem_max_tokens` to separate `high`/`x-high`/`max`). Locked to **`medium` (14336)** via `--think-max-tokens 14336` on the reasoning that Codex can't ask for a tier itself, so the server-side default *is* the operating point — pick deliberately, don't leave it at whatever the fallback formula happens to produce. If more or less thinking is wanted later, change this one flag (or explore whether Codex's own `/model` reasoning-effort picker can be made to pass `reasoning.effort` through to a custom provider — untested, the CLI warns "Model metadata for `dflash` not found" and effort tiers are one of the things that metadata would normally drive).

The wrapper script self-checks on every start that `/v1/models` reports
`context_length: 262144` and kills + hard-fails (non-zero exit) if it doesn't
— this deployment must never silently run at a reduced context.

### Why each flag, and what was tried and rejected

| Flag | Value | Notes |
|---|---|---|
| `--max-ctx` | **262144** | The model's true architectural max (`max_position_embeddings` / GGUF `context_length`). **Hard-locked, never reduce.** |
| `--kvflash` | `auto` (resolves to 16384-token resident pool) | Bounded KV residency — cold context pages to host RAM instead of needing full 262144-tokens-worth of VRAM. This is what makes 262144 possible on 24GB at all. Explicit `8192` was faster but smaller than a real Codex prompt (quality/correctness risk — rejected); `24576` OOM'd. |
| `--prefill-drafter` | `Qwen3-0.6B-Q8_0.gguf` | Gives KVFlash proper relevance-scored residency (`policy=drafter`) instead of recency-only LRU — much better long-context recall at the same pool size. |
| `--ddtree --ddtree-budget` | `22` | DDTree speculative verification. 22-24 measured flat; 28+ OOMs at this context. |
| `--draft-block-size` | `8` | The engine auto-caps verify width to 8 once context passes ~8192 tokens regardless of what's requested, so higher values are moot for any realistic (Codex-sized) prompt. |
| `--cache-type-k/v` | **f16** | Only lever ever moved *up* in precision from a prior q8_0 baseline (not down) — measured faster (~62 vs ~59 tok/s) since it skips a per-step dequant. `bf16` measured slower (55.6). **q4_0 (the engine's own non-laguna default) was never used — quality floor is q8_0-or-better, always.** |
| `--prefix-cache-slots` / `--prefill-cache-slots` / `--agent-turn-cache` | `32` / `16` / on | Reuses system RAM (98GB, mostly idle) to cache the KV state of repeated prompt prefixes. Codex resends a near-identical ~8-12K-token system-prompt + tool-schema block on every call — a cache hit takes prefill from ~11.6s to ~0ms. This is the one place "using more RAM" genuinely buys speed; RAM does not accelerate raw decode compute. |

Also tried and found to make no measurable difference at this config: `--specla` (+ `--specla-top-k`), `--kvflash-tau`, `--draft-residency persistent`, `--chunk` size.

## Measured performance

- Decode: **~59-62 tok/s** at full 262144 context (short prompt, cold cache)
- Prefill: ~11.6s for a realistic ~11.7K-token Codex-shaped prompt (cold), **~0ms on a prefix-cache hit**
- VRAM: ~19GB idle → ~23GB under load (24.5GB card, small safety margin — do not raise `--ddtree-budget` past 24 or add KV precision beyond f16)
- RAM: ~9GB used / 98GB total — ample headroom for running actual code/programs alongside the model

100 tok/s (and the originally-hoped-for 200 tok/s) was targeted but not reached without violating the context/quality floor; the measured bottleneck at 262144 context is the per-step target-model forward pass (DDTree verification, 2 target forwards/step) plus draft overhead — bounded by this card's compute/bandwidth (672 GB/s, well below even an RTX 3090), not by any of the config levers tried.

### Root cause, confirmed by direct measurement + community research

A [vLLM forum thread specifically about this exact GPU](https://discuss.vllm.ai/t/sm120-rtx-pro-4000-6-5x-throughput-gain-and-v0-18-1-regression-findings/2525) (RTX PRO 4000, SM120) independently found the same thing: **bandwidth is the SM120 bottleneck**, and its 140W power limit throttles clocks from a ~2300MHz boost down to ~1935MHz under sustained load. We reproduced this directly on our card:

- Rated max SM clock: **3090 MHz**
- Measured SM clock under actual sustained decode load: **~1935-2280 MHz** (mostly sitting at ~1950-2000MHz)
- Power draw pinned at **~145W** (the card's hard max — `nvidia-smi -q` shows min/max/default/current power limit all reading 145W, no headroom to raise it)
- Attempted `nvidia-smi --lock-gpu-clocks=3090,3090` to force max clock: **refused** — "current user does not have permission to change clocks for GPU" (Vast.ai's container is unprivileged; hardware clock/power control is not exposed to the tenant)

So the card runs real inference at roughly **65% of its rated boost clock**, purely because its power budget can't sustain the full clock, and this platform gives no way to override that. This is a genuine hardware+platform ceiling, not a missed config option — every software lever available (KV dtype, DDTree budget, KVFlash pool, `--specla`, chunk size, draft residency, GPU clock locking) has been tried. Reaching 100-200 tok/s on this model would require a GPU in the RTX 3090/5090/R9700 power-and-bandwidth class (all the cards in Lucebox's own published benchmarks), not this one.

## Codex CLI integration

- Installed: `@openai/codex` v0.154.0 (via `npm install -g` under nvm)
- Config: `/root/.codex/config.toml`
  ```toml
  model_provider = "lucebox"
  model = "dflash"

  [model_providers.lucebox]
  name = "Lucebox Local (Qwen3.8-27B-Uncensored + DFlash2)"
  base_url = "http://127.0.0.1:18081/v1"
  env_key = "LUCEBOX_API_KEY"
  wire_api = "responses"
  ```
- `LUCEBOX_API_KEY=local-no-auth` exported in `~/.bashrc` (dummy value — the local server has no auth)
- Verified end-to-end: `codex exec "..." < /dev/null` returns real model output.
- `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` **are** set as top-level defaults in `config.toml` (confirmed active via the `codex exec` banner: `approval: never`, `sandbox: danger-full-access`). First attempt at this exact combination was blocked once by the auto-mode safety classifier ("Create Unsafe Agents"); a later explicit request went through. **Consequence, spelled out**: every shell command / file edit the model decides to run executes immediately with no confirmation prompt and no sandbox containment. This was an explicit user choice, not a default worth re-adding casually if this box is ever repurposed.
- No `codex login` needed and none was done — pointing entirely at the custom `lucebox` provider means Codex never talks to OpenAI's auth or API at all (`codex doctor` shows "no Codex credentials found" as a harmless warning, not a blocker).
- Codex auto-added a `[projects."/workspace/lucebox"] trust_level = "trusted"` block to `config.toml` on first run from that directory (its own trust-on-first-use behavior, not something we set).

## Postmortem: multi-turn conversations hard-failing after ~15-17K accumulated tokens

**Symptom** (discovered live, via real Codex usage, not synthetic testing): a real multi-turn Codex session would work fine for the first several turns, then every subsequent turn would return an empty response — from the user's side, Codex looked "stuck" (no visible error, no output, `codex exec` and interactive mode both just went silent after a tool-calling exchange).

**Root cause, found in `/var/log/portal/llama-server.log`**: once the conversation's restored cache prefix plus the new incremental prompt exceeded the KVFlash resident pool (16384 tokens with `--kvflash auto`), the server logged
```
[kvflash] restored prefix (14848) + prompt (2338) exceeds pool 16384; pooled prefill requires a fresh request
[server] chat DONE ... ok=false ... out=0 ... error=prefill_failed
```
and returned a `status: "completed"` response with empty `output_text` — i.e. **the failure is invisible at the API/JSON level**; you have to check the server's own log line (`ok=false`, `error=prefill_failed`) to see it happened at all. This specific failure mode is triggered by `--agent-turn-cache` ("extend prefix caching through generated tool calls") trying to *restore* a cached snapshot into the bounded KVFlash pool — restoring requires the whole restored-prefix-plus-new-tokens to fit in the pool in one shot; a plain fresh (non-restored) request does not have this requirement and pages normally.

**First (wrong) fix attempted**: just made the KVFlash pool bigger (`--kvflash 65536` instead of `auto`/16384) to push the threshold out further. This loaded fine and passed a quick synthetic check, but **broke everything much worse**: baseline VRAM usage at pool=65536 was ~22.1GB, leaving only ~2.3GB free — not enough for the ~3.5GB "rollback cache" buffer (`--fast-rollback`, on by default) that gets allocated per-request. Every single request then failed with `ggml_backend_alloc_ctx_tensors failed for rollback cache` — again silently returning `status: "completed"` with empty output, so this looked identical to "working" unless you checked `ok=` in the log. **Lesson: always grep the server log for `ok=false` after any change, never trust the JSON body's `status` field alone — this engine can return `status: "completed"` on a real server-side failure.** `--kvflash 32768` was also tried and twice crashed the whole process outright (`CUDA error: the resource allocation failed` in `cublasCreate_v2`) — not reproduced further, treated as further evidence this pool size range is a bad place to operate, not investigated to a root cause.

**Actual fix**: removed `--agent-turn-cache` entirely, kept `--kvflash auto` (16384) and `--prefix-cache-slots`/`--prefill-cache-slots` as before (those still correctly cache and reuse an *exact repeated* prefix — e.g. Codex's unchanging system prompt — which is a different, unaffected code path). Verified with a script (`/workspace/sim_multiturn.sh`, still present) that resends a real growing conversation (same shared prefix + appended content each turn, exactly like Codex) for 22 turns straight through 17,781 accumulated tokens — **all 22 genuinely `ok=true`** (confirmed in the log, not just JSON status) — comfortably past the ~15-17K point that used to break. Also re-verified against a real `codex exec` call afterward.

**Known trade-off of this fix**: without `--agent-turn-cache`, a long tool-calling conversation no longer gets an incremental-restore speedup for the *growing* part of its history — each turn's full current prompt is freshly prefilled (KVFlash still pages/evicts normally, it just isn't restoring from a snapshot). Prefill time grows with conversation length as a result (observed ~24s prefill at ~17K tokens in the stress test) — turns get slower as a conversation grows long, but correctly complete rather than silently failing. The exact-repeated-prefix cache (Codex's system prompt / tool schemas, unchanged turn to turn) is untouched and still gets its fast-path. If someone wants to revisit `--agent-turn-cache` for the speed back, the pool-vs-rollback-cache VRAM interaction above needs solving first, not just cranking the pool size.

## Full testing log / lessons learned

Everything actually measured this session, in chronological order, so nobody has to re-discover any of this. All speed numbers are `decode_tokens_per_sec` from the real server, not estimates.

### Phase 0 — llama.cpp + MTP (superseded, kept for context)

Before finding Lucebox, the model was served via a from-source llama.cpp build (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`) using its built-in MTP (multi-token-prediction) self-speculative decoding, `--spec-type draft-mtp`.

- Architecture insight that made large context viable at all: this model (`qwen35`) is a **hybrid** — 64 layers, only 16 full-attention, 48 linear-attention/SSM (gated-delta-net). KV cache cost scales with the 16 full-attention layers only, not all 64, which is why huge context is cheap on this model family specifically.
- Context ceiling with plain llama.cpp + MTP, `-ngl 999` (forced full GPU), no speculative decoding: 131072 ✅ (21034 MiB) · 155648 ✅ (21970 MiB) · 163840 ✅ (22282 MiB, picked) · 167936 ❌ OOM. **Important gotcha**: forcing `-ngl 999` (llama.cpp) makes its internal `--fit` auto-tuner refuse outright ("n_gpu_layers already set by user... abort") whenever it can't also fit an *unrelated* thing it's trying to auto-tune (e.g. speculative-decode overhead) — this reads exactly like a real OOM but isn't one; the fix was leaving `-ngl` on auto, at the cost of silent CPU-layer-offload if the auto-fit under-provisions (see next point).
- **Trap**: at very large `--ctx-size` (e.g. 262144) with `-ngl` left on `auto`, llama.cpp's auto-fit will *silently* offload layers to CPU RAM to make the config "fit," tanking speed to **single digits tok/s** with no error at all — confirmed by watching the process pin all 39 CPU cores while GPU sat mostly idle. This is the single most important trap from the whole session: a working load ≠ a fast load; always benchmark actual decode tok/s, never just "did it start."
- MTP `--spec-draft-n-max` sweep (this is genuinely **prompt-dependent**, not a fixed constant — re-verify per workload rather than trusting one benchmark):
  - Short/simple prompt ("write a fibonacci function"): n_max=3 (default) → **56.9-58 tok/s** (best) · n_max=2 → 53.7 · n_max=1 → 45.3
  - Long/complex prompt (detailed red-black-tree explanation): n_max=2 → **47.0-48.2 tok/s** (best here) · n_max=3 → 44.3-44.4 · n_max=1 → 45.3 · n_max=4/5 → fit-check abort (see gotcha above)
  - Net lesson: don't tune a speculative-decode parameter against a single synthetic prompt and declare victory — it visibly flips depending on how predictable the generated content is.
- Tried `--spec-type draft-dflash` (llama.cpp's own, very recently merged upstream DFlash2 support) two ways: with no matching draft weights → no speedup (31.1 tok/s, same as no speculative decoding); with the z-lab DFlash2 GGUF as `--model-draft` → **4.1-4.13 tok/s, ~7.5× *slower*** than baseline (confirmed via server log: draft_n=33, accepted=13). llama.cpp's DFlash2 integration is evidently not yet performant on this build/GPU — this is what motivated finding Lucebox.
- Tried classic two-model speculative decoding (`--spec-type draft-simple` + a small standalone draft gguf): **segfaulted** on load. Abandoned, not investigated further (Lucebox's DFlash2 path made this moot).

### Phase 1 — Lucebox, moderate context (49152-53248), finding the speed knobs

Built Lucebox from source (`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120` — auto-detects the "consumer Blackwell" workaround for this card, no extra flags needed). First attempt to load the z-lab llama.cpp-format DFlash2 GGUF directly in Lucebox **failed to load** ("unexpected draft arch: dflash (expected qwen35-dflash-draft, dflash-draft, or gemma4-dflash-draft)") — Lucebox needs its *own* GGUF tagging, produced by its own conversion scripts, not the generic llama.cpp-ecosystem one. Fix: downloaded the source safetensors (`incoai/Qwen3.8-27B-DFlash2`, 3.85GB bf16) and ran Lucebox's own `scripts/convert_dflash_to_gguf.py` → `scripts/quantize_dflash_draft.py --scheme q8_0`, producing `/workspace/models/qwen38-dflash2-q8_0.gguf`, which loaded and worked ever after.

- `--draft-block-size` sweep at ctx=49152, `--ddtree` off (values above 16 rejected outright: "must be in [2, 16] for this drafter"):
  - 8 → 79.6 tok/s (accept_rate 60.4%) · 10 → 85.3 (55.1%) · **12 → 88.0-88.2 (44.8-47.6%, best)** · 13 → 88.1-88.2 (44.8%) · 14 → 84.9 (40.2%) · 16 → 83.7 (35.3%)
- `--ddtree` (on top of block-size 12/13) sweep: no ddtree → 88.1 · budget=8 → 81.8 (worse — too narrow a tree) · **budget=20 → 91.4, budget=22 → 90.0-91.9 (best zone)** · budget=24 → OOM at this ctx · budget=40 → real CUDA OOM (`cudaMalloc failed`)
- **Load-bearing lesson (cost real time this session)**: none of the above numbers survive contact with a *realistic* prompt. Codex sends ~8-12K tokens of system-prompt + tool-schema on every single call. The config above (block=12, ddtree budget=22, ctx=53248) OOM'd the instant a real ~11.7K-token prompt was sent — short synthetic benchmarks (a one-line "write fibonacci" prompt) completely hid this. **Always validate any speed config against a realistic-size prompt, not just a toy one, before trusting a benchmark number.** Fix at the time: dropped ddtree budget to 16 at fixed ctx (later superseded by the KVFlash approach below).
- Also root-caused a `--max-ctx` vs `-ngl`-equivalent illusion here too: Lucebox has its own `--fit`-like auto-sizing, and at the time it looked like real ceilings were being hit around ctx≈106-112K in "lean" mode (block=8, no ddtree, q8_0 KV) — this number is superseded by Phase 2 below and should **not** be treated as the real ceiling; it was the ceiling *without* KVFlash.

### Phase 2 — reaching the model's true max context (262144) via KVFlash

Found via targeted GitHub/web search after repeated direct requests from the user not to settle for less than the model's real architectural max. **KVFlash** (`optimizations/kvflash/README.md` in the Lucebox repo) is a bounded-KV-residency scheme: cold context chunks page to host RAM (bit-exact, recallable) instead of needing full-context VRAM; a small relevance-scoring drafter (default probe: `Qwen3-0.6B-BF16.gguf`, we used the Q8_0 quant since that's what's published) decides what stays resident. Published number that motivated trying it: "Qwen3.6-27B Q4_K_M on RTX 3090: native 256K context at 38.6 tok/s with 72 MiB resident KV."

- `--kvflash auto` at `--max-ctx 262144`: **loaded successfully at only 18574 MiB**, survived the realistic ~11.7K-token prompt, first real speed measurement **48.6 tok/s** (vs total failure before this). Auto-sizing picked a 16384-token resident pool (a "speed cap," not a VRAM cap — the doc is explicit that a bigger pool trades speed for recall, not the other way around).
- Adding `--ddtree --ddtree-budget 22` back on top of KVFlash at ctx=262144: **58.7-59.2 tok/s**, ~22.5GB VRAM, survived the realistic prompt. This is what got deployed as "done" for that phase of the session.
- `--prefill-drafter Qwen3-0.6B-Q8_0.gguf` (downloaded from `Qwen/Qwen3-0.6B-GGUF`) upgrades KVFlash from LRU-fallback (`policy=lru`, recency-only — logged warning: no drafter found) to proper relevance-scored residency (`policy=drafter`) — same speed (59.2 tok/s), meaningfully better long-context recall per the KVFlash doc's own needle-in-haystack numbers (88-100% recall at 6% residency, vs LRU which can evict the actual question if the pool is undersized). Kept, no reason not to.
- Explicit `--kvflash <N>` instead of `auto`: `8192` → 62.1 tok/s but **smaller than the realistic 11.7K-token prompt** (logged: "snapshot skipped: cur_pos 11720 exceeds pool 8192... pooled snapshots are a follow-up") — faster but a correctness/quality risk for real usage, rejected. `24576` → real decode-time OOM ("draft-kv bulk append failed"). **`auto` (16384) is the right answer, not a compromise.**
- Discovered a real self-check bug while wiring this up: the supervisor wrapper originally used the shared `pty` helper (`unbuffer -p ...`) on a command that was then backgrounded with `&` for a post-start self-check — `unbuffer` does not tolerate this combination and the whole process died near-instantly with a completely misleading (silent) failure. **Fix: don't wrap a backgrounded server process in `pty`/`unbuffer`; it flushes its own logs line-buffered fine without it.** This is now baked into the wrapper script.

### Phase 3 — squeezing decode speed further at fixed 262144 context (systematic pass)

Every experiment below held `--max-ctx 262144` fixed (never relaxed) and was validated against both a short-prompt speed test and the realistic ~11.7K-token prompt for OOM safety, per an explicit user rule that context size and KV-precision-or-better are non-negotiable invariants, never speed trade-offs (see `[[gpu-ctx-speed-no-compromise]]` in assistant memory).

| Lever tried | Values | Result |
|---|---|---|
| KV cache dtype | q8_0 (baseline) → **59.8 tok/s** | baseline |
| | **f16 → 61.5-62.1 tok/s (winner, adopted)** | faster, *higher* precision than q8_0, not lower |
| | bf16 → 55.6 tok/s | slower than f16, rejected |
| `--specla` (+ `--specla-top-k` 2/4/8) | 60.9-61.4 tok/s | no real change vs f16-only baseline (61.5); not adopted |
| `--ddtree-budget` re-sweep (on full KVFlash+f16 stack) | 22 → 61.5 · 24 → 61.2-61.3 · **26 → 57.6 (worse)** · 28 → real OOM | kept 22 |
| `--kvflash <N>` explicit | 8192 → 62.1 but unsafe for real prompt size (see Phase 2) · 24576 → OOM | kept `auto`=16384 |
| `--kvflash-tau` | 32 → 61.8 · 64 (default) → 61.5-62 · 128 → 61.9 | no real effect |
| `--draft-residency persistent` (vs `auto`) | 62.0 | no measurable difference from `auto` |
| `--chunk` | 512 (default) → 61.5-61.6 · 1024 → 61.9 · 2048 → 61.6 | no real effect |
| GPU clock lock (`nvidia-smi --lock-gpu-clocks=3090,3090`) | — | **refused**: "current user does not have permission to change clocks for GPU" (Vast.ai container has no hardware clock/power control exposed to the tenant) |
| GPU power limit raise | — | already at hard max (145W = min = max = default, per `nvidia-smi -q`) |
| ECC / persistence mode | — | already optimal (ECC off, persistence mode on) — nothing to gain |

**Net result of Phase 3**: only KV dtype (q8_0→f16) was a real, adoptable win (~+3-5%). Everything else was flat or actively negative. Measured under real sustained decode load, the GPU's SM clock sits at **~1935-2280 MHz vs a rated 3090 MHz max** — throttled by the fixed 145W power ceiling, confirmed independently by a [vLLM community thread about this exact GPU](https://discuss.vllm.ai/t/sm120-rtx-pro-4000-6-5x-throughput-gain-and-v0-18-1-regression-findings/2525) reporting the identical phenomenon (their card throttled 2300→1935MHz at 140W). **This is a hard hardware+platform ceiling, not an unexplored config space.** Final locked number: **~59-62 tok/s** at full 262144 context, f16 KV, no quality/context compromise anywhere.

## Operating notes

- Restart: `supervisorctl restart llama-server`; check: `supervisorctl status llama-server`
- Logs: `/var/log/portal/llama-server.log`
- Model/context sanity check: `curl -s http://127.0.0.1:18081/v1/models`
- The service is bound to `127.0.0.1` only (no public port assigned to this instance for it) — reach it via SSH local port-forward from another machine, or call it directly from processes running on this box (e.g. Codex).
- **Do not** lower `--max-ctx` or drop `--cache-type-k/v` below `f16`/`q8_0`-equivalent to chase speed — see `[[gpu-ctx-speed-no-compromise]]` in the assistant's memory for the standing rule and rationale.

## Phase 4 — OpenCode integration: streaming fixes and the unresolved TUI blank-screen bug

Goal: use this local model as OpenCode's backend, with `opencode run` and (ideally) the full interactive TUI both working. OpenCode (`Luce-Org`'s fork lives at `/workspace/opencode`, run from source via `bun run .../src/index.ts` — a wrapper at `/opt/nvm/versions/node/v24.20.0/bin/opencode` does this so the global `opencode` command always runs patched source, no rebuild needed to iterate) proxies through an internal HTTP server and talks to our Lucebox server via `@ai-sdk/openai-compatible` (`/v1/chat/completions`), configured in `/workspace/opencode.json`.

**Bug A (server-side): SSE responses weren't framed for incremental delivery.** `http_server.cpp`'s `send_sse_headers()` used `Connection: keep-alive` without `Transfer-Encoding: chunked`/`Content-Length` — spec-ambiguous HTTP/1.1 framing that some clients don't stream correctly even though raw `curl` tolerates it. **Fix:** changed to `Connection: close`. Verified via raw Node `fetch()`: 168 genuinely time-distributed chunks over 8.3s.

**Bug B (client-side, in OpenCode itself, not our server): `opencode run` printed nothing for 30-44s then dumped the full response at once, even after Bug A's fix.** Root cause: `packages/opencode/src/cli/cmd/run.ts` only listened for the `message.part.updated` event (fires twice per part: empty at start, full text at end — not incremental). The real per-token data flows through a *separate* event type, `message.part.delta` (wire shape `{sessionID, messageID, partID, field, delta}`, defined in `packages/schema/src/v1/session.ts`, published by `session.ts`'s `updatePartDelta`), which `run.ts` never subscribed to. **Fix:** patched `run.ts` to also handle `message.part.delta` — tracks `partKind` (text/reasoning) per part-start, writes `d.delta` directly to stdout as it arrives, and skips re-printing the full text on the part's terminal `message.part.updated` event (only emits the closing style-reset/newline then). Verified: fresh prompts now show ~10 chunks spread over ~4.5 real seconds instead of a single dump.

**A pre-existing, still-unexplained `Cannot find package 'react'` error** (both the official npm binary and our from-source build) at `packages/tui/src/config/index.tsx` — no direct `react` import exists anywhere in our source; the likely culprit is `@opentui/keymap`'s optional `src/react/index.js` adapter, but the exact transitive import chain was never confirmed. **Workaround (not a real fix):** `bun add react react-dom` at the opencode repo root satisfies whatever resolution was failing. This is harmless (confirmed no downstream breakage from adding a real React 19 to the tree — checked for duplicate/conflicting `solid-js` copies, found none) but the *actual* root cause of why react was needed is still unknown.

**Unresolved: the full interactive TUI (`opencode`, no subcommand) renders a permanently blank white screen — no crash, no error, no input box, just a blank frame forever.** Investigation (extensive, documented here so it isn't repeated):
- Ruled out: OpenTUI itself (a from-scratch minimal `@opentui/solid` "Hello, World!" app, installed fresh from the public npm registry with no relation to our monorepo, renders correctly in this exact container — the native renderer, terminal capability negotiation, and alt-screen handling all work fine).
- Ruled out: plugins (`opencode --pure` shows the identical blank screen).
- Ruled out: unstable dev branch (current HEAD is 1 unrelated commit ahead of the exact matching release tag).
- **Root-caused to:** the full TUI (unlike `opencode run` and `--mini`, both of which run the server in-process) spawns a **Bun `Worker` thread** (`packages/opencode/src/cli/tui/worker.ts`) and talks to it over a hand-rolled `postMessage`-based RPC (`packages/opencode/src/util/rpc.ts`). Traced with temporary debug logging (both in the worker's `onUnhandledRejection`/`onUncaughtException` handlers — which were silently-swallowing no-ops before this session, i.e. a worker-side exception would previously vanish with zero trace anywhere — and in the RPC `listen()`/`client()` functions) that the RPC channel itself works fine (`checkUpgrade`, `shutdown`, and `global.event` messages all flow correctly both ways), but **no `fetch` RPC call is ever made** during the whole hang — meaning the App component tree's providers (`SDKProvider`/`DataProvider`/`SyncProvider`, which would need to fetch config/session data over the worker-backed `transport.fetch`) never even attempt their first request. The hang is therefore somewhere in `App`'s early synchronous/reactive setup in `packages/tui/src/app.tsx` (candidates not yet individually eliminated: `renderer.waitForThemeMode(1000)` — confirmed NOT to hang in isolation via a standalone repro script using the exact same `createCliRenderer` config — or something in the `ready`-gating `pluginHost.start()` call, or one of the ~15 nested context providers between `ErrorBoundary` and the app body). **Workaround: `opencode --mini` uses the same non-worker code path as `opencode run` and is a fully working interactive substitute** (confirmed: renders the real UI, takes prompts, streams responses) — use this until the worker-thread hang is properly root-caused. Debug instrumentation added during this investigation was reverted; if resuming this, re-add logging to `packages/opencode/src/cli/tui/worker.ts`'s error handlers and to `app.tsx`'s provider tree first.

## Phase 5 — Codex integration: Responses API protocol gaps (this was the deep one)

Goal: use this local model as Codex CLI's backend with full parity to a real hosted-model experience — visible reasoning during long turns, working `apply_patch` file edits with a real diff view, no infinite retry loops. Codex (`@openai/codex` npm package — a thin JS launcher around a **compiled Rust binary** at `.../vendor/x86_64-unknown-linux-musl/bin/codex`; the `openai/codex` GitHub repo is open-source but what's *installed* here is a prebuilt binary, not buildable-from-source-in-place without a full Rust toolchain + rebuild) talks to our server via `-c model_providers.lucebox.wire_api="responses"` → our `/v1/responses` endpoint, driven by `/root/.codex/model_catalog.json` (hand-built model metadata; see Phase before this for its schema history).

**Bug 1: reasoning phase was 100% silent at the wire level — zero SSE events for potentially 40+ seconds, then everything (reasoning + answer) arrived in one burst.** This was *not* a rendering/buffering issue on Codex's side (verified directly with `strace -f -tt -e trace=write` on the Codex process: real write() syscalls, so whatever data it had, it *was* flushing promptly) — it was that our own server's `emit_reasoning_delta()` in `lucebox/server/src/server/sse_emitter.cpp` had a literal no-op for `ApiFormat::RESPONSES` (it correctly emits `reasoning_content` deltas for `OPENAI_CHAT` and `thinking_delta` content blocks for `ANTHROPIC`, but the `RESPONSES` case was just `break;`). Reasoning tokens were captured into `reasoning_text_` server-side but never streamed over the wire in Responses format — Codex's `show_raw_agent_reasoning = true` config had nothing to display, correctly, because we sent it nothing.
  - **Fix:** gave the Responses-format reasoning phase its own lazily-opened item lifecycle, matching the real OpenAI Responses API's reasoning-summary event shape: `response.output_item.added` (`type: "reasoning"`) → `response.reasoning_summary_part.added` → N × `response.reasoning_summary_text.delta` → `.done`/`.done` → `response.output_item.done`, *then* the message item opens (shifted to the next `output_index`). New emitter state: `started_in_thinking_`, `responses_reasoning_open_`, `responses_msg_item_open_`, `responses_output_index_`, `reasoning_item_id_`; new method `close_responses_reasoning_item()` called from every path that can end a reasoning phase (natural `</think>` close, a tool-call detected mid-reasoning, and end-of-stream/token-cap as a safety net). `model_catalog.json` also needed `"supports_reasoning_summaries": true` and `"default_reasoning_summary": "auto"` (were `false`/`"none"`) — Codex won't ask for/show summaries at all otherwise.
  - **Verified:** direct SSE trace showed 752 genuinely time-distributed `reasoning_summary_text.delta` events over 14.3s (was: 0 events, 100% silence) for the same prompt; real `codex exec` runs now show the dim/italic reasoning block appearing live instead of a single end-of-turn dump.

**Bug 2: `apply_patch` (the tool Codex uses for file edits, which is what drives the nice diff-view UI) was never even offered to the model.** `model_catalog.json`'s `experimental_supported_tools` was `[]`. **Fix:** add `"apply_patch_tool_type": "freeform"` (confirmed via Codex's own Rust deserialization error — `"freeform"` is currently the *only* valid enum variant — do not add `experimental_supported_tools: ["apply_patch"]`, the tool type field is what actually gates it).

**Bug 3: once offered, every `apply_patch` call failed with `Fatal error: tool apply_patch invoked with incompatible payload`.** Root cause (confirmed by capturing the actual model output via a local logging HTTP proxy in front of the server): `apply_patch` is a Responses API **"custom" tool** (`{"type": "custom", ..., "format": {"type": "grammar", "syntax": "lark", ...}}` — a real, distinct wire concept from a normal `"type": "function"` tool, confirmed against OpenAI's public docs). A custom tool's argument is a **raw string** (the patch text itself, field name `input`), not a JSON object — but our model (like effectively every open-weight model's tool-calling fine-tune) is architecturally incapable of *not* wrapping arguments in a JSON object regardless of system-prompt wording, so it always produced `{"patch": "*** Begin Patch\n..."}`. Two-part fix, both in our own server:
  1. `tool_parser.cpp`'s `add_call()` lambda: when `fn_name == "apply_patch"` and the parsed arguments are a JSON object with exactly one string-valued key, unwrap it to a bare string (regardless of what key name the model happened to use).
  2. `sse_emitter.cpp`: added `is_custom_tool(tools, name)` (checks the request's declared `tools` array for `"type": "custom"` on that tool name) and branched every RESPONSES-format tool-call emission site (the mid-stream delta/done pair, and the final `response.completed` output manifest) between the normal `function_call`/`response.function_call_arguments.*` shape and a `custom_tool_call`/`response.custom_tool_call_input.*` shape (item fields `call_id`/`name`/`input` instead of `arguments`; the `.delta`/`.done` event names are the analogous-by-convention names for custom tools — OpenAI's public docs confirm the item type and field names but don't document the exact streaming event names, so these were inferred from the established `function_call_arguments` naming pattern and empirically confirmed working end-to-end).
  - Also had to teach the model apply_patch's expected patch syntax at all (`model_catalog.json`'s `base_instructions` — our own generic system prompt never mentioned it) — this alone didn't fix the payload-shape bug but was necessary for the model to produce a *syntactically* valid patch once the shape bug was also fixed.
  - **Verified:** real diff view now renders correctly end-to-end (`diff --git a/... b/...`, `@@ ... @@`, `+`/`-` lines) for both file creation and in-place edits; 5 consecutive test rounds all succeeded with exactly one `apply_patch` call each (see Bug 4 — before this fix was complete, it looped).

**Bug 4 (symptom of Bug 3, fixed by the same root cause plus one more piece): `apply_patch` calls looped indefinitely** — the model kept retrying the same patch because it never saw confirmation that a previous attempt succeeded. Cause: `http_server.cpp`'s `normalize_chat_messages()` (which reconstructs conversation history from a Responses-format `input` array for re-prompting) only recognized `function_call`/`function_call_output` item types when replaying history — a `custom_tool_call`/`custom_tool_call_output` item (which is what a *previous* apply_patch call/result look like once Codex sends the conversation back on a later turn) was silently dropped, so the model's own past patch attempts and their results vanished from what it could see. **Fix:** added `custom_tool_call` (wrapped into the same one-key-JSON shape used by `render_tool_call_xml` for replay, so it looks like a normal past tool call to the model) and `custom_tool_call_output` (identical handling to `function_call_output`) to both `normalize_chat_messages()` and `is_continuation_request()`.

**Investigated and found NOT to be bugs (Codex's own design, not ours to fix):**
- `codex exec`'s `apply_patch` diff output is plain, uncolored text (no red/green ANSI) — confirmed this is *not* gated on being inside a git repo (tested both in and out of one, identical plain-text result); `codex exec` is documented as Codex's own **non-interactive** mode, and this is almost certainly a deliberate simplification for scripting/piping use there, unrelated to our local-model backend. Full interactive `codex` (real TUI) very likely colors it — not independently confirmed in this session because automating keystrokes into a full-screen TUI proved unreliable (same class of limitation as the OpenCode TUI investigation above); ask the user to eyeball a real interactive session directly.
- A synthetic 6-turn and a clean 2-turn `codex exec resume` chain both showed reasoning displaying correctly on every turn, and the resumed conversation history correctly included the assistant's own prior final-text message (`type: "message", role: "assistant"`) — a suspected "model can't see its own previous reply" bug turned out to be an artifact of a test run where turn 1 had been cut off mid-task by an impatient `timeout`, not a real defect. The user's separately-reported "working animation disappears after several turns of real interactive use" was **not reproduced** in these synthetic tests — needs a real repro from an actual interactive session (turn number, what was happening) to chase further.
- `experimental_supported_tools: ["update_plan"]` (to try to unlock Codex's task-list/plan UI) did not make `update_plan` appear in the tools offered to the model, and produced no error either — the gate for this specific tool (if reachable via config at all) is still unknown; not worth further binary archaeology unless requested again.

**Diagnostic techniques worth reusing** if more Responses-API protocol gaps turn up:
- A tiny local logging HTTP proxy (plain `http.server` in Python, forwards to `127.0.0.1:18081` while writing the request/response body to a file) placed in front of the server via `-c model_providers.lucebox.base_url=http://127.0.0.1:<proxy_port>/v1` is the fastest way to see *exactly* what Codex sends and what our server streams back, without needing `tcpdump` (not installed in this container).
- `strace -f -tt -e trace=write -o <file> <cmd>` (run inside `script -qc "..." <screenlog>` so Codex still detects a real TTY) gives ground-truth timing of what a CLI actually writes to its own stdout, independent of any test-harness-induced buffering (e.g. piping through `grep`, or `script --timing`'s coarser granularity, both of which can make genuinely-incremental output look like one burst).
- `strings <the codex binary>` (a compiled, locally-installed CLI we run — not decompiling anything we don't already execute) is a legitimate way to recover exact field/enum names for undocumented parts of a wire protocol; confirmed a "known-invalid value" error message from Codex's own Rust deserializer (`unknown variant 'function', expected 'freeform'`) is an even faster way to get the exact valid enum set directly from the tool itself.

## Bug 7: KVFlash pool-seeding — long sessions past the pool paid full reprefill every turn (fixed)

**Symptom:** once a session's cumulative context permanently exceeded the KVFlash
resident pool (16384 tokens on this card), `snapshot_save()` refuses to save
forever (`cache_.cur_pos > kvflash_tokens_` guard in `qwen35_backend.cpp`), so
`http_server.cpp` can never again find a restore slot and always calls plain
`generate()` with the full cumulative prompt — a ~24-34s full pooled/evicting
reprefill, repeating on **every** round-trip for the rest of that session/agent
loop. This is the actual mechanism behind "every message takes minutes with no
token output" once a long agent session runs past the pool.

**Fix (`qwen35_backend.h`/`.cpp`):** since the live KVFlash pool + recurrent
state are never torn down between HTTP requests on the same backend instance
(single FIFO worker, no other session interleaved), a new request that's a
genuine extension of the previous one can skip `kvflash_pager_.reset()` and the
recurrent-state reset, and only prefill the new suffix directly onto the
already-resident pool. `do_prefill()` gained a `kv_pool_continue` flag; a new
`kvflash_prompt_boundary_pos_` member tracks the end of the last genuine
*client* prompt (not `cache_.cur_pos`, which also includes this backend's own
raw generated tokens — not safe to match against a freshly re-rendered next
prompt, since chat-template retokenization at a message boundary is not
guaranteed byte-stable). `generate_impl()` computes a **longest-common-prefix**
match (not strict equality) against `kvflash_history_[0, boundary)` before
deciding whether to reset; any divergence (including the model's own last
reply) is simply reprocessed as part of the delta — always correct, just less
of a speedup when the match is short.

**Verified:** cold (post-restart) full reprefill of a ~24k-token conversation
vs. the same conversation continued incrementally produced the identical
answer (33.8s cold vs 0.2s continued, both correct); 10 consecutive
arithmetic-check turns past the pool boundary all correct at ~0.2s each
instead of ~24-34s; short in-pool sessions still use the existing
snapshot-restore path unaffected (`restore=true` unchanged there); a genuine
conversation switch (near-zero LCP) correctly falls back to a near-full
reprefill, no corruption.
