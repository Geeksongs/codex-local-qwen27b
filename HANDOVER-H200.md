# H200 adaptation — session log (2026-09-18)

Adapts the original 24GB-RTX-PRO-4000 setup documented in `HANDOVER.md` to a
shared 4x NVIDIA H200 (143.8GB each) box at `/workspace/python_song` on this
instance. Primary goal per user: maximize inference speed and context length
for OpenCode, since VRAM is no longer the constraint. Codex CLI was left
configured (the repo already ships it) but OpenCode was the actual target.

## What changed vs. the original repo

### 1. `lucebox-patch/` is stale against current upstream — do not apply it as-is

`NOTICE.md` lists 5 patched files (`tool_parser.*`, `sse_emitter.*`,
`http_server.cpp`), but `lucebox-patch/` on disk actually contains 7 — it
also has `qwen35_backend.h/.cpp` (the Bug 7 KVFlash-pool-continuation fix)
that was added to the directory without updating `NOTICE.md`.

Both sets fail to compile against current `Luce-Org/lucebox` HEAD:

- `qwen35_backend.h` reverts `Qwen35Config::target_path`/`draft_path` from
  `std::string`/`std::optional<std::string>` back to a `const char*` shape
  that predates a refactor upstream — breaks `backend_factory.cpp`,
  `qwen35moe_backend.cpp`, `bailingmoe3_backend.cpp` and more with
  "cannot convert ... to const char*" / "static assertion failed".
- `http_server.cpp` predates upstream additions to `GenerationInputs`
  (`hint_tokens`, `stall_tool_prefix_tokens`, `stall_action_suffix_tokens`,
  `stall_skip_tokens`) and a `sse_error_close_chunks` header declaration
  that no longer matches — "has no member named" / "no declaration matches".

**Root cause**: Lucebox is a fast-moving repo (see its own `CODEX.md`,
`RESULTS.md`, `docs/MODEL_LOAD_BALANCING.md`, hybrid-MoE and DeepSeek V4
sections — none of which existed when this repo's patches were cut). It has
since grown its **own native `/v1/responses` Codex integration**
(`server/CODEX.md`), so the old hand-patched reasoning-streaming /
`apply_patch` / tool-namespace fixes may already be superseded upstream.

**What this session did**: reverted all 7 files to clean upstream
(`git checkout --`) and built that instead. It compiles and serves correctly
against both raw `curl` and `opencode run` (see below). **Not independently
re-verified**: whether upstream's native Responses-API path still needs the
old Bug 1-4 fixes from `HANDOVER.md` (silent reasoning stream, `apply_patch`
freeform/custom-tool payload shape, tool-call replay dropping
`custom_tool_call` history) for a *Codex CLI* session specifically — this
session validated the OpenCode path end-to-end, not Codex CLI. If Codex CLI
integration misbehaves (empty reasoning stream, `apply_patch` loops), re-run
the diagnostic techniques in `HANDOVER.md` Phase 5 against current source
rather than re-applying the stale patch files.

### 2. Build target: `-DCMAKE_CUDA_ARCHITECTURES=90` (H100/H200), not 120

The original repo was tuned for Blackwell (SM120, RTX PRO 4000). H200 is
Hopper, SM90. Confirmed via the upstream README's own arch table
(`90 = H100`; H200 is the same compute capability, confirmed via
`nvidia-smi --query-gpu=compute_cap` → `9.0` on all 4 GPUs here).

```bash
export PATH=/usr/local/cuda/bin:$PATH   # nvcc isn't on PATH by default here
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build --target test_dflash dflash_server -j 96
```

CUDA 13.0 toolkit was already installed at `/usr/local/cuda` but not on
`PATH`; added to `~/.bashrc`.

### 3. Model precision: Q8_0, not Q4_K_M — because VRAM is abundant, not scarce

The model card's own perplexity table shows every quant from Q4_K_M through
Q8_0 sits inside one shared error-bar band around f16 — "not separable from
the f16 or from each other." With ~90-100GB free per H200 (this is a shared
box; ~40-60GB was already in use by other users' jobs on each GPU before
this session started), there is no VRAM reason to pick the smallest safe
quant. Used:

- Target: `Qwen3.8-27B-Uncensored-Q8_0.gguf` (29.0GB, fused MTP — MTP block
  is inert here since Lucebox drives speculative decode via the separate
  DFlash2 draft, not the MTP head)
- Draft: DFlash2, converted from `incoai/Qwen3.8-27B-DFlash2` (bf16
  safetensors) via Lucebox's own `scripts/convert_dflash_to_gguf.py` →
  `scripts/quantize_dflash_draft.py --scheme q8_0` (same recipe as the
  original `HANDOVER.md`; Lucebox needs its own GGUF tagging, the generic
  llama.cpp-ecosystem DFlash2 GGUF isn't compatible)
- KVFlash relevance-scoring prefill drafter: `Qwen/Qwen3-0.6B-GGUF`
  `Qwen3-0.6B-Q8_0.gguf` (same as original)

All under `/workspace/python_song/models/`.

### 4. KVFlash: kept, not removed — it's a speed feature here, not just a VRAM one

Initial assumption going in was "abundant VRAM means we can skip KVFlash's
bounded-residency paging and just use full native KV cache." **This is
wrong** — checked `optimizations/kvflash/README.md`'s own measured numbers:
decode speed with a full (unbounded) cache *degrades* as context grows
(13.1 tok/s at 256K vs. KVFlash's flat 38.6 tok/s on their reference card,
because the full-cache per-step KV read scales with context length while
KVFlash's scales with the fixed pool size instead). Kept `--kvflash auto`
(resolves to the same 16384-token pool as before — `auto`'s cap is
explicitly VRAM-independent, "capped where decode speed stays near the flat
optimum," per the same doc, overridable via `DFLASH_KVFLASH_MAX_POOL` but
not changed here since a wider pool trades this same decode speed away).

### 5. Flags changed from the original 24GB-card config, with on-hardware A/B numbers

Held fixed from the original: `--max-ctx 262144`, `--kvflash auto`,
`--cache-type-k/v f16`, `--prefix-cache-slots 32`, `--prefill-cache-slots 16`.

Swept on this hardware (short synthetic prompt, then re-validated against a
realistic ~24.8K-token prompt for OOM/correctness — same methodology
`HANDOVER.md` used, see its Phase-1 "load-bearing lesson"):

| Flag | Original (24GB card) | H200 | Why |
|---|---|---|---|
| `--draft-block-size` | 8 | **16** | 182.6 tok/s vs 127.3 tok/s on a short prompt (+43%), accept_rate 0.56 vs 0.35; re-checked at 24.8K-token prompt: 81.5 vs 79.4 tok/s, no regression, no OOM. Lossless (chain verification is greedy/byte-identical at any width per upstream's own R9700 write-up) — this is a free win specific to having spare compute per step, not something the 24GB card's docs even tested this high. |
| `--ddtree-budget` | 22 | **22 (unchanged)** | Tried 32: dropped to 125.6 tok/s on the same short prompt (worse than block=16/budget=22's 182.6) — the original card's finding that wider trees don't pay off held on H200 too, so this was *not* just a VRAM ceiling as first assumed. Do not widen this further without re-benchmarking. |
| `--think-max-tokens` | 14336 (explicit override) | **not set (family default applies)** | Not carried over — the original override was Codex-specific reasoning-budget tuning from `HANDOVER.md`'s Codex integration section; this session's target is OpenCode, which doesn't have the same Codex-specific "no reasoning.effort tier" problem. Revisit if Codex CLI is put into real use here. |
| `--agent-turn-cache` | off (VRAM-OOM'd against the rollback-cache buffer on 24GB) | **still off — not re-enabled** | The original blocker (rollback-cache buffer contention on a nearly-full 24GB pool) doesn't apply here, but the KVFlash Bug 7 pool-continuation fix that made this safe (`qwen35_backend.cpp` patch) does not apply cleanly against current upstream (§1) and was not re-derived this session. Both realistic-prompt test requests in this session had `agent_turn_cache_hit: false` and completed correctly without it. Turning this on is a real follow-up speed opportunity for long multi-turn OpenCode sessions but needs its own validation pass first — don't flip it on without re-running a long-session test past the 16384-token pool boundary.

**Measured, this session, single H200 (shared box — GPU0 had ~59GB already in
use by other tenants' jobs before this server started; own footprint is
~30GB weights+draft, growing to ~93GB with KV/scratch at the 24.8K-token
test)**:

- Short prompt (30 tokens in, 252-253 out): **182.6 tok/s** decode
  (vs. the RTX PRO 4000's 59-62 tok/s at full 262144 context — direct
  comparison isn't apples-to-apples since that number was measured at full
  context with a different prompt shape, but it's the same "cold, short
  generation" regime and the gap tracks H200's ~7x memory-bandwidth
  advantage, 4.8TB/s vs. 672GB/s)
- Realistic ~24.8K-token prompt (bigger than `HANDOVER.md`'s own ~11.7K
  reference prompt): prefill 20.4-22.9s (~1.1-1.2K tok/s), decode 79.4-81.5
  tok/s, `ok=true`, no OOM, no `prefill_failed`
- `/v1/models` confirmed `context_length: 262144` in every configuration
  tested

Not done this session (flagged as follow-up, not because it's expected to
matter, but because it wasn't checked): multi-GPU placement
(`--target-devices cuda:0,cuda:1` + `--target-split-mode`) — a single H200
already holds the model, draft, and full 262144-context KVFlash pool with
room to spare, so this wasn't needed for either VRAM or (per the sweep above)
speed. The other 3 H200s on this box are untouched and free for other work.

## OpenCode integration

Installed the **official** `opencode-ai` npm package (not the `Luce-Org`
source fork `HANDOVER.md` used) — `HANDOVER.md`'s own final entry on this
topic ("OpenCode integration — blank-screen bug resolved, default model
fixed") already concluded the from-source fork's TUI hang was actually a
generic `npm install -g` skipping `postinstall.mjs` (native binary staging),
fixable with a plain `--allow-scripts` install of the official package — so
this session started from that already-known-good state instead of
re-deriving it:

```bash
npm install -g --allow-scripts=opencode-ai opencode-ai@latest
```

Config at `~/.config/opencode/opencode.json` (and mirrored into this repo's
`opencode/opencode.json`) points `lucebox/dflash` at
`http://127.0.0.1:18081/v1`, set as both `model` and `small_model` so it's
active by default with no manual `ctrl+x m` switch. `opencode run "..."`
verified working end-to-end against the H200 server. The `opencode-patch/`
directory in this repo (streaming fix for `opencode run`'s
`message.part.delta` handling) was **not applied** — it patches TypeScript
source in a from-source checkout, not the npm-installed build; not needed
since `opencode run` already streamed/returned correctly in this session's
test. Full interactive TUI (`opencode`, no subcommand) was not smoke-tested
this session (only `opencode run`) — if it blanks, `HANDOVER.md`'s documented
workaround is `opencode --mini`.

## Launch

`supervisor/llama-server-h200.sh` (same context-length self-check pattern as
the original `llama-server.sh`, paths updated for
`/workspace/python_song/{lucebox,models,logs}`).

## Open follow-ups (not done, flagged for whoever picks this up)

1. `--agent-turn-cache` — real speed opportunity for long OpenCode sessions,
   needs the Bug 7 continuation logic re-derived against current upstream
   `qwen35_backend.cpp` first (§1), then a long-session test past the
   16384-token pool boundary before enabling.
2. Codex CLI path not re-validated against current upstream's native
   `/v1/responses` support (§1) — only OpenCode was checked end-to-end.
3. Full interactive OpenCode TUI (not just `opencode run`) not smoke-tested.
4. No GitHub push credentials available in this environment — the `h200`
   branch exists locally with these changes committed but has not been
   pushed to `origin` (`Geeksongs/codex-local-qwen27b`). Needs `gh auth
   login` or a token from the user.

## Session 2 — full parameter sweep + context-length ceiling investigation (2026-09-18)

### Context length: 262144 is a hard ceiling, not a tunable

Checked the GGUF metadata directly (not just the model card prose):

```
qwen35.context_length = 262144
qwen35.rope.freq_base = 10000000.0
qwen35.rope.dimension_count = 64
```

No `rope.scaling.*` keys of any kind (no YaRN factor, no NTK-aware scaling, no
linear interpolation). This is the model's native trained context, not an
artificially-capped value with an extension recipe sitting unused in the
checkpoint.

Checked whether Lucebox itself can force an extension anyway: grepped the
whole server source for `yarn`/`rope-scale` CLI flags. **YaRN support exists
in Lucebox, but only for the `laguna` (Laguna-XS.2) architecture**
(`src/laguna/laguna_target_loader.cpp` reads `laguna.rope.scaling.factor`
etc. from GGUF metadata) — there is no equivalent for `qwen35`, and no CLI
flag (`--rope-freq-scale`, `--yarn-*`, etc.) exposed by `dflash_server`
generally. `--help` confirms this: zero rope/yarn/scaling flags at all.

**Conclusion**: 262144 is the real ceiling for this model on this engine.
Extending it would require either (a) engine-side work to add qwen35 YaRN
support (nontrivial, and unvalidated — RoPE extension without fine-tuning
measurably degrades long-context recall/coherence past the trained length,
which conflicts with this deployment's quality floor), or (b) a different
checkpoint trained/extended further. Not attempted — recommended against
unless the user explicitly wants to trade quality for a longer nominal
window.

For scale: 262144 tokens is already near the top of what any local
open-weight model publishes (most cap at 32K-128K); this is not a small
number, it's within range of the largest hosted commercial context windows.

### Full parameter sweep (GPU3, quietest of the 4 at sweep time, held constant across all trials for a fair A/B — cross-GPU comparisons earlier in this session were shown to be too noisy to trust)

Base held fixed: target/draft Q8_0, `--max-ctx 262144 --kvflash auto --cache-type-k/v f16`.

**`--ddtree-budget`** (with `--draft-block-size 16`, already-established ceiling):

| budget | decode tok/s | accept_rate |
|---|---|---|
| 16 | 172.1 | 0.705 |
| 18 | 180.3 | 0.632 |
| 20 | 178.5 | 0.574 |
| 22 (prior default) | 184.1 / 185.3 / 184.5 | 0.556 |
| **24 (new default)** | **189.6 / 189.4 / 189.5** | 0.543 |
| 26 | 186.4 | 0.504 |
| 28 | 184.2 | 0.472 |
| 30 | 183.2 | 0.446 |
| 32 | 164.9 | 0.446 |
| 36 | 162.6 | 0.403 |
| 40 | 160.3 | 0.366 |

24 reliably beats 22 by ~3% across 3 repeated trials each (not noise) —
adopted as the new default. Confirms the shape found on the 24GB card
(peak then falloff) but the peak moved from 22 to 24 on this hardware/this
draft pairing. 32+ regresses clearly, same as before — **not** a VRAM
ceiling this time (H200 has room to spare at budget 40), a genuine
compute/tree-verification tradeoff.

**Everything else swept at block=16/budget=24 — all within ~1% of each
other, i.e. no real effect, confirming the same "flat" findings
`HANDOVER.md` reported on the 24GB card**:

| Flag | Result |
|---|---|
| `--specla --specla-top-k 4` | 189.7 tok/s (no change) |
| `--specla --specla-top-k 8` | 189.4 tok/s (no change) |
| `--draft-residency persistent` | 189.0 tok/s (no change) |
| `--chunk 1024` | 190.1 tok/s, prefill 421ms vs 427ms baseline — marginal prefill win, adopted since free |
| `--chunk 2048` | 189.4 tok/s (no change) |
| `--kvflash-tau 32` | 189.1 tok/s (no change) |
| `--kvflash-tau 128` | 189.4 tok/s (no change) |

### Final production config (deployed, running)

```
--target-device cuda:0 --draft-device cuda:0 \
--draft-block-size 16 \
--max-ctx 262144 \
--kvflash auto \
--ddtree --ddtree-budget 24 \
--chunk 1024 \
--cache-type-k f16 --cache-type-v f16 \
--prefix-cache-slots 32 --prefill-cache-slots 16
```

Re-validated on production (GPU0) against the same ~24.8K-token realistic
prompt used throughout this doc: `ok=true`, decode 79.7 tok/s, prefill 22.6s,
no OOM. Short-prompt decode varies 144-190 tok/s across checks purely from
other tenants' load on the shared GPU (confirmed by watching the same exact
config swing across repeated measurements) — this is now the dominant source
of variance, not any remaining engine flag.

`supervisor/llama-server-h200.sh` updated to match (`--ddtree-budget 24
--chunk 1024`).

Not swept (documented as still-open, not because expected to matter):
`--admission-coalesce-ms`, `--fa-window` (left at 0/full-attention
deliberately — the docs explicitly warn windowed attention risks tool-use
correctness at long context, not worth trading for speed here),
`--agent-turn-cache` (same blocker as Session 1: needs the Bug-7
continuation fix re-derived against current upstream first).

## Session 3 — YaRN context extension to 1.5M, and a major KVFlash correctness finding (2026-09-18)

User goal: push past the model's native 262144-token ceiling toward the
officially-documented 1M-token YaRN extension
(`Qwen/Qwen3.8-27B`'s own README, "Processing Ultra-Long Texts" section).

### YaRN was not wired up in Lucebox for qwen35 target models at all

Confirmed by reading source, not guessing: `ggml`'s YaRN implementation
(`rope_yarn()` in `ggml-cuda/rope.cu`, the standard Quesnelle/Peng algorithm)
is fully present and already used elsewhere in this codebase (`laguna`
target loader, and a narrow "legacy 8-layer drafter" special case) — but
`TargetWeights` (the qwen35 target's own weight/config struct,
`src/internal.h`) had no YaRN fields at all, and the qwen35 target's actual
`ggml_rope_multi()` call sites (`src/qwen35/qwen35_target_graph.cpp`) were
hardcoded to `n_ctx_orig=0, freq_scale=1.0, ext_factor=0.0` — plain RoPE,
no override possible from the CLI. `--max-ctx` had no validation against the
GGUF's native `context_length` either — you could already ask for a huge
`--max-ctx` and the server would allocate it, it would just silently produce
degraded/extrapolated output past the trained length with no warning.

**Added target-model YaRN support** (new code, not a patch — this is
original engineering done this session, not from `codex-local-qwen27b`'s
`lucebox-patch/`):

- `src/internal.h`: 6 new YaRN fields on `TargetWeights` (mirrors the
  existing `DraftWeights` fields) plus `native_context_length` (read from
  the GGUF's own `<arch>.context_length` key in
  `src/qwen35/gguf_target_loader.cpp`).
- `src/qwen35/qwen35_target_graph.cpp`: the two `ggml_rope_multi()` call
  sites (Q and K, inside the function used by all three qwen35 attention
  block builders) now read `w.rope_n_ctx_orig/freq_scale/ext_factor/
  attn_factor/beta_fast/beta_slow` from the weights struct instead of the
  hardcoded disabled values.
- New CLI flags: `--yarn-factor <F>` (>1.0 enables; unset = fully disabled,
  byte-for-byte the old behavior), `--yarn-orig-ctx <N>` (default: the
  GGUF's own native context length), `--yarn-beta-fast`/`--yarn-beta-slow`
  (ggml/llama.cpp standard defaults 32.0/1.0).
- Plumbed through the full `BackendArgs -> BackendPlan::Speculation ->
  Qwen35Config` pipeline (`backend_args.h`, `backend_factory.h`,
  `backend_plan.cpp`, `backend_factory.cpp` — only the qwen35 non-layer-split
  branch; layer-split and qwen35moe were not touched, not needed for this
  single-GPU deployment).
- `Qwen35Backend::apply_target_yarn_override()` (new method, called from
  both `init()` and `unpark()` — the second load path exists for VRAM-
  pressure park/unpark cycling and needs the override re-applied on every
  reload) sets the override and prints a startup banner, or a `WARNING` to
  stderr if `--max-ctx` exceeds native length with no `--yarn-factor` set
  (catches the previously-silent degradation case) or exceeds the
  YaRN-extended range the given factor actually covers.
- **Verified backward-compatible**: with no `--yarn-factor`, output and
  speed are unchanged from before this session (confirmed byte-for-byte
  coherent short-prompt output, same speed).
- One debugging note for whoever touches this next: `std::printf` (stdout)
  in this server is block-buffered when redirected to a file (not
  line-buffered), so a log line written early in startup may not appear in
  a tailed log file until much later output forces a flush — this cost real
  time this session chasing a "why didn't my log line print" false alarm.
  Use `std::fprintf(stderr, ...)` with an explicit `fflush` for anything
  you need to see immediately during debugging.

### Major finding: KVFlash silently corrupts long-context retrieval on this hardware — do not use it for anything that needs real long-context recall

This reverses Session 1/2's decision to keep `--kvflash auto`. Built a
needle-in-haystack test (`~296K`-token haystack of near-duplicate filler
code, one planted fact, asked to recall it) and ran it under several
conditions:

| Config | Context | KVFlash | Result |
|---|---|---|---|
| YaRN factor=4.0 | 296577 (34K past native) | `auto` (16384 pool) | **WRONG** (`XQ7-ALPHA` vs actual `XQ-7734-ZETA`) |
| YaRN factor=1.5 | 296577 | `auto` (16384 pool) | **WRONG** (`XQ7-44`) |
| No YaRN (native) | 243297 (well within 262144) | `auto` (16384 pool) | **WRONG** (`NIGHTINGALE` — total hallucination) |
| YaRN factor=1.5 | 296577 | **off** (full resident) | **CORRECT** (`XQ-7734-ZETA`, exact) |

The third row is the important control: the failure reproduces with **no
YaRN involved at all**, entirely within the model's native trained context,
with KVFlash's pool active. This isolates the bug to KVFlash, not to the
YaRN work above. Root cause (not fully diagnosed, but well-characterized):
KVFlash's bounded pool (16384 tokens = ~5.5% of a 296K-token prompt) relies
on a relevance-scoring drafter (`--prefill-drafter`, confirmed active via
the startup banner: `policy=drafter (attaches on first reselect)`) to decide
which chunks stay resident. For this prompt shape — a fact sentence
embedded in a long run of near-duplicate boilerplate code — that scorer
evidently fails to keep or recall the relevant chunk. KVFlash's own
published numbers (`optimizations/kvflash/README.md`) claim 88-100% needle
recall at 6% residency; this session's real-world result was 0/3. **Prompt
shape matters and the published benchmark does not transfer to every
prompt** — do not trust it without testing your own actual workload's shape.

**Consequence for this deployment**: KVFlash is no longer used at all.
Production runs a full resident KV cache (no bounded pool, no relevance
scoring, no eviction) — slower to scale VRAM-wise, but the only mode
verified correct for real retrieval at long context on this hardware.

### KVFlash was also not faster here, reversing the other half of Session 1/2's reasoning

Session 1/2 kept KVFlash partly because its own docs claim full-cache decode
speed *degrades* with context length on a bandwidth-starved card (13 vs
38.6 tok/s at 256K on their RTX 3090 reference). Directly measured on H200,
same ~24.8K-token prompt, only the KVFlash flag changed:

- With `--kvflash auto`: 79.7 tok/s decode
- Without (full resident): **99.4 tok/s decode** — faster, not slower

And at a short 30-token prompt: 182-190 tok/s with KVFlash vs **280 tok/s**
without. H200's ~7x memory-bandwidth advantage over the RTX 3090, combined
with this model's hybrid architecture (only 16 of 64 layers are full
attention — the rest are linear-attention/SSM, so the "cost scales with
context" problem KVFlash solves only applies to a quarter of the layers to
begin with), means the bandwidth-scarcity problem KVFlash was built to solve
essentially doesn't bite on this hardware+model combination. KVFlash's own
paging/scoring/eviction bookkeeping becomes pure overhead once that's true.
**Net effect: disabling KVFlash was a win on both correctness and speed
here — a reversal, not a tradeoff.**

### VRAM ceiling, mapped empirically (q8_0 KV cache, no KVFlash, this model)

Switching `--cache-type-k/v` from f16 to q8_0 (still within this
deployment's "q8_0-or-better" quality floor) roughly halved the KV cache's
VRAM footprint, which is what makes a multi-hundred-GB-scale context
tractable at all without KVFlash. Ceiling mapped by binary search, each
point validated against both a trivial prompt and the realistic
~24.8K-token prompt used throughout this doc (own footprint only, net of
whatever other tenants were using on the same GPU at measurement time):

| `--max-ctx` | `--yarn-factor` | Own footprint | Margin after a real request | Verdict |
|---|---|---|---|---|
| 1,048,576 (1.0M) | 4.0 | ~65 GB | ~39 GB free | safe |
| **1,572,864 (1.5M)** | **6.0** | **~86 GB** | **~17 GB free** | **safe — deployed** |
| 1,835,008 (1.75M) | 7.0 | ~95 GB | ~8.6 GB free | works, thin margin, not chosen |
| 2,097,152 (2.0M) | 8.0 | — | — | **OOM** on the per-request rollback-cache allocation (needs ~3.8 GB it didn't have) |

The 2.0M failure is a clean, reported error (`ok=false`,
`error=prefill_failed`, `cudaMalloc failed: out of memory` in the log) — not
a silent corruption, consistent with this engine's general behavior
(`HANDOVER.md`'s own "always grep for `ok=false`" lesson holds).

**This is a shared box.** All of the above "own footprint" numbers are net
of whatever else was running on that GPU at measurement time, which moved by
tens of GB over the course of this session (observed one GPU's other-tenant
usage go from 53GB to 119GB). A ceiling that fits today is not guaranteed to
fit tomorrow. 1.5M was chosen over 1.75M specifically for extra headroom
against exactly this volatility, not because 1.75M didn't technically work.

### The real cost of a large static `--max-ctx`: normal-size requests get slower too, not just VRAM

Ran matched accuracy+speed batteries (5 trials each, short ~3.1K-token and
medium ~52K-token prompts, distinct planted fact per trial) against two
servers differing *only* in `--max-ctx`/`--yarn-factor`:

- Accuracy: **100% vs 100%** (5/5 both lengths, both configs) — with
  KVFlash off, static YaRN at factor=6.0 measurably cost **nothing** in
  exact-recall accuracy at these lengths in this test. (5 trials per cell —
  enough to rule out a large effect, not enough to rule out a small one.)
- Speed: this is where the real cost showed up —

| Length | 262144 ctx, no YaRN | 1.5M ctx, YaRN factor=6.0 | Slowdown |
|---|---|---|---|
| ~3.1K tokens | 2.7-3.2s | 6.9-7.3s | **~2.6x** |
| ~52K tokens | 43.3-43.8s | 111-116s | **~2.6x** |

This ~2.6x tax applies to *every* request once the server is configured for
1.5M, including ones nowhere near needing it — it's the cost of the larger
KV-cache/bookkeeping structures being sized for the configured `--max-ctx`
regardless of how much of it a given request actually uses, not a YaRN-math
cost (accuracy was unaffected). **This is a genuine, measured tradeoff, not
a hypothetical one**: a single server that supports up to 1.5M tokens is
~2.6x slower on typical (short/medium) requests than a server capped at the
native 262144. A dual-endpoint setup (fast default + on-demand large-context
endpoint, request routed by expected size) would avoid this but was not
built this session — flagged as the clear next step if this tax turns out
to matter in practice for real OpenCode usage.

### Deployed production config

Moved off the original `--target-device cuda:0` convention this session
started with, onto `cuda:3` (this session's least-loaded GPU — **verify
current load before assuming this holds**, shared-box usage moves). Old
`cuda:0`/`cuda:1`/`cuda:2` test instances from earlier in this session were
torn down; only the `cuda:3` instance remains.

```
--target-device cuda:3 --draft-device cuda:3 \
--draft-block-size 16 \
--max-ctx 1572864 \
--yarn-factor 6.0 --yarn-orig-ctx 262144 \
--ddtree --ddtree-budget 24 \
--cache-type-k q8_0 --cache-type-v q8_0 \
--host 127.0.0.1 --port 18097
```

No `--kvflash`, no `--prefix-cache-slots`/`--prefill-cache-slots` overrides
(defaults apply; these only cache exact-repeated prefixes, e.g. Codex/
OpenCode's own unchanging system prompt, and are unaffected by the KVFlash
finding above — that's a completely different code path). `opencode.json`
(both `~/.config/opencode/opencode.json` and this repo's `opencode/
opencode.json`) updated to point at `:18097` with `context: 1572864`.
`supervisor/llama-server-h200.sh` updated to match.

### Open follow-ups from this session

1. **Root-cause KVFlash's actual recall failure mode** rather than just
   working around it by disabling the feature — would need instrumenting
   `KvFlashCrossTokScorer`/the drafter-relevance path directly. Not done
   this session; disabling it was the pragmatic fix given the deployment's
   correctness requirement.
2. **The ~2.6x static-max-ctx speed tax** — a dual-endpoint (small fast
   default + large on-demand) setup would let normal requests keep native-
   context speed while still offering the 1.5M window for the rare request
   that needs it. Real engineering work (routing logic, possibly two
   `dflash_server` processes on separate GPUs with OpenCode/Codex config or
   a proxy picking between them by estimated prompt size) — scoped but not
   built this session.
3. **1.75M / 2.0M were not re-validated with the accuracy battery** (only
   the 1.5M config was) — if a future session wants to push past 1.5M, redo
   the accuracy battery there too, not just the load/OOM check.
4. Same open items as Session 1/2: `--agent-turn-cache` still off, Codex CLI
   path not re-validated, full interactive OpenCode TUI not smoke-tested.

## Session 4 — OpenCode TUI smoke test (2026-09-18)

Ran `tools/opencode_tui_smoke_test.py` (previously flagged as "not smoke-
tested" in every prior session's open-follow-ups list) against the current
1.5M-context production deployment.

**Found and fixed a real bug in the script itself**: `b"\xc2\xb7"`
(the middle-dot `·` used as a streaming-indicator marker) was written as a
literal non-ASCII character inside a `b"..."` bytes literal
(`b"·" in chunk`), which Python rejects outright (`SyntaxError: bytes can
only contain ASCII literal characters`) — the script could not have run as
committed. Fixed by encoding it explicitly: `"·".encode() in chunk`.

All three checks pass against the 1.5M deployment:

```
OK: UI rendered (prompt box visible)
OK: default model shows expected 'dflash (local H200, 1.5M ctx)' in the footer
OK: message round-tripped, got a reply
```

Note the script's `expect_model` default (`"dflash (local)"`) no longer
matches this session's model display name (`"dflash (local H200, 1.5M
ctx)"` — set in `opencode.json`'s `provider.lucebox.models.dflash.name`) —
pass it explicitly as the second CLI arg, or the default footer check will
`WARN` (not fail) on a name mismatch even though the model is actually
correct.
