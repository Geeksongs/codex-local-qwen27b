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
