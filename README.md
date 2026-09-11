# Codex CLI + local Qwen 27B on a 24GB GPU

Configuration and patches for running [OpenAI Codex CLI](https://github.com/openai/codex)
entirely against a **local, uncensored Qwen3.8-27B model** served by
[Lucebox](https://github.com/Luce-Org/lucebox) (a speculative-decoding
inference engine), on a single 24GB-VRAM GPU (tested on an RTX PRO 4000
Blackwell) — full 262144-token context, real-time streamed reasoning, and
working `apply_patch` file edits, matching the experience of Codex against a
hosted model.

## Hardware / model

- GPU: 24GB VRAM (RTX PRO 4000 Blackwell in testing; any 24GB CUDA GPU should
  work)
- Model: `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` (Q4_K_M), served via
  Lucebox's DFlash2 speculative decoding + KVFlash bounded-residency KV cache
  (the piece that makes the model's full 262144-token context fit in 24GB —
  KVFlash bounds the *resident* KV pool independently of the logical context
  length, paging cold context to host RAM)
- Result: ~59-62 tok/s decode at full 262144 context (GPU-clock-throttling
  under sustained load is the real ceiling here, not a software one — see
  `HANDOVER.md`)

## What's in this repo

- **`lucebox-patch/`** — five modified Lucebox server source files (see
  `NOTICE.md` for how to apply them against an upstream clone). These close
  several gaps between Lucebox's OpenAI-Responses-API implementation and what
  Codex CLI actually expects on the wire:
  - Streams the model's reasoning/thinking phase as real-time
    `response.reasoning_summary_text.delta` events (previously: total silence
    during reasoning, sometimes 40+ seconds, then everything arriving in one
    burst)
  - Correct support for `apply_patch` (Codex's file-edit tool), including its
    non-JSON "freeform" argument format and the `custom_tool_call` wire shape
    Codex uses for it — needed for the diff view to render and for edits to
    not loop indefinitely retrying
  - Support for Codex's "namespace" tool grouping (`multi_agent_v1` and
    similar), which bundles several independently-callable sub-tools under
    one declaration
- **`codex/`** — the Codex CLI config (`config.toml`) and hand-built model
  metadata (`model_catalog.json`) needed to point Codex at a local
  OpenAI-Responses-API-compatible server instead of a hosted model
- **`supervisor/llama-server.sh`** — the exact working set of Lucebox server
  flags (context size, KV cache dtype, KVFlash pool sizing, DDTree/DFlash2
  speculative-decode settings) arrived at after a long tuning pass — see
  `HANDOVER.md` for what was tried and rejected along the way
- **`HANDOVER.md`** — full testing log: what was tried, what worked, what
  didn't and why, kept so none of it needs re-discovering

## Setup

1. Build [Lucebox](https://github.com/Luce-Org/lucebox) from source, then
   overwrite the files listed in `NOTICE.md` with the ones in
   `lucebox-patch/` and rebuild.
2. Download the model + draft/drafter GGUFs referenced in
   `supervisor/llama-server.sh` and adjust the paths there to match where you
   put them.
3. Run the server (`supervisor/llama-server.sh`, or adapt its `dflash_server`
   invocation to your own process manager). Confirm
   `curl http://127.0.0.1:18081/v1/models` reports the context length you
   expect before doing anything else.
4. Install Codex CLI, copy `codex/config.toml` to `~/.codex/config.toml` and
   `codex/model_catalog.json` to `~/.codex/model_catalog.json` (or point
   `model_catalog_json` in `config.toml` at wherever you put it).
5. Set `LUCEBOX_API_KEY` to any non-empty value (the local server doesn't
   enforce auth — Codex just needs the env var to exist) and run `codex`.

## Notes

- `approval_policy = "never"` and `sandbox_mode = "danger-full-access"` in
  the sample `config.toml` disable Codex's safety prompts entirely — this
  matches a fully trusted, single-user local setup. Tighten these for any
  other use case.
- See `HANDOVER.md` for the full rationale behind every flag, the bugs that
  were found and fixed to get here, and known remaining gaps.
