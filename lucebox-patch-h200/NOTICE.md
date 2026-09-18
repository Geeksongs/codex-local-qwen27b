This directory contains **original engineering done in this session** (not a
carry-forward of `lucebox-patch/`, which is stale against current upstream —
see `HANDOVER-H200.md`'s Session 1 section for why): target-model YaRN rope
scaling support for the `qwen35` architecture in
[Luce-Org/lucebox](https://github.com/Luce-Org/lucebox), which had no such
support at all before this (YaRN existed in the codebase only for the
`laguna` architecture and a narrow legacy-drafter special case — see
`HANDOVER-H200.md`'s Session 3 section for the full writeup).

Built against upstream commit `f6c5417136630d155485f5b7b1fecd8613585b09`
(2026-09-17). Given how fast this repo moves (see Session 1's notes on
`lucebox-patch/` breaking against current HEAD within what was probably
weeks), **check that commit still matches before assuming these apply
cleanly** — re-derive against current HEAD if not, following the same
approach documented in `HANDOVER-H200.md` (the `ggml_rope_multi()` YaRN
plumbing pattern, not a blind patch application).

Two ways to apply:

1. `git apply lucebox-patch-h200/yarn-target-support.diff` from a clean
   clone at the commit above, or
2. Drop the 10 files here into the same relative paths (overwriting
   originals), matching `lucebox-patch/`'s existing convention.

Modified files, staged at the same relative path they occupy upstream:

- `server/src/internal.h`
- `server/src/qwen35/gguf_target_loader.cpp`
- `server/src/qwen35/qwen35_target_graph.cpp`
- `server/src/qwen35/qwen35_backend.h`
- `server/src/qwen35/qwen35_backend.cpp`
- `server/src/common/backend_args.h`
- `server/src/common/backend_factory.h`
- `server/src/common/backend_factory.cpp`
- `server/src/common/backend_plan.cpp`
- `server/src/server/server_main.cpp`

New CLI flags added: `--yarn-factor <F>`, `--yarn-orig-ctx <N>`,
`--yarn-beta-fast <F>`, `--yarn-beta-slow <F>`. Fully backward compatible —
omitting `--yarn-factor` leaves behavior byte-for-byte unchanged (verified).
