This repository contains modifications to source files from
[Luce-Org/lucebox](https://github.com/Luce-Org/lucebox), which is licensed
under the Apache License, Version 2.0 (see `LICENSE`).

Modified files, staged under `lucebox-patch/` at the same relative path they
occupy in the upstream project:

- `server/src/server/tool_parser.cpp`
- `server/src/server/tool_parser.h`
- `server/src/server/sse_emitter.cpp`
- `server/src/server/sse_emitter.h`
- `server/src/server/http_server.cpp`

To apply: drop these files into the same relative paths in a clone of
`Luce-Org/lucebox`, overwriting the originals, then rebuild per that
project's own instructions.

Everything else in this repository (`codex/`, `supervisor/`, `README.md`,
`HANDOVER.md`) is original configuration and documentation, not derived from
lucebox.
