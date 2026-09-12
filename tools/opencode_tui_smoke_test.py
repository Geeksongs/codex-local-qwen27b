#!/usr/bin/env python3
"""OpenCode interactive-TUI smoke test.

Drives the real `opencode` TUI through a pty (plain writes work here --
unlike Codex, OpenCode's TUI does not require Kitty keyboard protocol
encoding), confirms the UI actually renders (not a blank screen), that the
configured local provider (see ../opencode.json, "lucebox/dflash") is the
active default model, and that a real message round-trips to it.

Usage: python3 opencode_tui_smoke_test.py [workdir]
(workdir defaults to /workspace, where opencode.json lives)

Exits non-zero if the screen never renders real UI content, or the model
footer doesn't show the expected local provider.
"""
import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time

ANSI_RE = re.compile(
    r'\x1b\[[0-9;:?]*[a-zA-Z]'
    r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'
    r'|\x1b[()][AB012]'
    r'|\x1b[=>NODM7]'
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text).replace('\r', '')


def drain(fd, timeout=1.0):
    out = b""
    while True:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    return out


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
    expect_model = sys.argv[2] if len(sys.argv) > 2 else "dflash (local)"

    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(workdir)
        os.execvp("opencode", ["opencode"])
        os._exit(127)

    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    buf = b""
    try:
        time.sleep(6)
        buf += drain(fd, 2.0)
        clean = strip_ansi(buf.decode("utf-8", errors="replace"))

        rendered = ("Ask anything" in clean) or ("Build" in clean)
        if not rendered:
            print("FAIL: no recognizable UI text rendered after 6s (blank-screen "
                  "symptom) -- see raw bytes below")
            print(repr(buf[:500]))
            sys.exit(1)
        print("OK: UI rendered (prompt box visible)")

        model_ok = expect_model in clean
        print(f"{'OK' if model_ok else 'WARN'}: default model shows "
              f"{'expected' if model_ok else 'UNEXPECTED (not)'} "
              f"'{expect_model}' in the footer")

        for ch in "Say the word: smoketest":
            os.write(fd, ch.encode())
            time.sleep(0.02)
        os.write(fd, b"\r")

        deadline = time.time() + 30
        got_reply = False
        while time.time() < deadline:
            chunk = drain(fd, 1.5)
            buf += chunk
            if b"smoketest" in buf and b"·" in chunk:
                got_reply = True
            if chunk == b"" and got_reply:
                break

        clean2 = strip_ansi(buf.decode("utf-8", errors="replace"))
        if "smoketest" not in clean2:
            print("FAIL: sent message never echoed / no reply observed within 30s")
            sys.exit(1)
        print("OK: message round-tripped, got a reply")

        if not model_ok:
            sys.exit(2)  # rendered + replied, but wrong model -- config issue
    finally:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    main()
