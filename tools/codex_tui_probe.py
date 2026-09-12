#!/usr/bin/env python3
"""Persistent Codex TUI regression probe.

Drives the REAL interactive `codex` TUI through a pty (Kitty keyboard
protocol encoding -- plain ASCII writes are silently ignored once Codex's
TUI switches into Kitty keyboard mode), across a battery of scenarios and
multiple exit/resume cycles, and scans the transcript for failure
signatures (tool-call errors, missing-field errors, stalls with no token
output, crashes).

Usage:
    python3 codex_tui_probe.py                 # run the full battery once
    python3 codex_tui_probe.py --cycles 5      # 5 exit/resume cycles
    python3 codex_tui_probe.py --workdir /tmp/x

Each scenario's raw + ANSI-stripped transcript is saved under
--logdir (default /tmp/codex_tui_probe_logs/<timestamp>/), and a summary
is printed (and saved as summary.json) at the end: pass/fail per
scenario/cycle, wall-clock time, and any matched failure signatures.

This is meant to be reused any time someone reports "Codex isn't working"
without more detail -- rerun this first before speculative debugging.
"""
import argparse
import json
import os
import pty
import re
import select
import signal
import sys
import time
import uuid

ANSI_RE = re.compile(
    r'\x1b\[[0-9;:?]*[a-zA-Z]'      # CSI sequences
    r'|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)'  # OSC sequences
    r'|\x1b[()][AB012]'
    r'|\x1b[=>NODM7]'
)

FAILURE_SIGNATURES = [
    "missing field",
    "Fatal error",
    "invoked with incompatible payload",
    "panicked at",
    "unknown variant",
    "connection refused",
    "ok=false",
    "out=0",
    "internal error",
    "Error:",
]

STALL_SECONDS = 45  # no new bytes for this long while "working" == a stall


def strip_ansi(text: str) -> str:
    clean = ANSI_RE.sub('', text)
    return clean.replace('\r', '')


def kitty_key(ch: str) -> bytes:
    return f"\x1b[{ord(ch)}u".encode()


class CodexTUI:
    """One live pty-backed `codex` (or `codex resume --last`) process."""

    def __init__(self, args, cwd, logfile):
        self.args = args
        self.cwd = cwd
        self.logfile = logfile
        self.pid = None
        self.fd = None
        self.buf = b""

    def start(self):
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(self.cwd)
            os.execvp("codex", ["codex"] + self.args)
            os._exit(127)
        self.pid = pid
        self.fd = fd
        time.sleep(2.5)
        self._drain(timeout=2.0)

    def _drain(self, timeout=1.0):
        out = b""
        while True:
            r, _, _ = select.select([self.fd], [], [], timeout)
            if not r:
                break
            try:
                chunk = os.read(self.fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        if out:
            self.buf += out
            with open(self.logfile, "ab") as f:
                f.write(out)
        return out

    def send_text(self, text, delay=0.02):
        for ch in text:
            if ch == "\n":
                os.write(self.fd, b"\r")
            else:
                os.write(self.fd, kitty_key(ch))
            time.sleep(delay)

    def send_ctrl(self, letter, delay=0.02):
        # Ctrl+<letter> as Kitty-encoded (codepoint + ctrl modifier=5)
        code = ord(letter.upper()) - 64
        os.write(self.fd, f"\x1b[{code};5u".encode() if False else bytes([code]))
        time.sleep(delay)

    def wait_for_idle(self, total_timeout=120, quiet_for=3.0):
        """Poll until the pty has been quiet for `quiet_for` seconds, or
        total_timeout elapses. Returns (timed_out: bool, last_activity_gap: float)."""
        deadline = time.time() + total_timeout
        last_data_t = time.time()
        while time.time() < deadline:
            chunk = self._drain(timeout=1.0)
            now = time.time()
            if chunk:
                last_data_t = now
            elif now - last_data_t >= quiet_for:
                return False, now - last_data_t
        return True, time.time() - last_data_t

    def kill(self):
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except Exception:
                pass

    def transcript(self):
        return strip_ansi(self.buf.decode("utf-8", errors="replace"))


SCENARIOS = [
    ("single_shell", "Run the shell command 'echo scenario_single_shell_ok' and tell me the output."),
    ("parallel_shell", "Run 'pwd' and 'date' as two separate shell commands and report both outputs."),
    ("file_write_apply_patch", "Create a file named probe_note.txt containing exactly the line: probe ok. Use apply_patch."),
    ("file_read_back", "Read back probe_note.txt and tell me its exact contents."),
    ("write_code_fib", "Create a file fib.py with a function fib(n) that returns the nth Fibonacci number "
                       "(0-indexed, fib(0)=0, fib(1)=1) using apply_patch, then run "
                       "'python3 -c \"import fib; print(fib.fib(10))\"' and tell me the output."),
    ("edit_existing_code", "Open fib.py, add a second function fib_iter(n) that computes the same value "
                            "iteratively (no recursion), using apply_patch to edit the existing file. "
                            "Then run 'python3 -c \"import fib; print(fib.fib_iter(10))\"' and report the output."),
    ("read_and_summarize_code", "Read the contents of fib.py with your file tools and tell me exactly how "
                                 "many functions are defined in it and their names."),
    ("long_running_cmd", "Run the shell command 'sleep 3 && echo slept_ok' and tell me the output."),
    ("arithmetic_no_tool", "What is 123456 + 654321? Reply with just the number, no tools needed."),
    ("multi_step", "First run 'echo step1', then run 'echo step2', then tell me you ran both in order."),
]


def run_scenario(tui, name, prompt, logdir, timeout=90):
    t0 = time.time()
    before_len = len(tui.buf)
    tui.send_text(prompt + "\n")
    timed_out, quiet_gap = tui.wait_for_idle(total_timeout=timeout, quiet_for=4.0)
    dt = time.time() - t0
    segment = tui.buf[before_len:].decode("utf-8", errors="replace")
    clean = strip_ansi(segment)
    hits = [sig for sig in FAILURE_SIGNATURES if sig.lower() in clean.lower()]
    stalled = timed_out and dt >= STALL_SECONDS
    result = {
        "scenario": name,
        "prompt": prompt,
        "duration_s": round(dt, 2),
        "timed_out": timed_out,
        "stalled": stalled,
        "failure_signatures": hits,
        "ok": (not hits) and (not stalled),
    }
    seg_path = os.path.join(logdir, f"{name}.txt")
    with open(seg_path, "w") as f:
        f.write(clean)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3, help="number of fresh-session + exit/resume cycles")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--scenario-timeout", type=int, default=90)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    workdir = args.workdir or f"/tmp/codex_tui_probe_{run_id}"
    logdir_root = args.logdir or f"/tmp/codex_tui_probe_logs/{run_id}"
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(logdir_root, exist_ok=True)

    # Ensure this scratch project is trusted so the trust-prompt doesn't
    # eat the first real keystrokes (a known false-positive-failure cause).
    config_path = os.path.expanduser("~/.codex/config.toml")
    marker = f'[projects."{workdir}"]'
    with open(config_path, "r") as f:
        cfg = f.read()
    if marker not in cfg:
        with open(config_path, "a") as f:
            f.write(f'\n{marker}\ntrust_level = "trusted"\n')

    all_results = []

    def quit_tui(tui):
        tui.send_ctrl('c')
        time.sleep(0.3)
        tui.send_ctrl('c')
        time.sleep(1.5)
        tui._drain(timeout=2.0)
        tui.kill()
        time.sleep(1.0)

    # --- Phase A: interrupt-mid-task probe ---------------------------------
    # Start a real, multi-step code-writing task, forcibly exit WHILE it is
    # still working (not waiting for completion), then resume and ask it to
    # finish -- this is the exact "write code, exit partway, resume, does it
    # finish correctly" scenario the user asked to cover, repeated across
    # several exit/resume hops before final verification.
    interrupt_dir = os.path.join(logdir_root, "interrupt_probe")
    os.makedirs(interrupt_dir, exist_ok=True)
    big_task = (
        "Create a file named calc.py implementing a small calculator module: "
        "functions add(a,b), sub(a,b), mul(a,b), div(a,b) (div raises ValueError "
        "on division by zero), and a class Calculator with a method history() "
        "that returns a list of all past operations performed via its own "
        "add/sub/mul/div methods (which should each record their call and "
        "result before returning it). Use apply_patch. Take your time and "
        "make sure it's fully correct."
    )
    print("=== interrupt-mid-task probe ===", flush=True)
    tui = CodexTUI([], workdir, os.path.join(interrupt_dir, "hop0_start.log"))
    tui.start()
    t0 = time.time()
    tui.send_text(big_task + "\n")
    # Deliberately do NOT wait for idle: interrupt partway through, while it
    # should still be mid-tool-call/mid-generation.
    time.sleep(6.0)
    mid_transcript_before_kill = strip_ansi(tui.buf.decode("utf-8", errors="replace"))
    still_working = ("tokens used" not in mid_transcript_before_kill)
    quit_tui(tui)
    all_results.append({
        "scenario": "interrupt_hop0_start", "cycle": "interrupt",
        "duration_s": round(time.time() - t0, 2),
        "interrupted_while_working": still_working,
        "ok": True,  # this hop's "success" is just that we could interrupt at all
        "failure_signatures": [],
    })
    print(f"  hop0: started task, force-exited after 6s "
          f"(still_working_when_killed={still_working})", flush=True)

    n_resume_hops = max(2, args.cycles - 1)
    for hop in range(1, n_resume_hops + 1):
        hop_logfile = os.path.join(interrupt_dir, f"hop{hop}.log")
        print(f"  hop{hop}: resume --last ...", flush=True)
        tui = CodexTUI(["resume", "--last"], workdir, hop_logfile)
        tui.start()
        if hop < n_resume_hops:
            # Intermediate hops: just re-open and immediately exit again,
            # to cover "exit many times in a row" before ever finishing the
            # task -- the failure mode reported as needing several exits
            # to reproduce.
            time.sleep(3.0)
            quit_tui(tui)
            all_results.append({
                "scenario": f"interrupt_hop{hop}_bounce", "cycle": "interrupt",
                "duration_s": 3.0, "ok": True, "failure_signatures": [],
            })
            continue
        # Final hop: ask it to finish the task, then verify.
        res = run_scenario(
            tui, "interrupt_finish",
            "Continue and finish the calc.py task from before if it isn't "
            "complete yet, then run "
            "'python3 -c \"import calc; c=calc.Calculator(); c.add(2,3); "
            "c.mul(4,5); print(c.history()); print(calc.div(10,2))\"' "
            "and report the exact output.",
            interrupt_dir, timeout=args.scenario_timeout,
        )
        res["cycle"] = "interrupt"
        all_results.append(res)
        status = "OK" if res["ok"] else "FAIL"
        print(f"     {status} ({res['duration_s']}s) sigs={res['failure_signatures']}", flush=True)
        quit_tui(tui)

    # Independently verify the file this task should have produced.
    calc_path = os.path.join(workdir, "calc.py")
    file_check = {"scenario": "interrupt_file_on_disk", "cycle": "interrupt",
                  "ok": False, "failure_signatures": []}
    if os.path.exists(calc_path):
        with open(calc_path) as f:
            src = f.read()
        has_all = all(tok in src for tok in ["def add", "def sub", "def mul",
                                              "def div", "class Calculator",
                                              "def history"])
        file_check["ok"] = has_all
        file_check["duration_s"] = 0
        if not has_all:
            file_check["failure_signatures"] = ["calc.py incomplete after resume+finish"]
    else:
        file_check["duration_s"] = 0
        file_check["failure_signatures"] = ["calc.py never created"]
    all_results.append(file_check)
    print(f"  file check: {'OK' if file_check['ok'] else 'FAIL'} "
          f"({file_check['failure_signatures']})", flush=True)

    # --- Phase B: plain repeated exit/resume + scenario battery -------------
    for cycle in range(args.cycles):
        cycle_dir = os.path.join(logdir_root, f"cycle{cycle}")
        os.makedirs(cycle_dir, exist_ok=True)
        logfile = os.path.join(cycle_dir, "raw.log")

        if cycle == 0:
            tui_args = []
        else:
            tui_args = ["resume", "--last"]

        print(f"=== cycle {cycle} (args={tui_args}) ===", flush=True)
        tui = CodexTUI(tui_args, workdir, logfile)
        tui.start()

        # Only run the full scenario battery on the first cycle; later
        # cycles just verify the resumed session still works at all
        # (that's the "exit N times then resume" failure mode being
        # chased) with one representative scenario, to keep this fast.
        scenarios = SCENARIOS if cycle == 0 else SCENARIOS[:2]

        for name, prompt in scenarios:
            print(f"  -> {name} ...", flush=True)
            res = run_scenario(tui, name, prompt, cycle_dir, timeout=args.scenario_timeout)
            res["cycle"] = cycle
            all_results.append(res)
            status = "OK" if res["ok"] else "FAIL"
            print(f"     {status} ({res['duration_s']}s) sigs={res['failure_signatures']}", flush=True)

        quit_tui(tui)

    summary_path = os.path.join(logdir_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    n_fail = sum(1 for r in all_results if not r["ok"])
    print(f"\n=== SUMMARY: {len(all_results) - n_fail}/{len(all_results)} scenarios OK ===")
    for r in all_results:
        if not r["ok"]:
            print(f"  FAIL cycle={r['cycle']} scenario={r['scenario']} "
                  f"dt={r['duration_s']}s sigs={r['failure_signatures']} stalled={r['stalled']}")
    print(f"Logs: {logdir_root}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
