#!/usr/bin/env python3
"""OpenCode exit/resume stress probe.

The OpenCode-side counterpart to codex_tui_probe.py's exit/resume cycling
(that script is Codex-CLI-specific -- Kitty keyboard protocol, `codex resume
--last` -- and doesn't apply here; OpenCode has its own `--continue`/
`--session` resume mechanism, tested via `opencode run`, a real subprocess
exit and restart on every cycle, not just a REPL loop).

Two things this checks, both real regressions that a naive "does it start"
smoke test would miss:

1. **N-cycle context continuity**: each cycle exits the process entirely and
   starts a brand new `opencode run --continue` process. A session-file bug,
   a KV-cache-vs-CLI-session mismatch, or a config regression that only
   shows up after several restarts would be invisible in a single exchange.
   Verified by planting a fact early and asking for it back several cycles
   later, plus growing the conversation each cycle so a later cycle's
   context is meaningfully larger than the first.

2. **Interrupt-mid-task recovery**: starts a real multi-step file-writing
   task, SIGTERMs the process partway through (simulating a real "closed
   the terminal mid-task" user action), then resumes with `--continue` and
   asks it to finish -- checking the actual file on disk afterward, not just
   that the CLI printed something plausible.

Usage:
    python3 opencode_resume_probe.py                    # default: 5 cycles
    python3 opencode_resume_probe.py --cycles 10
    python3 opencode_resume_probe.py --workdir /tmp/x

Exits non-zero if any cycle fails to recall the planted fact, if the
interrupt-recovery task's file doesn't end up correct on disk, or if any
`opencode run` invocation crashes / times out.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid

FAILURE_SIGNATURES = [
    "Error:",
    "FATAL",
    "panicked",
    "ECONNREFUSED",
    "Cannot find package",
]


def run_opencode(args, workdir, timeout=120):
    """Run `opencode run <args>` to completion as a fresh process. Returns
    (returncode, stdout_text) -- returncode None on timeout (process killed)."""
    cmd = ["opencode", "run"] + args
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                            timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or "") + (e.stderr or "")


def start_opencode_bg(args, workdir):
    """Start `opencode run <args>` as a backgrounded, killable process."""
    cmd = ["opencode", "run"] + args
    return subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)


def check_failures(text):
    return [sig for sig in FAILURE_SIGNATURES if sig in text]


def phase_continuity(workdir, cycles):
    print(f"\n=== Phase 1: {cycles}-cycle exit/resume context continuity ===")
    secret = str(uuid.uuid4())[:8]
    rc, out = run_opencode(
        [f"Remember this token: {secret}. Just acknowledge it in one short sentence, "
         f"don't repeat the token back yet."], workdir)
    if rc != 0 or check_failures(out):
        print(f"FAIL: initial message failed (rc={rc}): {out[-500:]}")
        return False
    print(f"OK: planted token {secret}, initial exchange clean")

    all_ok = True
    for i in range(cycles):
        # Every cycle both re-verifies recall of the ORIGINAL token (proves
        # the session file / resume path survived N restarts, not just 1)
        # and grows the conversation, so later cycles carry meaningfully
        # more accumulated context than the first.
        filler_ask = (f"Cycle {i}: briefly name one use case for a coding "
                       f"agent (a new, different one each time).")
        rc, out = run_opencode(["--continue", filler_ask], workdir)
        failures = check_failures(out)
        if rc != 0 or failures:
            print(f"FAIL: cycle {i} filler exchange failed (rc={rc}, "
                  f"failures={failures}): {out[-500:]}")
            all_ok = False
            continue

        rc, out = run_opencode(["--continue", "What was the token I told you "
                                 "to remember? Reply with just the token."], workdir)
        failures = check_failures(out)
        recalled = secret in out
        status = "OK" if (rc == 0 and recalled and not failures) else "FAIL"
        print(f"{status}: cycle {i}: rc={rc} recalled={recalled} "
              f"failures={failures}")
        if status == "FAIL":
            print(f"       reply was: {out.strip()[-200:]!r}")
            all_ok = False
    return all_ok


def phase_interrupt_recovery(workdir):
    print("\n=== Phase 2: interrupt-mid-task recovery ===")
    # Must live under workdir: opencode's permission sandbox auto-rejects
    # file writes outside the directory it was launched in ("external_directory"),
    # which is correct product behavior, not something to test around.
    target_dir = tempfile.mkdtemp(prefix="opencode_resume_probe_", dir=workdir)
    marker = str(uuid.uuid4())[:8]
    target_file = os.path.join(target_dir, "probe_output.txt")

    task = (f"Create a file at {target_file} containing exactly one line: "
            f"MARKER_{marker}. Use your file-write tool, then stop.")
    print(f"Starting task, will interrupt mid-flight: write {target_file}")
    # --auto: non-interactive `run` has no TTY to approve a permission
    # prompt, so file-write tools auto-reject by default with no --auto.
    # Fine for this controlled local probe; never do this against a real
    # untrusted task.
    proc = start_opencode_bg(["--auto", task], workdir)

    # Give it a moment to actually start working (past connection/prefill),
    # then interrupt it like a user closing the terminal mid-task -- before
    # it's had time to finish, not after.
    time.sleep(3)
    still_running = proc.poll() is None
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"OK: task interrupted after 3s (was still running: {still_running})")

    # Resume and ask it to verify/finish the job.
    rc, out = run_opencode(
        ["--continue", "--auto", f"Did you finish creating {target_file}? "
         f"If not, finish it now with exactly the content requested."],
        workdir, timeout=120)
    failures = check_failures(out)
    if rc != 0 or failures:
        print(f"FAIL: resume-and-finish exchange failed (rc={rc}, "
              f"failures={failures}): {out[-500:]}")
        return False

    if not os.path.exists(target_file):
        print(f"FAIL: {target_file} does not exist after resume+finish")
        return False
    content = open(target_file).read().strip()
    ok = content == f"MARKER_{marker}"
    print(f"{'OK' if ok else 'FAIL'}: file on disk after resume: {content!r} "
          f"(expected MARKER_{marker!r})")
    subprocess.run(["rm", "-rf", target_dir])
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--workdir", default="/workspace/python_song")
    ap.add_argument("--skip-interrupt", action="store_true",
                     help="skip phase 2 (interrupt-recovery)")
    args = ap.parse_args()

    ok1 = phase_continuity(args.workdir, args.cycles)
    ok2 = True if args.skip_interrupt else phase_interrupt_recovery(args.workdir)

    print("\n=== SUMMARY ===")
    print(f"continuity ({args.cycles} cycles): {'PASS' if ok1 else 'FAIL'}")
    if not args.skip_interrupt:
        print(f"interrupt-recovery: {'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
