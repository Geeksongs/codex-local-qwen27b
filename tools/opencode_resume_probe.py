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

3. **Real coding task, interrupted twice, with live output checked at each
   step**: writes an implementation + a test file, gets interrupted after
   the files exist but before tests run, resumes and gets interrupted again
   right as tests start, then resumes once more to let it finish. Prints the
   streamed output at each checkpoint (this is what proves incremental
   output is actually flowing, not just buffered to one final dump -- a
   real regression HANDOVER.md flagged before). Final correctness is
   verified two independent ways that don't trust the model's own "tests
   pass" claim or even assume its test file is unittest-shaped: running its
   test file as a plain script, and a hand-written ground-truth exercise of
   the LRUCache class from this script itself.

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


def run_opencode(args, workdir, timeout=180):
    """Run `opencode run <args>` to completion as a fresh process. Returns
    (returncode, stdout_text) -- returncode None on timeout (process killed)."""
    cmd = ["opencode", "run"] + args
    def _text(x):
        if x is None:
            return ""
        return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                            timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        # On a timeout, partial stdout/stderr can come back as bytes even
        # with text=True (the process was killed mid-decode) -- normalize
        # before concatenating, or this raises its own TypeError and hides
        # the real timeout as a confusing crash instead.
        return None, _text(e.stdout) + _text(e.stderr)


def start_opencode_bg(args, workdir):
    """Start `opencode run <args>` as a backgrounded, killable process."""
    cmd = ["opencode", "run"] + args
    return subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)


def start_opencode_bg_logged(args, workdir, log_path):
    """Like start_opencode_bg, but streams to a file on disk instead of a
    pipe, so a caller can tail it live while the process is still running
    (a plain PIPE only yields data to the parent on read, which doesn't let
    us print "here's what it's doing right now" mid-flight the way a real
    terminal would show)."""
    cmd = ["opencode", "run"] + args
    log_f = open(log_path, "wb")
    return subprocess.Popen(cmd, cwd=workdir, stdout=log_f, stderr=subprocess.STDOUT,
                             start_new_session=True), log_f


def tail(log_path):
    try:
        return open(log_path, "r", errors="replace").read()
    except FileNotFoundError:
        return ""


def kill_proc(proc):
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
        workdir, timeout=180)
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


def phase_coding_task(workdir):
    print("\n=== Phase 3: real coding task, interrupted twice, live output ===")
    task_dir = tempfile.mkdtemp(prefix="opencode_coding_probe_", dir=workdir)
    impl = os.path.join(task_dir, "lru_cache.py")
    test = os.path.join(task_dir, "test_lru_cache.py")
    logdir = tempfile.mkdtemp(prefix="opencode_coding_probe_logs_")

    def rel(p):
        return os.path.relpath(p, workdir)

    task = (f"In {task_dir}/, create lru_cache.py implementing an LRUCache "
            f"class (constructor takes capacity, methods get(key) and "
            f"put(key,value), evicts least-recently-used on overflow). Then "
            f"create test_lru_cache.py with at least 4 test cases covering "
            f"basic get/put, eviction order, updating an existing key, and "
            f"capacity=1. Run the tests with python3 and make sure they all "
            f"pass, fixing any bugs you find.")

    log1 = os.path.join(logdir, "step1.log")
    proc, f1 = start_opencode_bg_logged(["--auto", task], workdir, log1)
    time.sleep(7)  # long enough to see real tool calls, short enough to interrupt pre-test
    print(f"--- live output after 7s (should show tool calls in progress) ---")
    print(tail(log1).strip()[-800:])
    kill_proc(proc)
    f1.close()
    files_after_1 = {rel(impl): os.path.exists(impl), rel(test): os.path.exists(test)}
    print(f"OK: interrupted #1 at 7s -- files present: {files_after_1}")

    log2 = os.path.join(logdir, "step2.log")
    proc, f2 = start_opencode_bg_logged(
        ["--continue", "--auto", "Continue the task: run the tests and fix "
         "any failures until they all pass."], workdir, log2)
    time.sleep(6)
    print(f"--- live output after resume #1, 6s in (should show test run / debugging) ---")
    print(tail(log2).strip()[-800:])
    kill_proc(proc)
    f2.close()
    print("OK: interrupted #2 mid-test-and-debug")

    log3 = os.path.join(logdir, "step3.log")
    rc, out = run_opencode(["--continue", "--auto", "Confirm the task is complete."],
                            workdir, timeout=180)
    failures = check_failures(out)
    print(f"--- final resume output ---")
    print(out.strip()[-500:])
    if rc != 0 or failures:
        print(f"FAIL: final resume failed (rc={rc}, failures={failures})")
        return False

    if not (os.path.exists(impl) and os.path.exists(test)):
        print(f"FAIL: expected files missing after final resume: "
              f"impl={os.path.exists(impl)} test={os.path.exists(test)}")
        return False

    # Independent verification, two layers -- don't trust the model's own
    # "tests pass" claim, and don't even trust its test file is well-formed:
    #
    # 1. Run its test file as a plain script (works whether it wrote
    #    unittest.TestCase classes or a bare assert-based __main__ block --
    #    `-m unittest test_lru_cache` silently reports "Ran 0 tests" / rc=0
    #    on the latter style, a false pass this caught in practice: models
    #    don't consistently pick one test style across runs).
    script_verify = subprocess.run(["python3", "test_lru_cache.py"],
                                    cwd=task_dir, capture_output=True, text=True,
                                    timeout=30)
    script_out = script_verify.stdout + script_verify.stderr
    script_ok = script_verify.returncode == 0 and not check_failures(script_out)
    print(f"{'OK' if script_ok else 'FAIL'}: test file run as a script "
          f"(rc={script_verify.returncode}): {script_out.strip()[-300:]}")

    # 2. A hand-written, model-independent black-box check of the actual
    #    LRUCache class -- doesn't rely on the model's own test file being
    #    correct or even present, only on lru_cache.py exporting the class.
    ground_truth = (
        "import sys; sys.path.insert(0, %r)\n"
        "from lru_cache import LRUCache\n"
        "c = LRUCache(2)\n"
        "c.put('a', 1); c.put('b', 2)\n"
        "assert c.get('a') == 1, 'basic get failed'\n"
        "c.put('c', 3)  # should evict 'b' (a was just touched by get)\n"
        "assert c.get('b') == -1, 'eviction picked the wrong key'\n"
        "assert c.get('c') == 3, 'newly inserted key missing'\n"
        "c.put('a', 99)  # update existing key\n"
        "assert c.get('a') == 99, 'update of existing key did not stick'\n"
        "print('GROUND_TRUTH_OK')\n"
    ) % task_dir
    gt_verify = subprocess.run(["python3", "-c", ground_truth],
                                capture_output=True, text=True, timeout=15)
    gt_ok = gt_verify.returncode == 0 and "GROUND_TRUTH_OK" in gt_verify.stdout
    print(f"{'OK' if gt_ok else 'FAIL'}: independent ground-truth behavior check: "
          f"{(gt_verify.stdout + gt_verify.stderr).strip()[-300:]}")

    tests_ok = script_ok and gt_ok

    subprocess.run(["rm", "-rf", task_dir, logdir])
    return tests_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--workdir", default="/workspace/python_song")
    ap.add_argument("--skip-interrupt", action="store_true",
                     help="skip phase 2 (interrupt-recovery)")
    ap.add_argument("--skip-coding", action="store_true",
                     help="skip phase 3 (real coding task)")
    args = ap.parse_args()

    ok1 = phase_continuity(args.workdir, args.cycles)
    ok2 = True if args.skip_interrupt else phase_interrupt_recovery(args.workdir)
    ok3 = True if args.skip_coding else phase_coding_task(args.workdir)

    print("\n=== SUMMARY ===")
    print(f"continuity ({args.cycles} cycles): {'PASS' if ok1 else 'FAIL'}")
    if not args.skip_interrupt:
        print(f"interrupt-recovery: {'PASS' if ok2 else 'FAIL'}")
    if not args.skip_coding:
        print(f"coding-task (2 interrupts): {'PASS' if ok3 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)


if __name__ == "__main__":
    main()
