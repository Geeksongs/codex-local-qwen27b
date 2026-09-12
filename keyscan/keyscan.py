#!/usr/bin/env python3
"""
keyscan — 在 GitHub 的 LLM agent 项目里挖 LLM API key 并验证有效性。

流程:
  1. 用 GitHub Search API 批量搜 "llm agent" 相关仓库 (可离线: --repos 直接给列表)
  2. 对每个仓库做 shallow clone (带历史), 用 git log -p 扫全部 commit 的 diff
  3. 用一组正则识别各家 LLM 的 key (OpenAI/Anthropic/Gemini/Mistral/DeepSeek/...)
  4. 对每个 key 调对应 provider 的 /models 端点验证有效性
  5. 结果写 JSON + 打印表格

用法:
  python3 keyscan.py --max-repos 50 --max-commits 2000 --workers 8
  python3 keyscan.py --repos owner/repo,owner/repo2   # 离线, 跳过搜索
  python3 keyscan.py --token $GITHUB_TOKEN            # 提高 API 限额
"""
import argparse, json, os, re, subprocess, sys, time, shutil, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

GH_API = "https://api.github.com"

# ---------------------------------------------------------------------------
# 1. 各家 LLM key 的正则 (尽量精确, 降低误报)
# ---------------------------------------------------------------------------
KEY_PATTERNS = [
    # (provider, 正则, 说明)
    ("openai",    r"\bsk-[A-Za-z0-9_\-]{20,}\b", "OpenAI (sk-...)"),
    ("openai",    r"\bsk-proj-[A-Za-z0-9_\-]{20,}\b", "OpenAI project key"),
    ("openai",    r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b", "OpenAI (legacy ant)"),
    ("anthropic", r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b", "Anthropic (sk-ant-...)"),
    ("gemini",    r"\bAIza[A-Za-z0-9_\-]{35}\b", "Google Gemini (AIza...)"),
    ("mistral",   r"\bkey-[A-Za-z0-9]{32,}\b", "Mistral (key-...)"),
    ("deepseek",  r"\bsk-[A-Za-z0-9]{32,}\b", "DeepSeek (sk-...)"),
    ("groq",      r"\bgsk_[A-Za-z0-9]{20,}\b", "Groq (gsk_...)"),
    ("together",  r"\b[A-Za-z0-9]{20,}\.[A-Za-z0-9]{20,}\b", "Together (heuristic)"),
    ("huggingface", r"\bhf_[A-Za-z0-9]{30,}\b", "HuggingFace (hf_...)"),
    ("cohere",    r"\b[A-Za-z0-9]{8}-[A-Za-z0-9]{8}-[A-Za-z0-9]{8}-[A-Za-z0-9]{8}\b", "Cohere (uuid-like)"),
    ("replicate", r"\br8_[A-Za-z0-9]{20,}\b", "Replicate (r8_...)"),
    ("perplexity",r"\bpplx-[A-Za-z0-9_\-]{20,}\b", "Perplexity (pplx-...)"),
    ("ollama",    r"\bollama_[A-Za-z0-9]{20,}\b", "Ollama (ollama_...)"),
    ("fireworks", r"\b[A-Za-z0-9]{20,}\.[A-Za-z0-9]{20,}\.[A-Za-z0-9]{20,}\b", "Fireworks (heuristic)"),
    ("openrouter",r"\bsk-or-[A-Za-z0-9_\-]{20,}\b", "OpenRouter (sk-or-...)"),
    ("zhipu",     r"\b[A-Za-z0-9]{16}\.[A-Za-z0-9]{16}\b", "Zhipu/GLM (heuristic)"),
    ("moonshot",  r"\bmk-[A-Za-z0-9]{20,}\b", "Moonshot (mk-...)"),
    ("minimax",   r"\beyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b", "MiniMax (jwt)"),
]
# 编译一次
COMPILED = [(prov, re.compile(pat), desc) for prov, pat, desc in KEY_PATTERNS]

# 明显是占位符/示例的 key, 直接跳过
PLACEHOLDER = re.compile(
    r"(your[_-]?key|xxx+|<.*>|example|placeholder|changeme|dummy|fake|test[_-]?key|"
    r"abc+|123+|sk-[a-z]{4,}|sk-or-[a-z]{4,}|sk-ant-[a-z]{4,}|AIzaSyA[0-9]{2})", re.I)

# ---------------------------------------------------------------------------
# 2. GitHub 搜索
# ---------------------------------------------------------------------------
def gh_get(url, token=None, retries=3):
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "keyscan/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for i in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as r:
                return json.load(r)
        except HTTPError as e:
            if e.code in (403, 429) and i < retries - 1:
                # rate limit — 读 reset
                reset = e.headers.get("X-RateLimit-Reset")
                wait = 2 ** i * 2
                if reset:
                    try:
                        wait = max(0, int(reset) - time.time()) + 1
                    except Exception:
                        pass
                print(f"  [gh] rate-limited, sleep {wait}s", file=sys.stderr)
                time.sleep(min(wait, 60))
                continue
            if e.code == 404:
                return None
            raise
        except URLError as e:
            if i < retries - 1:
                time.sleep(2 ** i)
                continue
            raise
    return None

def search_repos(token, queries, per_query=30, max_repos=100):
    """返回去重后的 [owner/repo, ...]"""
    seen, out = set(), []
    for q in queries:
        for page in range(1, 4):  # 最多 3 页/查询
            url = (f"{GH_API}/search/repositories?q={q.replace(' ', '+')}"
                   f"&per_page={per_query}&page={page}&sort=updated")
            data = gh_get(url, token)
            if not data or not data.get("items"):
                break
            for it in data["items"]:
                full = it["full_name"]
                if full not in seen:
                    seen.add(full)
                    out.append(full)
            if len(data["items"]) < per_query:
                break
            if len(out) >= max_repos:
                break
        if len(out) >= max_repos:
            break
    return out[:max_repos]

# ---------------------------------------------------------------------------
# 3. clone + 扫 commit 历史
# ---------------------------------------------------------------------------
def clone_repo(full, workdir, depth=None):
    url = f"https://github.com/{full}.git"
    dest = os.path.join(workdir, full.replace("/", "__"))
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    cmd = ["git", "clone", "--quiet", "--filter=blob:none"]
    if depth:
        cmd += ["--shallow-since", f"{depth} days"]
    cmd += [url, dest]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        # 退路: 普通 shallow clone
        cmd2 = ["git", "clone", "--quiet", "--depth", "500", url, dest]
        r = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None, r.stderr.strip()[-200:]
    return dest, None

def scan_repo(dest, max_commits=2000):
    """git log -p 扫全部 diff, 返回 [(key, provider, commit, file, line)]"""
    found = []
    # 用 -p 输出所有改动行; 限制 commit 数
    cmd = ["git", "-C", dest, "log", "-p", "-U0", "--no-color",
           f"-n{max_commits}", "--diff-filter=ACMR"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        return found
    out = r.stdout
    cur_commit = "?"
    cur_file = "?"
    line_no = 0
    for line in out.splitlines():
        if line.startswith("commit "):
            cur_commit = line.split()[1][:12]
            cur_file = "?"
            line_no = 0
        elif line.startswith("diff --git"):
            m = re.search(r" b/(\S+)", line)
            cur_file = m.group(1) if m else "?"
            line_no = 0
        elif line.startswith("+") and not line.startswith("+++"):
            line_no += 1
            content = line[1:]
            for prov, rx, desc in COMPILED:
                for m in rx.finditer(content):
                    key = m.group(0)
                    if PLACEHOLDER.search(key):
                        continue
                    found.append({"key": key, "provider": prov,
                                  "commit": cur_commit, "file": cur_file,
                                  "line": line_no})
    return found

# ---------------------------------------------------------------------------
# 4. 验证 key 有效性 (调 /models)
# ---------------------------------------------------------------------------
VALIDATORS = {
    "openai":     ("https://api.openai.com/v1/models", "Authorization", "Bearer {}"),
    "anthropic":  ("https://api.anthropic.com/v1/models", "x-api-key", "{}"),
    "gemini":     ("https://generativelanguage.googleapis.com/v1beta/models?key={}", None, None),
    "mistral":    ("https://api.mistral.ai/v1/models", "Authorization", "Bearer {}"),
    "deepseek":   ("https://api.deepseek.com/v1/models", "Authorization", "Bearer {}"),
    "groq":       ("https://api.groq.com/openai/v1/models", "Authorization", "Bearer {}"),
    "huggingface":("https://huggingface.co/api/models", "Authorization", "Bearer {}"),
    "cohere":     ("https://api.cohere.com/v1/models", "Authorization", "Bearer {}"),
    "replicate":  ("https://api.replicate.com/v1/models", "Authorization", "Bearer {}"),
    "perplexity": ("https://api.perplexity.ai/v1/models", "Authorization", "Bearer {}"),
    "openrouter": ("https://openrouter.ai/api/v1/models", "Authorization", "Bearer {}"),
    "ollama":     ("http://localhost:11434/api/tags", None, None),
}

def validate_key(provider, key, timeout=15):
    """返回 (status, detail): status in valid/invalid/unknown/error"""
    v = VALIDATORS.get(provider)
    if not v:
        return "unknown", f"no validator for {provider}"
    url, hdr, tmpl = v
    if hdr is None:
        url = url.format(key) if "{}" in url else url
        headers = {}
    else:
        url = url
        headers = {hdr: tmpl.format(key)}
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as r:
            code = r.status
            if 200 <= code < 300:
                return "valid", f"HTTP {code}"
            return "invalid", f"HTTP {code}"
    except HTTPError as e:
        if e.code in (401, 403):
            return "invalid", f"HTTP {e.code} (auth failed)"
        if e.code == 404:
            return "unknown", f"HTTP 404 (endpoint/model list missing)"
        return "invalid", f"HTTP {e.code}"
    except (URLError, TimeoutError, Exception) as e:
        return "error", str(e)[:80]

# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--repos", default=None, help="owner/repo,owner/repo2 (离线)")
    ap.add_argument("--max-repos", type=int, default=50)
    ap.add_argument("--max-commits", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--workdir", default=os.path.join(os.path.dirname(__file__), "repos"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--days", type=int, default=0, help="shallow-since N days (0=full)")
    ap.add_argument("--queries", default="llm agent,ai agent,langchain agent,autogen agent,crewai agent,openai agent")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    t0 = time.time()

    # 1. 仓库列表
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()][:args.max_repos]
        print(f"[1/4] 离线仓库列表: {len(repos)} 个", file=sys.stderr)
    else:
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        print(f"[1/4] 搜索 GitHub: {len(queries)} 个查询...", file=sys.stderr)
        repos = search_repos(args.token, queries, max_repos=args.max_repos)
        print(f"      得到 {len(repos)} 个仓库", file=sys.stderr)

    # 2+3. clone + 扫描 (并行)
    all_hits = []
    def work(full):
        dest, err = clone_repo(full, args.workdir, depth=args.days or None)
        if not dest:
            return full, [], err
        hits = scan_repo(dest, args.max_commits)
        return full, hits, None

    print(f"[2/4] clone + 扫描 {len(repos)} 个仓库 (workers={args.workers})...", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for r in repos}
        for f in as_completed(futs):
            full, hits, err = f.result()
            done += 1
            if err:
                print(f"  ({done}/{len(repos)}) {full}: CLONE FAIL {err[:60]}", file=sys.stderr)
            else:
                print(f"  ({done}/{len(repos)}) {full}: {len(hits)} 个候选 key", file=sys.stderr)
            all_hits.extend([dict(h, repo=full) for h in hits])

    # 去重 (同一 key 只验证一次)
    uniq = {}
    for h in all_hits:
        uniq.setdefault(h["key"], h)
    print(f"[3/4] 去重后 {len(uniq)} 个唯一 key", file=sys.stderr)

    # 4. 验证
    if not args.no_validate:
        print(f"[4/4] 验证 key 有效性...", file=sys.stderr)
        for i, (key, h) in enumerate(uniq.items(), 1):
            status, detail = validate_key(h["provider"], key)
            h["status"] = status
            h["detail"] = detail
            print(f"  ({i}/{len(uniq)}) [{status:7}] {h['provider']:10} {key[:18]}... @ {h['repo']}", file=sys.stderr)
    else:
        for h in uniq.values():
            h["status"] = "skipped"
            h["detail"] = "no-validate"

    # 输出
    results = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "repos_scanned": len(repos),
               "total_candidates": len(all_hits),
               "unique_keys": len(uniq),
               "valid": sum(1 for h in uniq.values() if h.get("status") == "valid"),
               "keys": list(uniq.values())}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== 完成, 耗时 {time.time()-t0:.0f}s ===", file=sys.stderr)
    print(f"扫描仓库: {results['repos_scanned']}  候选: {results['total_candidates']}  "
          f"唯一: {results['unique_keys']}  有效: {results['valid']}", file=sys.stderr)
    print(f"结果: {args.out}", file=sys.stderr)

    # 终端表格
    print(f"\n{'STATUS':8} {'PROVIDER':11} {'KEY':24} {'REPO':30} COMMIT")
    for h in sorted(uniq.values(), key=lambda x: (x.get("status")!="valid", x["provider"])):
        print(f"{h.get('status','?'):8} {h['provider']:11} {h['key'][:22]:24} "
              f"{h['repo'][:28]:30} {h['commit']}")

if __name__ == "__main__":
    main()
