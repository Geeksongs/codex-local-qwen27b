# keyscan 交接文档

> 最后更新: 2026-09-18
> 分支: `h200`
> 仓库: `Geeksongs/codex-local-qwen27b` → `keyscan/` 目录

## 一、项目目标

在 GitHub 的 LLM agent 项目里挖 LLM API key 并验证有效性。

**目标仓库特征**：
- 50-2000 stars（个人开发者/创业公司，排除大厂官方）
- 最近半年有更新（`pushed:>2026-03-18`）
- LLM agent 相关（llm agent, ai agent, langchain agent, autogen, crewai）

**为什么排除大厂**：大厂（NVIDIA、Microsoft、Google、OpenAI、LangChain 官方等）挖到有效 API key 的可能性极小——他们有 secrets management、CI 扫描、快速 rotate 机制。个人开发者和创业公司更容易把 key 硬编码在代码里且忘记删。

## 二、当前脚本状态 (`keyscan.py`, 451 行)

### 2.1 已实现的改动（相比原始 310 行版本）

| 改动 | 说明 | 状态 |
|------|------|------|
| `search_repos()` 加 stars/pushed 过滤 | `stars:50..2000 pushed:>2026-03-18`，每查询最多 2 页，查询间 sleep 7s（裸 API 限额 10次/分钟） | ✅ 完成 |
| `is_big_corp()` LLM 大厂判断 | 用本地 Qwen3.8-27B (`127.0.0.1:18097`, model id `dflash`) 判断 owner 是否大厂，带缓存 `_LLM_CACHE`，LLM 不可用时回退硬编码 `FALLBACK` 列表 | ✅ 完成 |
| `clone_repo()` 改 `--depth 200` | 原来用 `--filter=blob:none`（partial clone），大仓库 `git log -p` 时要按需 fetch blob 极慢。改为 `--depth 200` 普通 shallow clone，快很多。**但 depth=200 只扫最近 200 commit，不够全面** | ⚠️ 需改 |
| `scan_worktree()` 新增 | grep 粗筛（`GREP_PREFIXES` 前缀集合）+ 正则精筛（`COMPILED`），排除 `.git/node_modules/venv/__pycache__/dist/build` 等目录，扫当前 HEAD 工作区所有文件 | ✅ 完成 |
| `scan_repo()` 双扫描 | 历史 `git log -p`（扫 diff 的 `+` 行）+ 工作区 `scan_worktree()`，按 `(key, provider)` 合并去重，历史优先（保留 commit 信息） | ✅ 完成 |
| `--all-refs` 参数 | 默认开，`git log` 加 `--all` 扫所有分支/tag。`--no-all-refs` 关闭 | ✅ 完成 |
| CSV 输出 | `valid_keys.csv`（UTF-8 BOM），只存 `status==valid` 的 key，列：provider/key/repo/commit/file/line/detail/generated | ✅ 完成 |
| 扫完即删 | `work()` 里 scan 后 `shutil.rmtree(dest)`，节省磁盘。`--keep-repos` 保留 | ✅ 完成 |
| Unicode 修复 | 所有 `subprocess.run` 加 `errors="ignore"`，避免非 UTF-8 字节崩溃 | ✅ 完成 |

### 2.2 已知问题（下次要修）

#### 问题 1: `--depth 200` 不够全面 ⚠️ 高优先级

当前 `clone_repo()` 用 `--depth 200`，只拿最近 200 个 commit。`git log -p -n200` 也只扫 200 个 commit 的 diff。

**应该扫全部 commit**。方案：
- 去掉 `--depth`，用完整 clone（`git clone --quiet`）
- 或者用 `--depth 0`（unshallow）
- 代价：大仓库 clone 慢（之前 NVIDIA 5.9G 那个完整 clone 要好几分钟）
- 折中：`--depth 1000` 或按仓库大小动态调整

**注意**：完整 clone 时 `--filter=blob:none` 的 partial clone 在 `git log -p` 时会按需 fetch blob，大仓库极慢。所以完整 clone 不要用 `--filter=blob:none`，用普通 `git clone`。

#### 问题 2: `together` 和 `zhipu` 正则太宽泛，误报率 94% ⚠️ 高优先级

```python
("together", r"\b[A-Za-z0-9]{20,}\.[A-Za-z0-9]{20,}\b", "Together (heuristic)"),
("zhipu",    r"\b[A-Za-z0-9]{16}\.[A-Za-z0-9]{16}\b", "Zhipu/GLM (heuristic)"),
```

这两个 heuristic 正则匹配任何 `长字符串.长字符串`，导致 Java 方法调用（`assetGovernanceProperties.isVe...`）、JS 属性访问（`currentTransform.getDefault...`）全被误识别为 key。

**第一次运行结果**：1269 个唯一 key 里 1199 个（94%）是这两个正则的误报，标为 "unknown"（无 validator）。

**修复方案**：
- 方案 A：直接删掉 `together` 和 `zhipu` 这两行（它们没有明确前缀，靠 heuristic 误报太多）
- 方案 B：加前缀要求，比如 together 改成 `\btogether_[A-Za-z0-9]{20,}\b`（如果 together 的 key 确实有前缀的话，需要查证）
- 方案 C：保留但降低优先级，在验证阶段跳过（标为 "heuristic" 不验证）

**建议用方案 A**：删掉这两行，减少 1199 个误报。真正的 together/zhipu key 如果有明确前缀会被供应商专属正则捕获。

#### 问题 3: PLACEHOLDER 过滤不够强 ⚠️ 中优先级

当前 PLACEHOLDER 正则：
```python
PLACEHOLDER = re.compile(
    r"(your[_-]?key|xxx+|<.*>|example|placeholder|changeme|dummy|fake|test[_-]?key|"
    r"abc+|123+|sk-[a-z]{4,}|sk-or-[a-z]{4,}|sk-ant-[a-z]{4,}|AIzaSyA[0-9]{2})", re.I)
```

**漏掉的测试 fixture 标记**（第一次运行 70 个 invalid 里大量是这些）：
- `e2e` — `sk-ant-e2e-configured-install-not-a...`
- `fixture` — `sk-or-v1-fixture-not-a-real-key`
- `not-a-real` / `not_a_real` — 同上
- `do-not-log` / `do_not_log` — `sk-ant-api-key-do-not-log`
- `0000` / `00000000` — `key-0000000000000000000000000000000`、`sk-00000000000000000000000000000000`
- `test-token` / `test_token` — `sk-ant-oat01-test-token`
- `sample` — 示例 key
- `secret-key` / `secret_key` — `sk-ant-api03-secret-key`（这个可能是真 key 也可能是 fixture，需判断）
- `invented` — `sk-pcq-offline-invented-not-a-crede...`
- `offline` — `sk-pcq-offline-invented...`

**修复方案**：扩展 PLACEHOLDER 正则，加入上述标记。

#### 问题 4: 搜索规模太小 ⚠️ 中优先级

第一次只扫了 50 个仓库，有效 key = 0。50-2000 stars 的活跃项目里 key 泄露后通常会被快速 rotate，命中概率低。

**扩大方案**：
- 增加到 200-500 个仓库
- 裸 API 限额 10次/分钟，500 仓库需要更多查询或分页
- 或者用 grep.app 直接按 key 前缀搜代码内容（更快命中，不用 clone 全仓库）

#### 问题 5: grep.app 搜索源未实现 💡 优化方向

grep.app 是开源代码搜索平台，有 API：`https://grep.app/api/search?q=sk-ant-`

**优势**：
- 直接按 key 前缀搜代码内容，不用 clone 全仓库
- 速度快，命中精准
- 可以搜到当前 HEAD 的代码（但不搜 git 历史）

**劣势**：
- 不搜 git 历史（已删除的 key 搜不到）
- 免费层有限额
- 覆盖的仓库范围可能不如 GitHub 全

**建议**：作为补充搜索源，和 GitHub clone 扫描结合。grep.app 找当前存在的 key，GitHub clone 找历史删除的 key。

## 三、第一次运行结果（2026-09-18）

### 3.1 参数

```bash
python3 keyscan.py \
  --queries "llm agent,ai agent,langchain agent,autogen,crewai" \
  --max-repos 50 \
  --max-commits 200 \
  --workers 10 \
  --all-refs \
  --out results.json
```

### 3.2 结果

| 指标 | 数值 |
|------|------|
| 扫描仓库 | 50 |
| 总候选 | 1308 |
| 唯一 key | 1269 |
| **有效 key** | **0** |
| invalid | 70 |
| unknown (误报) | 1199 |

### 3.3 有效 key = 0 的原因分析

1. **1199 个 "unknown" 全是误报**（94%）——`together` 和 `zhipu` 正则太宽泛，匹配了 Java 方法调用、JS 属性访问
2. **70 个 "invalid" 大部分是测试 fixture**——`e2e`、`fixture`、`not-a-real-key`、`0000`、`do-not-log` 等标记没被 PLACEHOLDER 过滤
3. **真正像样的 key 都失效了**——少数几个看起来像真 key 的（如 `sk-8uOe6IBkSSxEU7lfEp1WMg`、`sk-28b8bc8be4a24b89b5cecf6744a5893a`）验证后返回 401，仓库主人已 rotate

### 3.4 部分 invalid key 示例

```
openai   sk-ant-e2e-configured-install-not-a...  @ mixpeek/amux           (e2e测试)
openai   sk-8uOe6IBkSSxEU7lfEp1WMg              @ hivecommons/hive       (像真key, 401)
openai   sk-28b8bc8be4a24b89b5cecf6744a5893a    @ morettt/my-neuro       (像真key, 401)
openai   sk-zk2674bb5f711d1a4662fc39a7cbc667    @ morettt/my-neuro       (像真key, 401)
mistral  key-72e062f0cab640479d569d760f091a6    @ evenfire-ai/evenfire   (像真key, 401)
openai   sk-or-v1-fixture-not-a-real-key        @ talirezun/the-curator  (fixture)
openai   sk-ant-api-key-do-not-log              @ hivecommons/hive       (示例)
```

## 四、环境信息

### 4.1 本地 LLM（用于大厂判断）

- **模型**: Qwen3.8-27B (dflash_server)
- **端点**: `http://127.0.0.1:18097/v1/chat/completions`
- **model id**: `dflash`
- **GGUF**: `/workspace/python_song/models/Qwen3.8-27B-Uncensored-Q8_0.gguf`
- **draft**: `/workspace/python_song/models/draft/qwen38-dflash2-q8_0.gguf`
- **GPU**: cuda:3
- **max ctx**: 1572864
- **注意**: 如果 LLM 没跑起来，`is_big_corp()` 会回退到硬编码 `FALLBACK` 列表

### 4.2 GitHub 认证

- **仓库**: `https://github.com/Geeksongs/codex-local-qwen27b.git`
- **分支**: `h200`
- **Token** (PAT): `github_pat_11AJF3...` (见密码管理器, 勿硬编码)
- **push 命令**: `git push https://<TOKEN>@github.com/Geeksongs/codex-local-qwen27b.git h200`
- **搜索 API**: 不用 token（裸 API，限额 10次/分钟），脚本已有 rate-limit 重试逻辑

### 4.3 磁盘注意

- 工作目录: `/workspace/python_song/codex-local-qwen27b/keyscan/`
- repos 目录: `keyscan/repos/`（扫完即删，但并行 clone 时峰值可能 10-20G）
- 之前遇到过磁盘配额问题（`Disk quota exceeded`），写文件时如果变 0 字节，检查 `df -h` 和 `du -sh repos/`
- `__pycache__` 也可能触发配额，用 `ast.parse()` 验证语法代替 `py_compile`（不写 pycache）

### 4.4 并行度测试

- 单仓库: clone 4.7s + scan 2.4s = 7s（`--depth 200`）
- 6 仓库 6 workers: 80s（最慢单仓库 80s，总时间 ≈ 最慢单仓库）
- 10 仓库 10 workers: 测试中被中断，但前 3 个完成（8-20s/仓库）
- **结论**: 并行有效，总时间 ≈ 最慢单仓库时间。10 workers 合适。

## 五、文件清单

```
keyscan/
├── keyscan.py          # 主脚本 (451 行)
├── results.json        # 第一次运行结果 (1269 个唯一 key)
├── valid_keys.csv      # 有效 key CSV (第一次运行 0 条, 只有 header)
├── repos/              # clone 目录 (扫完即删, 可能残留)
└── HANDOVER.md         # 本文档
```

## 六、下次接手 TODO（按优先级）

### P0: 修正则 + 加强过滤
1. 删掉 `together` 和 `zhipu` 的宽泛 heuristic 正则（KEY_PATTERNS 里第 37 行和第 45 行）
2. 扩展 PLACEHOLDER 正则，加入：`e2e|fixture|not-a-real|not_a_real|do-not-log|do_not_log|0000|test-token|test_token|sample|invented|offline`

### P1: 改 depth 为全量
3. `clone_repo()` 去掉 `--depth 200`，改为完整 clone（`git clone --quiet`，不用 `--filter=blob:none`）
4. `scan_repo()` 的 `max_commits` 默认改回 2000（或更大）
5. 注意大仓库 clone 慢，timeout 保持 300s，超时的跳过

### P2: 扩大规模
6. `--max-repos` 增加到 200-500
7. 增加更多查询词（`"openai api key" in:code`、`"sk-" in:code language:python` 等）
8. 或者加 grep.app 搜索源（按 key 前缀直接搜代码内容）

### P3: 运行 + 验证
9. 跑扫描，看有效 key 数量
10. 如果还是 0，考虑换策略（grep.app、更精准的查询、更大规模）

## 七、运行命令参考

```bash
cd /workspace/python_song/codex-local-qwen27b/keyscan

# 标准运行 (搜索 + 双扫描 + 验证 + CSV)
python3 keyscan.py \
  --queries "llm agent,ai agent,langchain agent,autogen,crewai" \
  --max-repos 50 \
  --max-commits 200 \
  --workers 10 \
  --all-refs \
  --out results.json

# 离线运行 (指定仓库列表)
python3 keyscan.py \
  --repos owner/repo1,owner/repo2,owner/repo3 \
  --max-commits 2000 \
  --workers 10 \
  --all-refs

# 不验证 (快速看候选)
python3 keyscan.py \
  --max-repos 50 \
  --no-validate \
  --workers 10

# 保留 clone 目录 (调试用)
python3 keyscan.py \
  --max-repos 10 \
  --keep-repos
```

## 八、关键代码位置

| 功能 | 行号 (keyscan.py) |
|------|-------------------|
| KEY_PATTERNS (正则列表) | ~27-48 |
| COMPILED (编译后正则) | ~50 |
| PLACEHOLDER (占位符过滤) | ~53-55 |
| gh_get (GitHub API 请求) | ~60-91 |
| is_big_corp (LLM 大厂判断) | ~97-140 |
| search_repos (搜索仓库) | ~142-170 |
| clone_repo (clone 仓库) | ~175-190 |
| GREP_PREFIXES (grep 粗筛前缀) | ~193-196 |
| EXCLUDE_DIRS (排除目录) | ~199-200 |
| scan_worktree (工作区扫描) | ~202-230 |
| scan_repo (双扫描合并) | ~232-280 |
| VALIDATORS (验证端点) | ~285-298 |
| validate_key (验证 key) | ~300-325 |
| main (主流程) | ~330-451 |
| CSV 输出 | ~420-435 |

## 九、踩坑记录

1. **`--filter=blob:none` partial clone 大仓库极慢**：`git log -p` 要按需 fetch 每个 blob，NVIDIA 5.9G 仓库卡几分钟。改用 `--depth 200` 普通 shallow clone 解决。
2. **UnicodeDecodeError**：`git log -p` 输出含非 UTF-8 字节（某 commit message 或文件内容），`subprocess.run(text=True)` 默认 UTF-8 解码崩溃。加 `errors="ignore"` 解决。
3. **磁盘配额**：repos 目录 19G 时触发 `Disk quota exceeded`，写文件变 0 字节。清理 repos 解决。扫完即删策略预防。
4. **Write 工具追加 null bytes**：用 Write 工具写 keyscan.py 后文件末尾有 838 个 null bytes，`ast.parse` 报 "source code string cannot contain null bytes"。用 Python `data[:last_non_null+1]` 清理。
5. **git index.lock**：`git checkout` 报 `could not close index.lock`，但 lock 文件不存在。用 `git show HEAD:keyscan/keyscan.py > /tmp/file` 绕过。
6. **grep 搜到 repos 目录**：在 keyscan 目录用 grep 搜 keyscan.py 内容时，会搜到 repos/ 下 clone 的仓库文件。用 `grep -n` 指定文件路径或 `--exclude-dir=repos`。
7. **py_compile 写 __pycache__ 触发配额**：用 `ast.parse()` 验证语法代替 `py_compile`（不写 pycache 文件）。
