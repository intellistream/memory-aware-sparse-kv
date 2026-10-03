# 离线 Transition Working-Set Cache

本页记录旧工作区的历史实现与验收；历史运行数据保留在本地恢复归档中，未纳入当前 Git 仓库。以下 `/workspace` 路径不是重建后 Pod 的持久目录。

实现日期：2026-09-28。入口为 `m0a/transition_replay.py`；只使用 Python 标准库，不启动模型或 NPU，不改动 instrumentation、native selected set、模型权重、镜像、Driver/CANN 或 serving 行为。

当前实现解决离线数据合同、候选预测、缓存身份与传输回放。它不代表在线 offload 已验收。接受的工程运行为 `m0a/runs/transition_cache_20260928_r03/`，输入继承 `transition_cache_20260928_r02/fixture/`；旧运行及失败记录保持原样。

## 运行

在服务器 `/workspace/memecho` 执行；同步后也可在本地根目录执行：

```bash
python3 -m unittest discover -s m0a -p 'test_*.py' -v

# 每次使用不存在的目录。该 fixture 的 token/选择/容量全部是人工数据。
python3 m0a/transition_fixture.py --output-dir /tmp/transition-fixture-NEW
python3 m0a/transition_replay.py \
  --trace-dir /tmp/transition-fixture-NEW/traces \
  --sidecar /tmp/transition-fixture-NEW/sidecar.json \
  --config /tmp/transition-fixture-NEW/config.json \
  --output-dir /tmp/transition-replay-NEW
```

真实数据用同一 CLI 替换三个输入；必要时传 `--source-root` 指定 compressor 源码根目录。输出目录存在时拒绝覆盖。生成 `events.jsonl`（每个事件 × 策略 × 粒度）和 `report.json`（汇总、输入与实现 SHA-256、限制）。报告不读取 trace 时间戳计算 stall。

旧 M0-A 的 boundary 相邻两行 trace 不足以回放 32 个 token，不能补齐或复制行；必须重新采集所需位置并提供真实 token sidecar。`synthetic: false` 时还要求固定 profile 的完整层数。合成 fixture 明确允许单层子集，报告包含 `lane_counts`。

## Sidecar 合同（独立 schema v1）

保留原生 trace schema v1/v2。顶层字段为 `schema_version: 1`、显式布尔值 `synthetic`、`snapshots` 和 `events`。fixture 的 `sidecar.json` 是完整可运行示例。

每个 snapshot 保存：

| 字段 | 含义 |
|---|---|
| `snapshot_id`, `session_id`, `context_version`, `sequence` | 快照、会话、有效上下文版本和会话内事件序号 |
| `model_id`, `model_revision`, `tokenizer_sha256` | 完整模型与 tokenizer 身份；不能用模型简称替代 |
| `token_ids` | 实际送入模型的准确 token 序列，包含 chat template；不能用长度、文本近似或 hash 代替序列 |
| `generated_tokens` | 快照时已生成 KV 的原始 token 前缀长度，不能大于恢复位置 |
| `regions` | 按 token 划分的完整、不重叠半开区间：`start`, `end`, `type`, `created_at` |
| `lanes` | 每个 rank/layer/KV 类型的库存与必要的 compressor 依赖 |

region type 有界为 `system/user/assistant/tool/memory/task/other`；跨 region 的 compressor 支持区间生成 `mixed` 特征。`created_at` 为创建该 region 的事件序号，不能来自未来。普通 KV 的 `generated_ids` 必须等于 `[0, generated_tokens)`，防止 exporter 以未来 selected set 筛选库存。Compressed KV 库存必须完整对应支持映射中已经 ready 的单位。

每个 event 必须保存 `event_id/request_id/run_id`、`previous_snapshot_id/snapshot_id`、`event_type/previous_state`、`event_position/resume_position`、`workload_id/episode_id/source_trace_id/pair_id/variant/repetition/trajectory_id/split`。event type 支持六类 M0-A 事件及 control 的 `no_event`。`split` 由语料制作者明确指定 `train` 或 `eval`。`source_trace_id` 应是跨重复运行稳定的、已限定命名空间的来源身份。

`trajectory_id` 区分真实独立缓存路径、成对分支和重复运行；同一路径的事件按 predecessor 顺序提供。若下一次恢复需要复用本窗口新生成的 KV，在当前 event 提供 `post_window_snapshot_id`，引用窗口结束快照：模型/session/version/token 序列/sequence 与当前快照相同，`generated_tokens = min(resume_position + 32, len(token_ids))`，库存更新为该时点的完整 KV。下一 event 的 predecessor 引用这个快照。缺少该字段时只承接当前恢复快照已生成的 KV，不能伪称复用了本窗口的新 KV。

真实 sidecar 必须由 tokenizer 和实际 KV 生成状态导出，并与原始 response 的 `prompt_token_ids_sha256`、tokenizer 文件 hash、run/request identity 独立核对。CLI 校验 trace 的上下文长度、模型 identity、rank 文件名、请求唯一映射、完整 selected IDs 和因果可用性；原生 trace 本身没有 token 序列，不能仅凭本 CLI 证明 sidecar token 序列与实际请求相同。这里不从旧 pairs 文本猜测 token IDs。

## 候选有效性与 DeepSeek 支持映射

候选必须同时满足：同一 session/model/revision/tokenizer；之前已经生成；当前已经生成；对应 native KV ID 的旧、新支持区间和 ready position 一致；其全部依赖 token 位于准确公共前缀。插入/改写 token 后，公共前缀之后的普通 KV 和跨越改写点的 compressor KV 全部失效。有效前缀可以迁移到新 `context_version`，旧版本后缀不能命中。相同版本内 token 改写被拒绝。

GLM 使用普通 KV 的 `[id, id+1)` 支持区间，ready position 为 `id`。DeepSeek 的 raw ID 是 compressed-KV 序列 ID，绝不按原始 token 位置除以 4 学习 region。

每个 DeepSeek lane 必须提供 `support_intervals`：字符串 native ID 映射到 `[start, end, ready_position]`，以及 `support_evidence`：

```json
{
  "path": "source/实际实现路径.py",
  "sha256": "完整源码文件SHA256",
  "start_line": 1,
  "end_line": 20,
  "explanation": "针对固定 compressor 配置说明依赖区间与输出可用位置"
}
```

路径必须在 `--source-root` 内，校验源码 hash 和行范围；所有原生 selected IDs 都必须有映射，且依赖及 ready position 因果可用。缺失时拒绝 region 学习和该配置的回放。应检查固定模型 compressor 的 overlap、state、chunk 和 kernel 语义后导出映射。现有 ratio=4 合同与逻辑 block 映射不是 region 依赖证明；本轮没有为真实 DeepSeek kernel 宣称一个未经验证的映射。单元测试中的非均匀重叠 compressor 是明确标注的人工合同。源码 hash 校验能证明引用完整性，不能自动证明 exporter 对源码的解释正确。

## 学习与策略

目标为恢复调用后的前 32 个 suffix prefill token 的 native selected KV 并集；最后不足 32 个 token 时按实际长度回放，并记录 `suffix_tokens`。每个 lane 需要 `resume-1` 的 warmup selected set 和整个窗口的完整 selected set；缺行、重复行、错误 rank、错误映射及非因果选择均失败。

Transition Table 的特征为每层 `region_type × age_bucket × 16 个相对位置 bin`。age 分为 0、1、2–3、4–7、8+；单个表最多 8×5×16=640 个 cell。命中率为 `(hits+1)/(accessible+2)`，hits 是窗口内被选中的可候选单位数量，accessible 是所有有效候选单位数量。学到的是类别统计，不是跨请求的绝对 KV/block ID。

优先查询 `previous_state + event`；该层历史事件观察少于 8 次时回退 event 表。多个 rank 的同一个事件只计一次历史观察；机会数与命中数按各 rank 库存累加。表按模型/revision/tokenizer/layer/KV kind 隔离，LRU 最多 `max_tables` 个上下文；每个表只保留一个最近 observation identity 用于 rank 去重。

训练事件严格先让所有策略、两个粒度预测和回放，再更新统计；eval 表冻结，不更新也不调整表 LRU。所有 train 事件先处理，随后评估 held-out 组。训练与评估按 session、workload/episode、workload/pair 和 source trace 的连通分量隔离；任何一个链接跨 split 都拒绝。重复运行及 event/control 不能换 group label 绕过检查。缓存按 trajectory 隔离，结束的 trajectory 会释放。

| 策略 | 候选排序 |
|---|---|
| `demand_only` | 不预取，缺失时同步补齐 |
| `sequential_selected_set` | 已知 warmup selected ID 的原生顺序，随后按相邻 native ID 距离排序；不是 ECHO 实现 |
| `region_recency` | 同样的 region/age/bin 统计，所有事件合并，事件与 previous state 不作为条件；相同分数偏向较新的位置 |
| `transition` | 条件概率排序，平分时按 recency 排序 |
| `transition_sequential` | 交替取 Transition 与顺序候选并去重 |
| `shuffled_events` | train/eval 分别固定 seed 打乱 event label，并训练独立表；state 保留，因此仍可能携带信号 |
| `oracle_lookahead` | 唯一允许看未来 selected set 的候选排序；仍执行同容量、预算和 LRU，是可执行参照 |

所有普通策略的打分和排序逻辑都不访问未来 selected IDs；共同回放驱动只在 oracle 分支使用未来选择，普通策略预测后才提取目标标签。报告还给出 `oracle_upper_bound`：初始缺失的必要旧 KV 传输字节减去相同预取预算得到乐观同步字节下界；放松淘汰压力，因此是改善上界，不是可执行的最优调度。可执行 lookahead 与这个乐观界不能混为一谈。

## 容量、传输与报告

配置必须明确指定容量、每窗口预取预算和每种 native KV 单位的字节数。示例数字仅为人工工程参数，不能当成 GLM/DeepSeek 的实际 KV 大小：

```json
{
  "schema_version": 1,
  "capacity_bytes": 512,
  "prefetch_budget_bytes": 256,
  "kv_unit_bytes": {"kv_token_position": 2},
  "max_tables": 256,
  "bootstrap_draws": 1000,
  "seed": 7
}
```

两个粒度分别回放：native 单位和 128 个 native KV 单位的 page；compressed page 是 128 个 compressed KV 单位。page 总按完整 128 单位收取容量与传输字节，保留 native 有效位掩码，包含多个 region 或部分有效前缀时不能把整页当成有效候选。缓存身份包含 session、model/revision/tokenizer、有效版本、rank、layer 和 KV 类型。

任何一次 attention 的完整有效 selected set（按相应粒度换算）超过容量时拒绝配置。同步 demand 时保护该 attention 的所有需求页，加载完毕后逐一断言全部原生 selected KV 已就绪。错误预测只能改变缓存与传输；selected set 不变。每个事件仅在窗口开始时预取一次；所有策略使用相同容量与预算上限。

新 KV 根据生成时点在本地写入缓存，不计同步 recall；若随后淘汰再读取则计 recall。假设 host store 保留所有生成的 KV，未模拟 write-back DMA 和计算成本；缓存仅覆盖 native sparse selected KV，没有模拟模型全部 KV、SWA、compressor state 或实际物理布局。warmup 传输不计入窗口指标。

`events.jsonl` 记录 precision/recall、候选与目标数量、prediction SHA256、预取/有效/浪费 payload 字节、同步 recall 字节、污染、预取淘汰和规划 CPU 时间。prediction 指实际传输的缺失候选；precision/recall 用候选范围内的 native KV 单位计算，空预测或空目标记 0。有效预取字节是被真正预取且在仍驻留时被 attention 使用的不同 native KV payload；浪费包含无用单位及 page 放大。已经淘汰并由 demand 重载的单位不能再算预取命中。污染字节定义为相对 demand-only 的正向净同步字节增加，另外保留被预取淘汰的 native 单位计数。

`report.json` 仅用 eval 汇总。每个策略/粒度提供逐事件及独立连通组均值的 p50/p95/p99、micro precision/recall、独立组 bootstrap 95% 区间、event−control 配对差值，以及策略−顺序基线的同步字节差值。每组先平均重复和相关分支；不足两个独立组时区间为 null。CPU 规划时间使用 `process_time_ns()`，覆盖打分、排序、去重、预算装包与缓存预取操作，不是 IO、模型执行或 recall stall。

## 工程验收结果（2026-09-28）

服务器和本地完整回归均为 **41/41 通过**（原有 18 项 + 新增 23 项）。
接受运行 `transition_cache_20260928_r03` 使用 64 个训练事件、24 个 eval 事件；eval 来自 3 个独立合成 episode，含成对分支与两次重复。
输入为单个 GLM profile 层的人工 native trace：88 个请求、2,904 行；总计 1,232 条策略/粒度回放记录、39,424 次 attention 就绪断言。容量 512 字节，每窗口预取上限 256 字节，native 单位人为设为 2 字节。

下表为 eval 每事件平均同步 recall 字节与 micro prediction 指标：

| 策略 | native 同步字节 | page128 同步字节 | native precision / recall | page128 precision / recall |
|---|---:|---:|---:|---:|
| `demand_only` | 8.000 | 256.000 | 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| `sequential_selected_set` | 8.000 | 64.000 | 0.00000 / 0.00000 | 0.02344 / 0.75000 |
| `region_recency` | 0.000 | 64.000 | 0.03125 / 1.00000 | 0.02344 / 0.75000 |
| `transition` | 0.000 | 0.000 | 0.03125 / 1.00000 | 0.03125 / 1.00000 |
| `transition_sequential` | 0.000 | 0.000 | 0.03125 / 1.00000 | 0.03125 / 1.00000 |
| `shuffled_events` | 1.000 | 85.333 | 0.02734 / 0.87500 | 0.02083 / 0.66667 |
| `oracle_lookahead` | 0.000 | 0.000 | 1.00000 / 1.00000 | 0.03125 / 1.00000 |

Transition 两个粒度均平均预取 256 字节，其中有效 payload 8 字节、浪费/放大 248 字节。native 无事件 region/recency 同样达到零同步字节，说明该粒度的 fixture 不能证明事件条件有额外收益。目标模式由 fixture 人工设定；这些数字不能支持真实 workload 的 20% 贡献门槛。

Transition 规划 CPU 时间（服务器，毫秒）为：

| 粒度 | p50 | p95 | p99 |
|---|---:|---:|---:|
| native | 5.462 | 5.610 | 5.756 |
| page128 | 3.880 | 3.914 | 3.920 |

完整汇总、全部策略尾部数据、配对差值及独立组区间保存在历史归档 `m0a/artifacts/transition-cache-20260928/report.json`，逐事件记录保存在同目录的 `events.jsonl`。接受运行使用同一 fixture 在本地再次 CLI 回放并比较所有非 CPU 时间结果，验证日志与 SHA-256 随产物保存。

## 同步与归档

所有程序和文档先在 `hust:/workspace/memecho` 修改，每批立即按新增/修改文件清单同步到 `/home/ruan/memecho`，然后核对 SHA-256。持久化同步 helper 在本地执行：

```bash
python3 scripts/sync_transition_cache.py \
  m0a/working_set.py m0a/transition_replay.py \
  m0a/transition_fixture.py m0a/test_transition_replay.py \
  docs/transition-working-set-cache.md docs/experiment-handoff.md m0a/README.md
```

helper 只下载显式相对路径，不做删除同步；SSH 禁用配置中的端口转发，避免并行只读连接争抢转发端口。校验失败立即退出；必须恢复同步链路后才能继续服务器修改。`m0a/replay-sync.jsonl` 在服务器追加记录并同步本地，保存 server/local 路径、SHA-256 和 UTC 同步时间；该账本本身不递归记录自己的 hash。运行日志和轻量报告在模块验证后另批同步。最终运行的 `SHA256SUMS` 及本地验证日志随产物保存。

`server-backup-20260926` 继续作为原始封存快照。当前代码、文档与轻量报告进入现有 `code-archive` Git，形成 commit 与完整 bundle。该归档当前没有 remote，GitHub 上传沿用已有授权，在目标仓库地址明确后执行。

## 在线阶段的限制与门槛

合成 fixture 仅验证工程链路，即使其字节收益很大也不能作为研究效果证据。已有真实 GLM/DeepSeek trace 缺准确 episode/region/KV sidecar 或完整窗口时不能冒充可回放语料。

在线阶段仍要求至少两个代表性 agent workloads；在同容量、并发和实际传输条件下，相对最强顺序基线的同步 recall 或 stall 至少改善 20%，并验证 native selected set 与输出等价。还需测真实 DMA、write-back、TTFT/TPOT、吞吐、KV residency、图重编译、规划开销与 stall。离线字节模型不能替代这些验证。
