# 访问历史预取（issue #6，任务 6）

实现入口为 `m0a/history_prefetch.py`，CPU 回放入口为
`python -m m0a.history_replay`。这是一组候选预测基线，不改变原生 indexer
选集，不加载模型，也不将候选覆盖率解释为同步回载量或在线加速。

## 预测与更新

每个任务、run、request、context version、rank、layer、prefill/decode 分别维护
实例。`global` 指同一个 KV 命名空间和计算阶段内跨事件阶段的全部已完成历史；
不是跨任务共享逻辑 block ID。`stage` 在每个新事件 occurrence 冷启动，不能
复用过去出现过的 stage ID。上下文重写、截断后的 ID 复用或模型改变必须换命名空间。

`PredictionContext` 只含 query 序号、预测时刻、stage 和预测前已生成且可搬运的
eligible blocks。`predict(context, budget=...)` 返回按优先级排列的 `Candidate`，
其中带有 block 访问次数、最近完成 query、区域访问次数和区域最近完成 query，
可供任务 8 复用。只有 `observe(query=..., completion_step=..., selected_blocks=...)`
接收真实选集，必须在 query 完成后调用。调用者应将原生 token ID 按协议映射并去重
为逻辑 block，剔除 padding；此接口拒绝负数、重复、布尔值和乱序输入。

三种排序信号分别是访问频率、最近访问、历史区域统计。区域是明确的固定地址区间
`block // region_blocks`，不冒充语义区域。每个 query 对同一区域最多计一次；
区域策略可预测同一区域内以前未选过的 eligible block。频率并列时按最近访问排序，
最后按 block ID 保证确定性。阶段策略没有全局回退，以便直接测量冷启动代价。

## 因果性与采集衔接

回放 JSONL schema v1 每行包含：

```text
schema_version, source_kind (synthetic_fixture | native_trace),
task_id, run_id, request_id, context_version, rank, layer,
phase (prefill | decode), query, batch, prediction_step, completion_step,
stage, event_type, block_count, eligible_blocks, selected_blocks
```

query 从 0 连续递增，batch 从 0 连续递增。一个 prefill 可见性批次内的所有
query 先统一预测，再发布该批次完成选集，不能把 JSONL 行顺序当成在线完成顺序。
同一 batch 的 stage、event_type、预测和完成时刻相同，下一 batch 必须严格晚于
上一 batch 的完成；decode 每 batch 一个 query。时刻是由采集端证明的逻辑事件序，
不是 CPU 回放耗时。eligible_blocks 必须来自预测前快照，不能从未来原生选集构造。

`block_count` 用于验证完成时选集越界；选集可以包含预测时尚不可搬运的新 block。
候选始终受 eligible blocks 和同一候选预算约束，oracle 也不能选不可搬运的 block。
task 1 的统一协议确定后，采集适配器需将 namespace、可见性和 block 单位映射到上述
接口；这份 schema 不替代 task 1 协议，也不绕过 task 4 的真实性与稳定性验收。

## 运行与检查

```bash
python -m unittest m0a.test_history_prefetch -v
python -m unittest m0a.test_history_glm_import -v
python -m m0a.history_fixture --output m0a/runs/history-fixture/access.jsonl
python -m m0a.history_replay \
  --trace m0a/runs/history-fixture/access.jsonl \
  --budget-blocks 2 --region-blocks 4 \
  --output-dir m0a/runs/history-replay-NEW
```

输出目录必须不存在。`summary.json` 记录输入 SHA256、配置、候选 precision、
原生选集 coverage、未命中候选数量、CPU 预测耗时，以及逐独立 task 的 coverage。
按 prefill/decode、事件类型及 boundary/steady 窗口分组；`predictions.jsonl` 保存
逐 query 候选、阶段偏移和相邻选集 Jaccard。boundary 默认为阶段开始后的 4 个
query，可通过 `--boundary-queries` 调整；第一次 query 的 Jaccard 是 null。
跨阶段第一行的 Jaccard 仅是事后诊断，不传入预测器。

对照包含不预取、上一选集复用、全局/阶段各三种历史策略和候选 oracle 上界。
上一选集复用不是 ECHO；selected-only trace 不含复现 ECHO 所需完整 indexer scores。
oracle 在所有因果预测完成之后单独构造，不用于更新任何策略。

fixture 的 persistent 场景保留热点，switching 场景每 8 个 query 改变热点，
用于检查跨阶段保留与冷启动的不同表现。它是人工工程输入，不是银行/零售的业务
任务，也不支持泛化结论。不训练、不在评估任务上调参；真实数据的 task_id 还须由
数据集侧把重复、改写和分支合并到同一独立任务，不能把重复样本当独立任务。

## 旧 GLM trace 的补充诊断

`history_glm_import` 只导入显式指定 case 的 no_offload、repeat 0，核对独立审计的
压缩文件 SHA256、measurement 的选集 SHA256、6 层及连续 31 个 decode query。
旧 trace 没有 prefill 完成批次信息，因此不推断该信息，也不把不同请求拼成连续阶段。
每个 case 只包含一个 stage，global 与 stage 在此诊断中应完全一致。

```bash
python -m m0a.history_glm_import --run-dir /path/to/audited-glm-run \
  --case-id retail_tool_result_8192_0:event \
  --case-id banking_knowledge_tool_result_8192_0:event \
  --block-tokens 1 --output-dir m0a/runs/history-glm-input-NEW
python -m m0a.history_replay \
  --trace m0a/runs/history-glm-input-NEW/access.jsonl \
  --budget-blocks 512 --region-blocks 128 \
  --output-dir m0a/runs/history-glm-replay-NEW
```

`--block-tokens` 必须显式给定。1 表示原生 KV token 单位；128 表示固定逻辑页。
比较时保持单位一致，不能把不同宽度的候选命中率直接比较。仅允许预测前已完整
生成的 block；不完整的尾页保守排除。manifest 保存输入审计/measurement/trace
摘要与证据限制。这些旧样本仍有固定续写和周期性对照，不能替代新的 Agent 数据集。

## 当前边界

任务 6 的预测代码和因果性检查可独立运行。真实 Agent 的热点是否跨 query/阶段
持续，需要 task 2–4 的有效输入。本回放不模拟缓存容量或搬运，不输出同步等待、
TTFT、TPOT、HBM 节省或字节收益；任务 5/10/18 的统一缓存与实际搬运路径负责这些
测量。所有策略候选预算一致；在线接入时需再由同一个缓存管理器按驻留、容量和
字节预算调度。这里的 candidate_waste 是未命中候选数，不是实际浪费搬运量。
