# Issue 6：任务 1–10 集成入口

统一口径见 [实验协议](issue6-experiment-protocol.md)。本套代码保留真实模型
能力失败、原始不稳定样本和受控回放，按证据验收，不以生成图表代替任务完成。

| 任务 | 实现/入口 | 本环境可以验证的范围 |
| --- | --- | --- |
| 1 | `issue6_protocol.py`、统一协议 | 输入可见性、namespace、容量/预算、phase 隔离 |
| 2 | `issue6_serve`、`issue6_workload`、`issue6_agent_probe` | 真实 GLM/GLM benchmark 与独立工具探针；失败如实保留 |
| 3 | `issue6_workload` | 两域固定任务、分组划分、原始事实/动作摘要、非周期只读对照 |
| 4 | `issue6_capture`、扩展 `glm_native_trace` | native indexer/attention 交接、前驱/候选、完整 score、重复和离线重算 |
| 5 | `issue6_evaluate.Cache`、ECHO-style | 等容量/预算的无预取、上一选集、decode EMA 与候选 oracle |
| 6 | history frequency/recency/region、stage frequency | 真实受控 decode 和阶段切换 fixture；跨实际事件结论待数据 |
| 7 | `SemanticLSA` | 训练任务拟合分布语义、片段映射、索引/编码/检索开销 |
| 8 | `EventModel` | 训练标签学习区域优先级；事件/无条件/无历史消融 |
| 9 | adaptive / no-semantic / no-event / fixed | 已完成反馈驱动预算分配、阶段重置、组合开销 |
| 10 | `issue6_evaluate` | 候选、同步回载、总搬运、浪费、开销、任务级区间、自动 PNG/PDF |

任务 2 的实际工具闭环未通过时，任务 3 的自然完成有效性，以及任务 4–10
针对自主 Agent 的研究验收均保持未完成。受控工程回放独立记录，不转换为
自主 Agent trace，也不替代项目默认 DeepSeek 的八卡实验。

## 运行

先通过当次 SSH 入口确认物理卡及运行时编号。不要复用历史端口或物理卡号。
现有 `activate.sh` 保留 CANN/ACL 路径，追加项目路径时不要覆盖原 `PYTHONPATH`。
输出目录必须全新；所有生成结果位于 Git 忽略的 `m0a/runs/` 中。

完整脚本：

```bash
bash scripts/run_issue6_suite.sh "$PHYSICAL_NPU" "$RUNTIME_DEVICE" my-new-run
```

脚本在回环端口 8866 启动已封存 tiny 模型；运行 τ³ 用户模拟器、真实环境及
Agent，再做独立的 seeded-customer 工具探针；关闭服务并确认退出后，采集
受控 native trace、独立复算并进行 CPU 评估。它不下载或自动替换模型。
绘图依赖在运行目录内的独立 `analysis-env` 安装，固定为 Matplotlib 3.9.4、
NumPy 1.26.4 和 ContourPy 1.3.2，
保留安装日志与环境 freeze，不修改原固定运行环境。

分步运行也可：

```bash
# 在激活的固定 tau3 环境中，已有本地 GLM 服务时：
python -m m0a.issue6_workload --tau-source "$TAU3_SOURCE" \
  --output m0a/runs/agent-NEW --max-tokens 512 --reasoning-effort low
python -m m0a.issue6_agent_probe --tau-source "$TAU3_SOURCE" \
  --materials m0a/runs/agent-NEW/prompt-material.jsonl --output m0a/runs/probe-NEW

# 关闭模型服务后，激活固定原生环境，设置本次可见性：
python -m m0a.issue6_capture --materials m0a/runs/agent-NEW/prompt-material.jsonl \
  --physical-npu "$PHYSICAL_NPU" --runtime-device "$RUNTIME_DEVICE" \
  --repeats 2 --output m0a/runs/native-NEW
python -m m0a.issue6_capture --audit m0a/runs/native-NEW > m0a/runs/native-NEW-audit.json
# 在有 m0a/issue6-analysis-requirements.txt 依赖的独立分析环境中：
python -m m0a.issue6_evaluate --trace m0a/runs/native-NEW/trace.jsonl.gz \
  --native-audit m0a/runs/native-NEW-audit.json \
  --capacity 4096 --capacity 8192 --budget 128 --budget 512 \
  --output m0a/runs/evaluate-NEW
python -m unittest m0a.test_issue6_suite m0a.test_glm_native_trace -v
```

完整 score 的 float32 参考是诊断路径，会同步 NPU。采集日志和模型响应中的
耗时不能用来报告在线预取性能。新接口的容量模拟不会修改原生 tensor、
原生选集或模型权重；使用必要 KV 前始终保证同步缺失召回。

## 产物与复核

Agent `manifest.json` 保存固定 benchmark 源摘要和全部请求日志摘要。
`episodes.json` 包含失败原因；`dataset.jsonl` 保存完整任务与划分；
`prompt-material.jsonl` 保存受控输入及工具 schema。数据校验拒绝跨划分变体
或事实/评价动作改变；这些检查不声称官方 benchmark reward 已通过。

native `cases.json` 保存实际 token IDs、片段精确映射、校准续写与材料来源；
每案例重复的 `*.jsonl.gz` 保存完整原生行与 score；`stability.json` 包含不稳定
位置，`trace.jsonl.gz` 仅导出 repeat 0 的连续 decode 供评估。manifest 对所有
产物做摘要，另存物理/逻辑映射、代码和 native capture 摘要。

独立审计重读 raw trace、前驱派生信息、score/top-k 一致性和导出内容，重新
计算稳定性；不会只读取通过标志。采集完整性与模型自由输出稳定性分别解释。
native 评估必须绑定独立审计与归档摘要。不稳定输入默认拒绝；若明确要分析
不稳定样本，使用 `--include-unstable` 保留全部样本并在报告中携带审计状态。

评估输出 `summary.json`、`per-task.csv`、`predictions.jsonl`、
`fitted-models.json`、`comparison.png` 和 `comparison.pdf`。拟合参数、代码摘要及
Python/NumPy 版本一并保存。共享语义特征先统一准备，构建/编码/检索成本单列，
不把先运行策略承担的建索引成本算作其独有预测开销；构建时间包含编码，不能
将重叠组件重复相加。图仅展示第一容量、最大预算的一个切片，
全部配置在 JSON/CSV 中。oracle 只用作诊断；报告字节量明确标记为模拟。
逐 query JSONL 保留完整 namespace 和按优先级排列的候选 ID。六面板图同时展示
候选精度、原生覆盖率、同步回载、浪费预取、总搬运和 predict 调用开销；
该开销不包含输入解析、Context 构造、native score 重建及单列的共享特征准备。
