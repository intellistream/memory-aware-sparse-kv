# 实验进展与证据边界

本页记录定性时间线。原始测量、日志、结果索引、图表和校准续写不进入当前 Git 文件树；旧公共提交仍可从 Git 历史读取，HUST 的原始运行资料保存在 `/root/experiments/` 和归档目录。下列结论不得代替原始运行状态。

## 时间线

| 时间 | 阶段 | 已确认的进展 | 尚未跨过的门槛 |
| --- | --- | --- | --- |
| 2026-09-16 至 09-20 | Stage 0 | DeepSeek 八卡服务、固定输入的吞吐基线及 profiler 采集完成；短固定续写重复一致。 | 旧的较长续写曾出现末 token 差异；未采原生 selected set，也未验证预取效果。 |
| 2026-09-24 至 09-26 | M0-A | DeepSeek 原生 DSA selected-set hook 的 pilot 经独立审计通过；GLM tiny 单卡 SFA trace 和等长成对输入通过工程预验证。 | 原 pilot 的验证器曾误判 request ID 与层数；GLM tiny 的合成成对实验不能外推到 DeepSeek 或真实 agent。 |
| 2026-09-28 | 离线回放 | Transition Working-Set Cache 的合成 sidecar、因果检查、训练/评估隔离和多策略 CPU 回放通过工程验收。 | 人工 KV 大小、缓存容量及合成选集不能用于在线延迟或真实 DMA 主张。 |
| 2026-09-29 至 10-06 | DeepSeek 与 τ³ | 逐步建立两域真实任务的采集、提示封存、预检、哈希同步和服务恢复。一次旧 trace 的独立 CPU 后处理完成；最新允许输出漂移的运行完成请求与最终哈希。 | 最新正式 trace 因采集位置超出真实请求上下文而失败，sidecar 与四组回放未开始；重复输出和 selected set 漂移仍限制严格比较。失败状态保留原样。 |
| 2026-10-10 | ECHO 代码阶段 | 增加完整分数合同、CPU 回放、原生 top-k 核对和受控比较门槛。 | NPU 上完整分数来源及融合、可计时的异步 host-to-device 路径尚未实现。 |
| 2026-10-12 | GLM 在线原型 | 单卡 shadow-KV 原型用真实 agent 工具事件提示完成三模式正式运行；输入、固定续写、原生 top-k、KV payload 与注意力核对通过。银行稳定子集内，顺序执行的预取原型显示延迟增加；零售只作探索性观察。 | 原生 NPU KV 仍保留，预取与计算未异步重叠；这不是 HBM 节省、DeepSeek 效果或论文系统的证据。 |

研究目标仍是：在至少两个代表性真实 agent workload 上，保持模型原生 selected set 与输出，在相同容量和设备预算下，相对最强顺序预取基线降低同步 recall 或 stall。当前证据未达到此门槛。

## 下一步

1. 查明 DeepSeek trace 的真实请求长度、CP padding 和位置归属，保留旧失败记录；修复后在新运行中复核原始 trace。
2. 用封存提示完成 sidecar 和四组 CPU 回放，再单独处理输出及 selected-set 稳定性。
3. 为 ECHO 原型取得原生完整分数和真实异步 KV 搬运路径，按固定预算及独立 episode 进行在线对照。

运行方法见 [DeepSeek Pod 手册](deepseek-pod-runbook.md)、[GLM 单卡手册](glm-echo-single-npu-online.md) 和 [研究合同](research-contract.md)。
