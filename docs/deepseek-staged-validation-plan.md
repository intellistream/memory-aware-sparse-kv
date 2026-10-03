# DeepSeek 分层工程验收计划

## 背景与目标

`deepseek_20260930T034055Z_dfe4add7` 的八种配置均未通过 20 次重复输出诊断。最后两种配置各有 4 次差异，首次差异位于生成 token 索引 20。隔离的 20 次算子观察仍有输出差异，但未形成可独立重放的“同输入、异输出”算子样本。真实 8-token compressor 缓存合同通过；原服务已恢复健康，最终 467 个文件通过 SHA-256 校验。该运行的报告和证据保存在 `m0a/runs/deepseek_20260930T034055Z_dfe4add7/`。

下一轮以**原部署命令和配置**独立验证原生 trace、sidecar 与四组 CPU 回放。严格输出重复性维持“未通过”的独立结论；分层工程验收通过不代表原定 48+48 输出稳定性通过，也不证明在线 KV 恢复、DMA stall 或 offload 收益。

## 实现变更

1. 启动入口增加 `--validation-mode trace-replay`；默认仍为当前严格模式。新模式与 `--repair`、`--diagnostic-only` 互斥，默认仅打印计划，仍须 `--execute` 才运行。保存原命令、镜像/源码哈希、模式与上述失败 run ID。实验容器使用原部署命令、环境及八卡配置，trace-off/on 仅采集 hook 有别。
2. 新模式在既有 preflight、输入 token 校验和真实 compressor 合同通过后，跳过候选配置筛选，分别执行 48 次 trace-off 和 48 次 trace-on，沿用原有 12 对输入、并发 1 和 32-token 窗口。请求失败、响应结构错误、实际 prompt token IDs 不一致或次数不足仍是硬失败。重复输出和 off/on 输出差异逐项记录首个不同 token、完整请求身份与原始响应，但不阻断采集；与硬失败分开保存。
3. 原生 trace 的硬门槛保持：48 个唯一请求；每请求边界前 1 位及后 32 位、21 个 C4 层的完整 CP 查询归属；rank/chunk 元数据、512-wide selected set、唯一行、因果范围、准确模型及输入关联。跨请求 selected set 和配对前缀相等性改为诊断指标；不一致时配对效果分析标记为探索性，不能据此宣称策略收益。
4. sidecar 从**每个 trace-on 请求自己的准确 prompt token IDs 与原生 selected set**构建，不使用 trace-off 的输出作为快照身份。保持现有源证据、C4 支持区间和真实 NPU provenance。完成 per-rank/aggregate × 64/128 MiB 四组回放，继续硬性检查 sidecar 完整性、容量、就绪与污染等每条轨迹的工程约束。
5. 成功状态单列为 `engineering_validated`，报告给出 `validation_mode=trace-replay`、严格输出验收 `not_qualified`、输出差异计数和 selected-set 一致性计数。同步监护进程仅在原服务恢复健康、原配置一致、最终 SHA-256、commit 和 bundle 完成后接受该状态。失败路径保留证据并恢复原服务。

## 验证与验收

- 单元测试覆盖模式解析及互斥、严格模式行为不变、输出差异仅记录、请求/输入错误仍失败、selected-set 重复差异仅降级配对结论，以及缺行、CP 错配、越界 ID 继续失败。
- 八卡隔离运行完成 48+48 请求、原生记录、sidecar 和四组回放；分别核验产物完整性与具体报告状态。原服务恢复后检查 `/health`、`/v1/models`、八卡和容器配置，再核对最终同步哈希与证据包。
- 若 trace hook 导致输出差异，报告观察到的差异，不据此推断 hook 的因果影响；若原生记录或回放任一硬门槛失败，整轮工程验收失败，不通过挑选稳定请求补足。

## 固定条件

模型 revision、W8A8、TP=8、DSA CP、原部署配置和合成输入保持不变。不升级软件栈或直接开启 batch invariance。沿用 15 秒心跳、900 秒单次等待、三次同步重试、每 run 六小时上限及独立恢复预算。所有实验只在隔离容器进行；原服务在回放前恢复。
