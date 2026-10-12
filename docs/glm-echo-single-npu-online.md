# GLM tiny 单卡 ECHO 风格预取原型

## 范围

本实验只检验单张物理 910B2（设备 6、进程内 `npu:0`）上的简化 GLM 原型。`no_offload` 保持原生稀疏注意力；`offload_no_prefetch` 与 `offload_echo_prefetch` 都从固定的 pinned CPU KV 源向每层 3,072 token、16 token 页的 NPU scratch 搬运 KV。后者额外运行 256-bin prefill 与 α=0.5 的 decode EMA 候选规则，每次最多预取 512 token。两种搬运模式在每个目标 query 上计算相同的完整候选分数，并与原生 top-k 逐行核对；原生选择决定保底召回和最终注意力索引。

**`prototype_hbm_shadow=true`。** 原生 NPU KV 缓存仍供 indexer 和正确性参照使用，故此实验不测量 HBM 节省。Python adapter 在当前 query 已有完整分数之后，按先预取、再保底召回的顺序执行，未与 indexer 或注意力异步重叠；命中只表示该 query 的预测 token 被原生选择。本实验测的是这种在线顺序执行原型的实际开销，不能代表论文系统或 DeepSeek 的性能。

## 固定输入与配置

- 模型：`inference-optimization/GLM-5.3-0.6B-A0.4B`，revision `20aae340d157bd2215b9d81165be87e94686dfdf`。`config.json` SHA-256 `09d6d0b4b1131cfc79e280bd71a38f2700b433d868d34d02e92c5e56900d2cdd`，`tokenizer.json` `44556920e4507367e1615610a11aabf215ed123be7ec45ff217891b1b5a6a21a`，`model.safetensors` `51bba6b768ddcdb97b7595d0b9a3c18cb82c4f8efa77e2f7924c4d1d46cb3aed`。
- 输入来自封存的 τ³ 48 个 event 提示，零售和银行各 24 个独立 episode。源 pair 文件 SHA-256 `3ec0dd5e14a7192aebe294c406670db5237890aa9e6bc116ee61cb56de8afdb7`。[GLM 输入](../m0a/artifacts/glm_tau3_event_cases.json) SHA-256 `6de364b9944bcbf2078568c8a08584dd68fc33ec7751f9244609486a1aafdb20`；[固定续写](../m0a/artifacts/glm_tau3_calibration.json) SHA-256 `1aae432308d41fc7b5eec5b211aa66ca344a207b60166b58d019ecf397360083`；[432 请求顺序](../m0a/artifacts/glm_tau3_432_schedule.json) SHA-256 `27942f06b3ed97e308455143333d70ccf69164722a40482493b7aeb34282f3b7`。请以文件内容重新计算哈希；运行 manifest 也记录输入、代码和模型哈希。
- GLM 模板重新封存了输入 token IDs。为使所有 event 边界后有至少 32 个 prefill token，追加了同一个中性 user 消息，原事件边界不变。这改变了原始提示的尾部，应作为外推限制。长度 7,521–32,650 token，低于 65,536。每个案例先在原生模式采样一次 32 token 续写，随后所有模式用 vLLM logits processor 强制相同的 32 token。
- 三模式各三轮，固定 TP=1、BF16、单并发、eager、block size 128、关闭 prefix caching。每轮同一案例顺序，模式顺序轮换；三次预热不计时。每请求记录所选集合完整压缩 trace、输出、哈希、TTFT、TPOT、吞吐、预取/召回字节及正确性标记。worker 上限 12 小时。

## 正确性与计时

在正式计时前，单算子 NPU probe 已检验原生 128-token 页与 scratch 16-token 页的注意力输出一致（该 probe 最大绝对误差 0）。三模式单案例在线烟测的两种搬运模式各核对 189 行完整候选分数、378 行 KV payload 与注意力；所选集合跨三模式的 SHA-256 相同，在线注意力最大绝对误差 0，固定输出 token 一致。运行中每个请求重做上述门槛；任何失败将中断实验。

`prefill_window_ms` 是目标 32 个 prefill query 的 indexer 复算与 adapter/scratch 注意力工作耗时之和，并非全模型 prefill wall time。它可比较两个搬运模式的目标窗口开销；`no_offload` 的该字段为 0，不应用于原生对比。TTFT、TPOT 与 throughput 取自在线 vLLM 请求。完整原生注意力同时运行，供逐行数值核对；这项参照开销计入请求延迟。预取与保底召回当前顺序执行，因此无法推断真正异步融合实现的收益。

## 预注册判定

主差值为 `offload_echo_prefetch` 减 `offload_no_prefetch` 的 `prefill_window_ms` 与 TPOT。按 episode 先平均三次差值，再在每个领域按封存输入长度的三个 8-episode 层做分层 bootstrap，四个领域×指标比较各用 98.75% 双侧区间。某领域至少 20 个 episode 在全部重复、全部模式的原生 selected set 完全一致，才对该稳定子集给受控结论；保留全部漂移并另报探索性全样本。受控区间下界大于零时，才报告该领域该指标有统计支持的方向性下降，并同时报告实际毫秒幅度。输入/续写或分数/KV/注意力门槛失败则没有性能结论。

分析命令（正式运行完成后）：

```bash
cd /root/memecho-echo-code
python3 -m m0a.glm_online_stats \
  --cases /root/echo-glm-calibration.json \
  --measurements /root/echo-glm-formal-<run-id>/measurements.jsonl \
  --output /root/echo-glm-formal-<run-id>/analysis.json
```

此阶段不会对论文系统或 DeepSeek 做性能断言。要检验节省 HBM 的规模效应，仍需在 DeepSeek 上实现并验证不保留原生 KV shadow 的融合路径。

## 2026-10-12 正式运行状态

修正预取候选顺序后，三模式在线烟测 `/root/echo-glm-adapter-smoke-6` 通过所有正确性门槛。两种搬运模式各核对 189 行分数和 378 行 scratch KV/注意力，最大注意力绝对误差为 0；预取模式记录到 6,027,264 字节的本 query 误预取。CPU 全套回归通过 169 项，11 项按环境跳过。

正式运行已由 [启动器](../m0a/launch_glm_formal.sh) 在服务器启动，run ID 为 `20261012T1003Z`，初始 PID 为 `85054`。日志在 `/root/echo-glm-formal-20261012T1003Z.log`；manifest、逐请求 `measurements.jsonl`、压缩 selected-set trace 和最终 `analysis.json` 在 `/root/echo-glm-formal-20261012T1003Z/`。启动时已核验 manifest 的 `formal=true`、`requested_measurements=432`、物理设备 6、输入与调度哈希。**本次正式结果尚未完成，不能据烟测给出性能结论。**
