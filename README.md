# Memory-Aware Sparse KV

研究 agent 状态转移能否提前预测模型原生 sparse attention 选中的 KV，在保持 selected set 和输出不变的条件下减少同步召回。

当前进展见 [定性时间线](docs/experiment-progress.md)。Stage 0 和 M0-A 的采集工具链已验证，合成离线回放已通过工程检查。DeepSeek τ³ 的最新真实运行未通过 native trace 校验；GLM 单卡 shadow-KV 原型完成在线受控实验，但不能代表异步预取或 HBM 节省。研究目标尚未验收。

- `m0a/`：实验代码、测试、prompt 与固定配置。
- `scripts/`：DeepSeek 运行和收尾入口。
- `docs/`：研究合同、运行手册和定性进展。
- `paper/`、`src/`：论文源文件与机制代码。

任务 4 的单卡采集与离线复核见 [GLM 原生 trace 手册](docs/glm-native-trace-audit.md)；
任务 6 的预测接口与 CPU 回放见 [访问历史预取](docs/access-history-prefetch.md)。

生成的测量、日志、分析、结果索引、图表与校准续写保存在被忽略的运行目录或 HUST 归档，不进入本分支的 Git 文件树。已有公共提交不改写，可按提交历史复核早期记录。运行说明见 [DeepSeek Pod 手册](docs/deepseek-pod-runbook.md) 和 [GLM 手册](docs/glm-echo-single-npu-online.md)。
