# 单卡 GLM 自动工程验证

本页保留旧 Docker 工作区的历史运行说明。重建后的 Kubernetes Pod 未核实这些路径或服务管理方式；不要在新 Pod 上直接执行下面的 `--execute` 命令。

执行入口：

```bash
python3 scripts/launch_single_npu_validation.py --device 0 --execute
```

省略 `--execute` 只打印计划，不创建目录或连接服务器。仅允许设备 0；默认通过已配置的 `hust` SSH 主机连接 `/workspace/memecho`。本地和服务器每次使用新的 UTC 时间加随机后缀运行 ID，代码部署到运行目录的 `implementation/`，保留服务器原代码、输入和现有容器。

启动入口验证部署代码哈希，创建脱离终端会话的服务器 worker 和本地同步监护进程。只有确认两个 PID 存活、服务器 heartbeat 新鲜、`00-bootstrap` 的四个文件 SHA-256 校验通过后，才写出 `startup-evidence.json` 并返回。此时实验仍在后台运行。

运行目录是 `m0a/runs/single_npu_<UTC>_<random>/`。本地查看 `status.json`、`sync-status.json`、`sync.log`；服务器查看同名目录中的 `worker.log`。最终结果写入 `report.json` 和 `replay-{64,128}mib/report.json`。`startup-evidence.json` 证明启动成功；`status.json` 的 `passed/failed` 才是实验结果，`sync-status.json` 表明最终同步是否成功。

流程依次为启动同步、NPU/端口/模型/tokenizer/镜像检查、输入补足、trace-off 48 请求、trace-on 48 请求、NPU 释放、CPU 回放和报告。每一阶段关闭文件后发布明确文件清单，服务器必须收到本地校验 ACK 才进入下一阶段。同步失败最多三次，通知服务器停止；失联时服务器同步等待上限和总时限也会触发清理。服务健康和单次请求上限分别为 900 秒，整个服务器 worker 上限六小时，heartbeat 每 15 秒更新。

等长的 12 对原始输入使用相同中性后缀补足边界后 32-token 窗口，重新验证共同前缀、严格等长和完整 token 哈希。每个 API 响应保留准确 prompt/generated token IDs，并校验重复稳定性和 trace-on/off 输出签名。所有采集行校验因果范围；回放输入仅保留每个请求的边界前一位置及之后完整 32 个位置，要求每个位置六层（3 computed + 3 reused）。原始全局采集范围的额外合法行保留在 `raw-traces/`。

Sidecar 使用准确 API token IDs 导出逻辑前缀快照、版本、全部普通 KV 库存、region 和原始 pair/episode/source 身份。无精细语义标注的 region 为 `other`。合成输入保持 `synthetic: true`，选择来源另标注为 `real_npu_native`；逻辑库存不代表在线 KV 恢复。8K 组训练、32K 组工程评估，成对分支与重复同组；这一合成划分不是任务泛化证据。

固定 BF16 MLA 每层每 native KV attention payload 为 `(512 + 64) × 2 = 1152` 字节。运行记录镜像内 MLA/SFA 源码、SHA-256 和行号，布局或固定模型配置不符即失败。该字节模型不包括 indexer keys/scales、分配器元数据或 DMA 开销。64/128 MiB 容量和每窗口 8 MiB 预取预算应用于全部七种策略与 native/page128 两种粒度。每次模拟 attention 前原生 selected KV 全部就绪；报告保留 precision/recall、有效和浪费字节、同步 recall、污染、规划 CPU 时间和分组配对区间。

容器名包含运行 ID，并带 `memecho.validation.run` 归属标签；清理必须匹配标签，只停止自己的容器。NPU 请求结束后立即释放容器，再执行 CPU 校验和回放。失败保留响应、日志和资源快照，并由监护进程收集最终文件清单。

本地工作区缺少可用 Git 时，在临时独立仓库对准确代码快照创建 commit 与 `implementation.bundle`。任务结束后，监护进程另外提交最终报告、代码和采集校验清单，生成 `local-archive/final-artifacts.bundle`，同步回服务器并重新验证最终文件清单。不需要 GitHub URL。

验证命令：

```bash
python3 -m unittest discover -s m0a -p 'test_*.py'
bash -n m0a/serve_glm53_tiny_trace.sh
```
