# GLM 单卡在线原型运行手册

本实验使用单张 Ascend 910B2 上的 GLM tiny 模型，比较 `no_offload`、`offload_no_prefetch` 和 `offload_echo_prefetch`。后两者使用相同的 CPU KV 来源、NPU scratch 和完整分数核对；预取模式额外预测候选，原生 selected set 始终决定最终需求。原生 NPU KV 缓存保留作 indexer 和正确性参照，因此它是 shadow-KV 原型，没有 HBM 节省证据。具体定性进展见[时间线](experiment-progress.md)。

`m0a/glm_online_run.py` 从外部 `--cases` 读取已校准的输入和固定续写，按案例、重复与轮换模式确定性生成请求顺序。若提供 `--sealed-schedule`，会同时核对外部调度中的输入 SHA-256 和每条请求。运行器从代码中读取固定参数，不读取 Git 中的校准输出。输出目录必须是新的运行目录；如果它位于代码仓库内，只允许使用被 Git 忽略的 `m0a/runs/`。

HUST 的历史输入与模型分别归档在 `/root/experiments/glm-20261012/inputs/` 和 `/root/models/echo-glm-model/`。默认启动方式：

```bash
cd /root/memory-aware-sparse-kv
bash m0a/launch_glm_formal.sh <new-run-id>
```

启动器将新结果写入 `/root/experiments/glm-20261012/runs/<new-run-id>/`。可用 `GLM_CASES`、`GLM_MODEL`、`GLM_RUN_ROOT`、`GLM_SEALED_SCHEDULE` 指定外部路径。`GLM_SEALED_SCHEDULE` 仅用于核对；缺省时调度仍由输入确定性生成。运行目录保存 manifest、实际调度、测量、selected-set trace、分析和完成哈希，不纳入 Git。

解释边界：原型在每个 query 的完整分数可用后顺序执行预取及保底召回；没有与 indexer 或注意力异步重叠。固定续写和提示尾部的中性扩展也限制端到端 agent 外推。历史正式运行的原始数据、哈希及图表位于远端归档；旧 Git 提交可查询原报告。
