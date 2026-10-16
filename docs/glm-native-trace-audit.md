# GLM 原生选择采集与复核（issue #6，任务 4）

`m0a.run_glm_native_trace` 在既定 GLM tiny 单卡工程模型上采集原生选择，
`m0a.glm_native_trace` 在不加载模型的情况下重新核验归档。该入口用于完善
任务 4 的工程链路，并为任务 6 提供有明确批次边界的输入。任务 2/3 的自主
Agent 与新数据集仍是研究验收的前置条件。

## 捕获位置与因果边界

采集器只读包装 `DeviceOperator.indexer_select_post_process` 和
`execute_sparse_flash_attention_process`：保留前者返回的原始 tensor，
在后者调用原生 attention **之前**读取所接收的 top-k，并逐项比较两者。
不写入 tensor，不替换选集，不重新计算 attention。CPU 拷贝会同步设备，
因此这一运行配置不能用于性能测试。

适配范围固定为已封存 revision 的六层模型、单请求、2048 个原生选择槽。
层 0/1/2 计算 indexer，层 3/4/5 均复用最近的源层 2；归档复核再次比较共享层
与源层。模型配置、tokenizer 和权重须通过既有 SHA256 校验，其他布局拒绝运行。

每个案例保存事件边界前一个 query、边界开始的 32 个 prefill query、
最后一个 prefill query，以及连续 31 个 decode query，共 65 × 6 = 390 行。
32 个输出 token 对应 31 次 decode attention，最后一个输出不会再作为输入。
原始槽位保留 `-1` padding，合法 ID 不得重复或超过当前 query 位置。

每行保存实际 query/key 序列长度、批次起点、批次大小及行偏移。候选空间
用半开区间 `[0, query_start)` 表示当前批次前已经生成的逻辑 KV token；
它不是已实现的 Host 缓存清单。同一 prefill 批次中的相邻 query 虽然有
前驱选集，预测时不能使用这个选集。`predecessor_visible_before_batch`
据真实批次边界确定；首个 prefill 批次的候选空间可以为空。

未捕获窗口之外的前驱明确记为 null，不假装采集了完整 prefill 历史。
`selected_blocks_128` 是原生 token ID 的逻辑页映射，原始 token 选择同时保留。
历史回放只导出连续 decode，按请求/层冷启动，不用采样 prefill 伪造访问历史。

## 运行与独立复算

连接服务器时使用本次资源分配给出的 SSH 入口。先检查挂载的 `/dev/davinci*`，
并在未设置可见性时核验运行时逻辑设备编号；物理设备编号不能直接当作逻辑编号。
以下变量由本次设备检查赋值，不沿用历史值：

```bash
source /root/experiments/memkv-env-89df7fad/activate.sh
cd /path/to/issue6-native-trace-code
export ASCEND_RT_VISIBLE_DEVICES="$RUNTIME_DEVICE"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/m0a:${PYTHONPATH:-}"
CASES=/root/experiments/glm-20261012/inputs/echo-glm-calibration.json
RUN=m0a/runs/native-trace-NEW
python -m m0a.run_glm_native_trace \
  --cases "$CASES" --physical-npu "$PHYSICAL_NPU" --runtime-device "$RUNTIME_DEVICE" \
  --case-id retail_tool_result_8192_0:event \
  --case-id banking_knowledge_tool_result_8192_0:event \
  --repeats 3 --output-dir "$RUN"
python -m m0a.glm_native_trace --run-dir "$RUN" --cases "$CASES" \
  > "${RUN}-independent-audit.json"
python -m m0a.history_replay --trace "$RUN/history-decode.jsonl" \
  --budget-blocks 512 --region-blocks 128 --output-dir "${RUN}-history"
python -m unittest m0a.test_glm_native_trace m0a.test_history_prefetch \
  m0a.test_history_glm_import -v
```

每个案例先重复运行关闭采集的自由贪心输出，再运行开启采集的自由贪心输出，
既检查原生自由生成自身的稳定性，也比较采集开关；固定续写的相同输出不能
替代此检查。首次输出始终保留为对照，不通过丢弃首次请求来消除差异。
随后至少重复两次固定
续写，逐行比较选集和原始顺序。集合稳定但顺序不同单列记录，不丢弃差异样本。
原生自由生成不稳定或采集开关输出不一致也写入 `status=unstable`；完整性或
格式错误使运行失败。旧归档未重复自由生成时，离线结果的该字段为 null，
不推断其稳定性。开启采集
后的贪心原始 trace 也保存，以便检查捕获覆盖范围。

`manifest.json` 记录输入摘要、模型 revision、代码摘要、设备映射、运行库
版本及原生算子源文件摘要。每次运行要求全新目录；已有运行不会被覆盖。
`audit.json` 包含每次请求的 trace 路径、摘要、实际输出及稳定性差异。
`*.jsonl.gz` 保存全部被采样的原生行，`history-decode.jsonl` 只导出 repeat 0
的两域 decode，共 372 行。

离线复核重读封存输入和原始压缩 trace，检查文件摘要、案例/重复次数覆盖、
全部行及共享层、前驱派生信息、稳定性统计、自由贪心比较和历史回放导出。
它不只读取采集器的通过标志。复核输出写在运行目录外，避免改变封存文件集合。
摘要用于发现文件改变，不是对运行来源的密码学证明。

## 解释范围

封存输入来自 DeepSeek 驱动的 Agent 轨迹，当前 GLM 固定续写仍含周期模式；
自由贪心检查只验证本次两个请求的输出一致性，不证明 GLM 能自主完成 benchmark。
全局与阶段历史在这份单阶段 decode 输入中应一致，跨阶段持续性仍需新数据。
回放使用 token 单位、512 候选预算；它不模拟缓存容量、搬运或同步回载。
本入口不证明异步重叠、HBM 节省、端到端加速，也不替代默认 DeepSeek 模型的八卡实验。

运行 trace、日志与结果放在被 Git 忽略的运行目录中，PR 只提交代码、测试和说明。
