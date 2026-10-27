# Issue 6：无八卡条件下的理论分析与 CPU 模拟

日期：2026-10-22。对象：[issue #6](https://github.com/intellistream/memory-aware-sparse-kv/issues/6)。

本次交付的是可执行的 KV 预取离散事件模拟器、参数扫描和旧原生 trace 回放。可以验证因果性、容量约束、搬运守恒，以及给定输入和时序假设下的收益条件；不能产生 DeepSeek 的真实原生选择，也不能代替真实 Agent 泛化、张量正确性或八卡在线性能验收。

## 开源工具调研与选择

以下按 2026-10-22 查阅的官方仓库及文档判断，未声称穷尽所有模拟器。

| 工具 | 适合回答的问题 | 对本 issue 的缺口与使用决定 |
| --- | --- | --- |
| [Vidur](https://github.com/microsoft/vidur)，MIT | LLM serving 调度、部署容量、请求延迟 | [新增模型需要 profiling](https://github.com/microsoft/vidur/blob/main/docs/profiling.md)，官方说明 profiling 后可 CPU 模拟。未找到本项目 DeepSeek-V4/Ascend 的已校准配置或逐 query 原生稀疏 KV 预取接口。后续适合承接请求级调度。不能把 A100/H100 配置当成当前 NPU 实测。 |
| [SimAI](https://github.com/aliyun/SimAI) | 集合通信、网络拓扑、计算通信重叠；也有多请求 inference 组合 | 适合后续细化八卡通信。当前需求首先缺的是选集 trace 与 Host–设备预取缓存逻辑，接入完整通信栈不会补出这些证据。 |
| [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass)，BSD-3-Clause | 推理算子、硬件结构、设计空间分析 | [硬件配置接口](https://github.com/PrincetonUniversity/LLMCompass/blob/main/docs/run.md)可用于未来估算计算阶段；仍需新增 V4 稀疏路径与 Ascend 校准。 |
| [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) | 无 GPU 的 OpenAI/vLLM 接口、路由、流式响应和延迟测试；有 prefix KV cache | 官方描述响应为 echo、随机文本或数据集响应。适合验证 Agent 客户端工程，不能证明模型自主工具决策或 attention 选集。 |
| [SimPy](https://simpy.readthedocs.io/en/stable/)，MIT | 通用离散事件和共享资源争用 | 本次实际使用 `simpy==4.1.1`。保留项目预测器和缓存规则，仅新增时序与争用层，便于解析校验和以后替换硬件参数。 |

结论：有可复用的开源底座，没有在本次查阅范围内找到能够直接完成全部 issue 验证的现成配置。采用 SimPy 不意味着已经模拟了 DeepSeek 网络本身。

## 理论收益条件

设每个 query 的原生选集在起点后 `L` 微秒就绪；信号可见延迟为 `d`，完整预测开销为 `c`；预取缺失单元数为 `n_p`，预取后仍需同步回载的数量为 `n_m`。每单元 `s` 字节，带宽 `B` GB/s（十进制），每批最多 `g` 单元，批启动延迟 `a` 微秒：

```text
T(n) = ceil(n/g) * a + n*s/(1000*B),  T(0)=0
W_prefetch = max(0, d+c+T(n_p)-L) + T(n_m)
W_baseline = T(n_0)
单 lane query 服务时间 = L + W + A
获益条件：T(n_0)-T(n_m) > max(0, d+c+T(n_p)-L)
```

这里 `A` 为 attention 计算时间，公式适用于无争用、整批预取完成后才处理 demand 的本次调度策略。`n_m` 由真实缓存状态决定，包含错误预取造成的污染，不能直接用候选覆盖率代替。多流共享 DMA 时必须再计算排队。

解析例子（不是实测）：冷缓存需搬 512 个 C4 单元，16 GB/s、每 128 单元启动 5 us，则搬运为 `52.768 us`。若完美预测花 5 us、提前量 50 us、attention 20 us，无预取服务时间为 `122.768 us`，预取为 `77.768 us`，减少约 36.65%。若没有提前量且预测花 50 us，即使 100% 命中也从 `72.768 us` 变成 `122.768 us`。因此准确率本身不足以决定是否部署。

- `P=0` 应退化为无预取。
- 完美且足够早的预取可以隐藏回载；完美但晚到的预取也可能变慢。
- 缓存已有选中 KV 时，重复预测命中不产生新的节省。
- 预取用错字节、批粒度或预算会改变启动延迟和污染程度。
- 观察上一 query 与当前 indexer scores 的可见时刻不同。本实现把 ECHO 完整 score 的最早可见点保守设为 `L`，不假定免费提前获得。
- 增加上下文长度、减少相邻选集重叠、发生热点切换、加入共享链路竞争，均应独立做敏感性分析。

## 模拟边界与实现

入口为 `m0a/issue6_simulator.py`，研究矩阵为 `m0a/run_issue6_simulation_study.py`。

1. 沿用 `Context → predict → Observation` 接口，普通预测器无法读取当前选集；oracle 单列且不参与训练。
2. 沿用共同 LRU，固定每 request/rank/layer 的容量和预算，预取目的槽在提交时即占容量；只对不驻留数据搬运。每个 lane 从空缓存开始。
3. SimPy FIFO 链路同一时刻只传输一个 batch；一个 query 等待自己的整个预取批次排空后，再同步召回缺失选集，然后执行 attention。无预取取消、抢占或跨 query 悬挂传输。这是明确的保守实现选择，不是所有异步预取实现的性能下界或上界。
4. 单批可由多个逻辑块组成，按 `ceil(n/g)` 收取启动开销；不模拟物理地址连续性、PCIe 包、Host gather、cacheline 或页放大。相关成本应通过实测 `a/B/g` 或后续页级模型补齐。
5. C4 单元坐标和生成边界复用协议 v2，`s=1024`，每四 token 生成一个新单元；W8A8 权重量化不自动意味着 KV 为 8 bit。仅计 C4 KV，不计 SWA、C128、indexer、compressor、权重和 runtime。
6. 共享 DMA 的并发实验是相同输入的独立 lane 副本，拥有独立缓存和 CPU worker。不是 TP rank，也不是整个模型的层调度。副本不增加独立样本数；输出是每层 query 服务时间，不叫 TPOT 或 TTFT。
7. `prediction_us` 包含编码、检索、索引维护等的摊销假设；不把离线 Python 运行时当成设备可部署开销。全参数是扫描值，没有用测试集挑出的结果冒充实测值。
8. 每次输出带代码哈希、输入哈希、训练任务列表、拟合参数、事件时间线、逐任务 CSV 和 task-cluster bootstrap。区间只刻画给定时序参数和输入下的任务差异，不覆盖硬件模型误差。

## 实验矩阵与数据来源

**合成 C4 压力实验**：种子 7/19/31；每种子 12 个任务，按任务固定 4/4/4 train/validation/test；32 连续 decode query；4096 个初始 C4 单元（16384 token）；top-k 512；容量 512/1024 单元；预算 128/512。比较 16 种已实现策略，selected-only 数据不提供 ECHO。

四种场景：稳定热点、每 8 query 热点切换、独立随机选集、误导语义提示。热点场景从四分之一区域抽取约 80% 选集，其余为随机干扰。区域文本和事件标签是人为构造，事件模型在训练任务拟合；误导场景的事件模型也可能学会系统性标签关系，不能将它描述为未见事件测试。它们不能证明任务 11–17 的自然语义泛化。

合成事件在同一 namespace 内改变提示/热点，仅用于测试统计重置与适应；不等同于实际工具返回造成的 prompt/context version 变化。`boundary_wait_us` 是每阶段前四 query（包含首次冷启动）的均值，`steady_wait_us` 是其余 query；两者差值不能单独解释为事件的因果影响。

**时序扫描**：带宽 8/16/32 GB/s，提前量 0/10/50/200 us，预测成本 0/5/50 us；批启动 5 us、批粒度 128 单元、attention 20 us。另测 2/4 条独立流共享 DMA、2048/16384 个初始单元的上下文压力。所有时间都未作 NPU 校准。

**旧 GLM 真 trace 回放**：使用既有 `issue6-native-20261020-r02/trace.jsonl.gz`，校验独立审计、manifest 与 trace 哈希，完整保留 6696 行作为训练/验证/测试来源。只评估既定 test tasks，保留原生单位（1152 字节）和模型身份；不将它改名为 DeepSeek。容量 4096、预算 512；提前量 0/50 us；9 种策略含 ECHO，时间仍为模拟假设。原始 trace 是受控续写，不是自主 Agent 运行。

## 本次完成结果

远端 `study-r01` 全部完成：18 组运行、988 条配置×策略汇总记录（不是 988 个独立样本）；13 项单元测试通过。独立输出审计通过：314 个文件哈希、3952 行逐任务指标、228 个事件日志、7799 条已保存 query 时间线。CLI 零预算冒烟测试也通过。结果完整备份到本地 `D:/codex/memkv/results/issue6-simulation-20261022/`，包括 `study-r01/RESULTS.md`、JSON/CSV、三张图和环境/测试日志。

以下为合成热点切换场景，合并三种子共 12 个合成测试任务；容量/预算均 512 单元，带宽 16 GB/s、提前量 50 us、完整预测成本 5 us。正数代表单层 query 服务时间减少：

| 预测器 | 候选覆盖 | 时间减少，95% 条件区间 | 总搬运 KiB/query |
| --- | ---: | ---: | ---: |
| 无预取 | — | 基准 108.17 us/query | 352.2 |
| 上一选集 | 31.2% | 0.0% | 352.2 |
| 全局频率 | 14.2% | -11.5% [-11.7%, -11.3%] | 814.7 |
| 当前阶段频率 | 34.1% | 0.4% [0.4%, 0.5%] | 469.6 |
| 事件+历史 | 41.5% | 2.0% [1.9%, 2.1%] | 597.9 |
| 自适应组合 | 32.7% | -2.3% [-2.9%, -1.8%] | 729.1 |
| 诊断 oracle | 约 100% | 22.9% [22.7%, 23.0%] | 468.4 |

这些结果支持以下**模拟机制结论**：

- 上一选集的候选命中大量落在本来驻留的 KV 上，所以覆盖不等于有效提前搬运。
- 热点切换时，全局频率历史可能过时；当前阶段统计在本构造下更稳，但改善很小且增加搬运。
- 现有自适应组合没有稳定压过简单基线：稳定热点约 +0.6%，切换 -2.3%，随机 -1.5%，误导提示 -4.0%。没有为了得到正收益删除负结果。
- 误导提示使纯 LSA 方案在该构造下慢约 19.3%；不能推广为真实语义检索普遍失效。
- 旧 GLM 真选集回放中，自适应组合在假设提前量 50 us 时约 +1.6%，提前量为 0 时约 -21.2%。这验证了时序假设会改变结论，不是实际推理加速。
- ECHO 当前完整分数到达时刻按保守模型设为原生选集就绪；此设定下的负结果不能证明 ECHO 在线实现或 Agent 场景失效。

下一步方法优先级应是：在预算内优先选取预测为需要且当前不驻留的 KV，结合阶段统计，并增加按预计净等待收益决定是否预取的准入判断。正式比较仍应固定同一套缓存、预算和时序口径；不能把本次人工区域标签的事件收益视为真实 Agent 事件学习成功。

## 复现

远端独立目录：`/root/experiments/issue6-simulation-20261022/`。模型共享挂载只读。未改变 `/root/experiments/memkv-env-89df7fad/code` 或已有 runtime。

从 `code/` 目录执行（新输出目录不可已存在）：

```bash
../venv/bin/python -m unittest m0a.test_issue6_simulator -v
../venv/bin/python -m m0a.issue6_simulator \
  --scenario shift --capacity 512 --budget 512 \
  --lead-us 0 --lead-us 50 --prediction-us 5 --output ../quick-NEW

../venv/bin/python -u -m m0a.run_issue6_simulation_study \
  --output ../study-NEW \
  --native-trace /root/experiments/issue6-tasks1-10-20261020/code/m0a/runs/issue6-native-20261020-r02/trace.jsonl.gz \
  --native-audit /root/experiments/issue6-tasks1-10-20261020/independent-native-audit.json \
  --model-config /models/DeepSeek-V4-Flash-W8A8/config.json
../venv/bin/python -m m0a.summarize_issue6_simulation ../study-NEW
../venv/bin/python -m m0a.audit_issue6_simulation ../study-NEW
```

依赖版本在 `m0a/issue6-simulation-requirements.txt`；安装日志和完整 freeze 随结果留档。当前 venv 继承主机系统包，安装输出中的既有服务栈依赖冲突也留档；本次仅导入 NumPy、SimPy 和 Matplotlib，不运行服务栈。

## 对 issue 完成状态的影响

| 任务 | 本次提供的证据 | 仍需真机/真实数据 |
| --- | --- | --- |
| 1、5–10 | 因果接口、统一容量预算、机制对照、消融和时序成本模型；旧 GLM trace 条件回放 | DeepSeek 自主 Agent 真实选集上的方法收益 |
| 2–4 | 复用已有受控原生 GLM 数据，审计输入；不新增模型运行 | DeepSeek 工具闭环、数据有效性、八卡原生采集与稳定性 |
| 11–17 | 合成热点、误导提示、随机需求、长度和事件边界压力测试 | 真实任务/模板隔离、同义改写、需求变体、新实体工具、跨域和真实事件 |
| 18–20 | 抽象容量、DMA 重叠、就绪顺序和争用的模拟验证 | 真实 Host–设备缓存、张量与输出一致性、无完整 shadow、TTFT/TPOT、全机显存 |
| 21 | 可复跑代码、报告、图表和日志索引 | 最终在线结果后再汇总完整验收 |

有卡后先测本系统的分块 DMA 延迟/带宽、预测全开销、信号就绪和 attention 时间；用独立数据校准模型，再用另一组实测 trace 验证误差。之后才讨论全模型调度、TP 通信和端到端收益。当前不应关闭 issue #6。
