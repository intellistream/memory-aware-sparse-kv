# issue6 CPU 验收记录（2026-10-27）

本轮按用户要求先完成现有 CPU 资源可验证的部分。对应 [issue6](https://github.com/intellistream/memory-aware-sparse-kv/issues/6)，原文快照见 `issue6-requirements-snapshot-20261027.json`。**本轮完成 CPU 机制与数值验证；issue6 整体验收仍未完成。**

## 运行来源

- 全量回归针对下述固定归档加本轮补丁，不包括本地原有、尚未提交的 Agent 集成修改。
- 基线提交：`65f5af11565123b5ea44d3e6ba8796fa69a73c2d`，从已封存的 Git archive 恢复后，在独立目录修改。
- 远端目录：`/root/experiments/issue6-cpu-acceptance-20261027/`；原固定环境和历史实验结果保持原状。
- 模型：`/models/DeepSeek-V4-Flash-W8A8`，只读。索引列出 70 个分片，均存在；本轮加载的 C4 组件涉及 26 个实际权重张量，214.85 MiB，并保存内容凭据。分片存在性检查不代表全部权重已读取或哈希。
- 资源预检：16 核 CPU 配额、64 GiB 内存；没有 `/dev/davinciN`，NPU 健康检查超时。因此没有原生八卡推理。
- 环境：Python 3.12、torch 2.10.0+cpu、NumPy 1.26.4、SimPy 4.1.1、safetensors 0.8.0、matplotlib 3.9.4。数值实验设置 torch 8 线程；CPU 墙钟耗时不作为设备时延。
- 所有指标的来源、参数、原始结果及源码哈希在 `evidence/` 中；历史回放输入另封存于 `inputs/prior-cpu-run-r01/`。

## 本轮修改与验证

1. **跨请求身份与版本安全**：rank 组必须属于相同 task/run/request/context/layer/phase/query/token position；被替代的 context version 禁止再次激活。所有 H2D/D2H 事件保留完整 namespace 和调度 step，审计使用 rank + step，避免不同请求的 query 0 混淆。
2. **成本配置约束**：实际 replay 再检查模型、设备标签和并行 rank 数，避免初始化时匹配但回放时套用错误成本曲线。现有曲线仍为假设，未获得 Ascend 实测校准。
3. **数据契约**：根轨迹、模板标识及可选实体通过传递闭包分组；组不可为凑比例而拆散。新增改写事实/评估/工具契约、不同需求的成功条件、新实体映射，以及五类事件的因果性与版本检查。契约通过不能证明语义等价或任务可完成。
4. **长上下文组件验证**：16K/32K × 稳定/切换，四组都使用相同种子和 16 query；真实 checkpoint C4 layer 2、原生数学形式的 top-512、FP32 参考累加、BF16 KV。
5. **可复现入口**：新增资源预检、数据审计、长上下文 driver 和一键 CPU 验收脚本；保留失败日志和硬件用例跳过原因。

新增边界测试 6 项、数据/事件/模型清单契约测试 12 项通过。其中随机调度测试为 40 个固定种子 × 2 rank × 12 query，共 960 次 payload 回调，覆盖不同块大小、时机、链路共享和容量约束；这些不是独立 Agent 任务。

完整回归：282 项中 278 项通过、4 项跳过、无失败；随后在固定 τ³ 环境补跑其中 2 项并通过。合计 280 个不同用例已执行通过，余下 NPU 专用检查及仓库外 Stage 0 测量复算未验证。补跑是依赖/消息接口测试，不是一次自主 Agent 任务实验。

### 真实权重长上下文结果

| 输入长度 | 受控条件 | query 数 | 稀疏 gather 与 dense-mask 最大绝对差 | 四种缓存策略取回后的输出误差 |
| --- | --- | ---: | ---: | ---: |
| 16,384 | stable | 16 | 4.0531158447e-6 | 0 |
| 16,384 | shift | 16 | 4.0531158447e-6 | 0 |
| 32,768 | stable | 16 | 3.5762786865e-6 | 0 |
| 32,768 | shift | 16 | 3.3378601074e-6 | 0 |

共 64 次 dense-mask/gather 数值对照，256 次缓存取回输出对照，全部通过已有容差和精确相等检查。首个 query 的已完成历史 C4 单元分别为 4096/8192，top-512 约占 12.5%/6.25%。没有用缩小维度或缩小 top-k 来降低难度。

输入来自受控层隐藏状态，不是 tokenizer 后的完整模型推理。stable/shift 表示隐藏状态原型条件，不能称作真实工具返回、记忆写入或 Agent 任务切换。此实验仅增加长上下文数值证据，不构成任务 16 的自然语言干扰泛化验收。四种缓存策略的既有顺序回放独立检查真实张量；其附带的旧版时延字段不能与下述异步调度指标混为一谈。

### 异步模拟与独立审计

在修复后的实现上重新执行 2 条受控真实权重 trace × 6 种系统配置 × 5 种策略，共 60 组。另有无预取、全局频率各 8 query 的 BF16 payload 回调，输出与归档 CPU 基线完全一致，最大绝对差 0，并核对 26 个实际权重张量凭据。

独立审计核对 62 个结果文件、8656 条 rank-query、44035 次传输、22388 条 payload 状态转移和 121 个 Python 源码哈希，全部通过。新增审计要求每次传输只能匹配一个 rank + step，且 namespace 一致。

这些是单层配置扫描，时间仍来自假设；八个逻辑 rank 使用选集副本，没有八卡设备运行。报告 `mean_group_service_us`，不报告实测 TTFT/TPOT。oracle 使用当前标签且仍受缓存、分块及链路约束，只作诊断，不代表数学最优调度。

## 211 个原始任务的数据审计

读取固定 τ³ 源码中的银行 97、零售 114 个作者原始任务。源文件 SHA256 随报告保存，没有生成伪 Agent trace，也没有改写原任务事实。

初步的动作名/参数字段结构作为模板代理，连同根任务和提取到的实体形成 84 个连通分组；确定性划分为训练 127、验证 42、测试 42：

| 域 | 总任务 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: | ---: |
| banking_knowledge | 97 | 52 | 36 | 9 |
| retail | 114 | 75 | 6 | 33 |

按这些已标注关系检查无泄漏，组不会为平衡样本数而拆开。**动作结构不是经审核的业务语义模板，实体提取也只覆盖已实现字段**；这不是对所有模板和实体无泄漏的证明。各域划分并不均衡，需在正式实验前审核任务族、实体覆盖和测试规模，不应为了好看而事后拆组。

同时提取表达相近但 evaluation 不同的现有任务候选对（最多 30 对）。这里只是机械候选筛选，不证明它们是合格的“相似措辞、不同需求”变体，也没有验证可完成性。改写/新实体/事件函数当前由单元夹具验证；`observed_agent` 只检查证据引用存在，引用内容及真实执行仍须额外审计。上述事实全部写成 JSON 中的显式 false/限制字段。

## 与真实八卡运行的区别

完整机制说明继续见 [模拟器指南](issue6-simulator-guide.md)。当前路径为：

`实际 attention 权重 + 受控隐藏状态 → CPU indexer/压缩/attention → 固定最终选集 → 因果预测器 → SimPy 缓存/链路/逻辑 rank → 就绪检查与可选真实 BF16 attention 回调`。

- 数值层检查稀疏选择、压缩、取回后 attention 输出；不执行完整 43 层、mHC、MoE、logits、采样和工具调用。
- 调度层明确 absent/reserved/inflight/ready、容量占位、淘汰、需求召回、本地新生 KV 和写穿 D2H；状态存在于 CPU，实际 HBM 地址、pinned memory、DMA stream 和 HCCL 尚未验证。
- 时间层按依赖与链路队列推进，不用 Python 运行耗时代替 NPU 成本；CPU 线程池、Host pack/内存带宽竞争和完整 MoE 通信图仍缺失。
- 设备账本是显式假设下的容量估算；逻辑上不留 full shadow，不等于在线设备代码已经移除完整 KV shadow。
- 只直接依赖 SimPy；没有把 Vidur、SimAI、ASTRA-sim 或 LLMCompass 的现成结果当成 V4/Ascend 校准。现有开源方案的采用边界见指南。

## issue6 逐项验收矩阵

以下“已有/本轮”表示工程证据范围，不是把对应原待办标为全部完成。

| 编号 | CPU 可核对的证据 | 仍缺少的正式证据 |
| --- | --- | --- |
| 1 | C4 协议、因果接口、容量/预算及跨请求版本边界 | 实机采集位置和信号提前量 |
| 2 | 已有 Agent/服务接口测试；本轮未新增 Agent 运行 | GLM 工具闭环与模型能力验收；DeepSeek 自主运行 |
| 3 | 211 原始任务清单、分组与变体机械契约 | 自然对照、业务合法性、可完成性和任务族审核 |
| 4 | 原始记录校验；真实权重 CPU 选集和可复算结果 | 原生 NPU trace、重复稳定性及采集位置对齐 |
| 5 | 相同容量/预算下 none/last/history/oracle 模拟 | 真实 Agent 上公平 ECHO 风格对照；当前 selected-only 回放未认证 faithful ECHO |
| 6 | 历史策略、因果测试、受控 stable/shift 回放 | 真实任务热点持续性和切换效果 |
| 7 | 已有语义接口/映射测试 | 真实片段检索覆盖率、编码/维护/检索成本 |
| 8 | 事件接口及本轮因果/来源契约 | 真实事件在历史之外的增益 |
| 9 | 已有组合预算与完成反馈因果测试 | 真实阶段收益和切换成本 |
| 10 | 数值/调度审计、等待和完整流量账本、假设性能图 | 独立真实任务的区间、信号消融与测量开销 |
| 11 | 根轨迹/结构模板代理/实体传递闭包与不泄漏审计 | 人工确认模板族、未见任务预测效果 |
| 12 | 事实/评估/工具不变的机械检查 | 语义等价审核、各自原生选集和运行效果 |
| 13 | 原始相近措辞/不同评估候选对及成功条件检查 | 需求差异、合法性和可完成性确认，原生 trace |
| 14 | 实体保留分组和一一映射检查 | 完整实体覆盖、未见工具行为/名称分离测试 |
| 15 | 两域独立清单与固定划分 | 参数冻结后的银行→零售、零售→银行真实零样本测试 |
| 16 | 16K/32K × stable/shift 真实权重组件数值对照 | 相关信息距离、自然无关历史和任务可完成性 |
| 17 | 五类注入事件契约与失效版本拒绝 | 逐事件真实运行结果；不得把注入事件算真实覆盖 |
| 18 | 异步状态机、共享链路、屏障、容量/流量审计 | 真实 Host—设备管理、重叠测量、无 full shadow 性能路径 |
| 19 | 256 次长上下文取回输出 + 16 次异步 payload 输出对照 | 原生选集、KV、全模型输出/行为一致；隔离校验开销 |
| 20 | 60 组单层模拟场景 | 八卡完整 TTFT/TPOT、实际内存与两域在线泛化 |
| 21 | 本报告、逐项矩阵、哈希、日志和一键入口 | 汇总上述原生实验后的最终报告与演示 |

## 复现与文件索引

从仓库运行，输出必须使用新目录：

```bash
TAU3_PYTHON=/root/experiments/memkv-env-89df7fad/tau3/bin/python \
bash scripts/run_issue6_cpu_acceptance.sh \
  /root/experiments/issue6-simulation-20261022/venv/bin/python \
  /root/experiments/memkv-env-89df7fad/tau3-source \
  /root/experiments/issue6-cpu-acceptance-20261027/inputs/prior-cpu-run-r01 \
  /root/experiments/issue6-cpu-acceptance-20261027/reproduction-NEW
```

`MODEL_DIR` 默认指向约定的只读模型目录。全量测试兼容项目包导入和旧顶层导入；无 NPU 用例应显式 skip。新输出不可覆盖既有证据。一键入口中的各阶段已分别实际执行，脚本另经 `bash -n` 检查；没有额外声称整段 shell 已再次运行。

| 路径（相对远端实验根目录） | 内容 |
| --- | --- |
| `evidence/readiness.json` | 设备/配额/依赖/模型索引预检 |
| `evidence/full-tests-r03.log` | 最终完整回归记录；此前失败的 full-tests.log / full-tests-r02.log 也保留 |
| `evidence/tau3-supplemental-tests.log` | 固定 τ³ 环境补跑两项依赖测试 |
| `evidence/async-boundaries.log`、`contracts-tests.log` | 新边界和数据契约专项测试 |
| `evidence/benchmark-split-audit.json` | 211 个原任务、源哈希、分组、划分和候选对 |
| `evidence/long-context-r01/manifest.json` | 四组数值实验、权重与源码凭据 |
| `evidence/async-study-r01/manifest.json`、`audit.json` | 配置、回调对照和独立审计 |
| `evidence/async-study-r01/*.png` | 带模拟范围标注的服务时间和 rank 时间线图 |
| `evidence/acceptance-summary.json` | 本轮机器可读摘要及未完成范围 |
| `SHA256SUMS.json` | 打包时各文件哈希 |
| `changes.patch` | 相对于 65f5af1 源码归档的本轮补丁 |

完整回归第一次使用 unittest discover，因本目录是命名空间包而未启动；第二次按模块加载发现旧顶层导入缺少路径，195 个测试中出现 5 个导入错误和 3 个 skip。最终入口同时配置仓库根及 m0a 的 PYTHONPATH。失败记录保留，不通过删测试或放宽数值阈值消除失败。
