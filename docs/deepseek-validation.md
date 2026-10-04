# 八卡 DeepSeek 验证设计与历史记录

当前公开任务的无人值守入口、验收门槛及状态文件见
[Pod 运行手册](deepseek-pod-runbook.md)。执行需要显式传入
`--execute --runtime pod --validation-mode trace-replay --workload tau3_v1.0.1`。
返回码 0 仅证明后台 worker、独立恢复监护和本地同步监护已接管，
实际结论由 `final-status.json` 的独立门槛决定。

公开任务固定于 τ³-bench v1.0.1，银行检索使用离线 BM25。
有 48 对任务链，每对在 trace-off/on 各运行事件与对照的两次重复，
每阶段预期 192 次请求。输出或原生 selected set 不稳定时不通过验收。
CPU 回放只判断顺序预取局部性；缺少 indexer 分数与时序回放时，
ECHO 机制结论明确为未达验收。

以下是旧合成输入和 Docker 实验的历史设计，不代表公开任务运行结果。

下面保留的是旧 Docker 环境的历史执行设计，供后续 Pod 适配参考，不是当前可执行步骤。历史入口会在 `m0a/runs/deepseek_<UTC>_<random>/` 保存代码部署、commit 和 bundle。

先保存当前 `memecho-vllm-ascend` 的身份、完整配置和模型/tokenizer 证据，核验服务健康及完整输入 token IDs，再暂时停止并保留原容器。所有诊断在带任务归属标签的独立容器中进行；实验容器继承环境、挂载、设备、IPC 和通信配置。端口探测采用 `SO_REUSEADDR` 并确认无监听；启动前等待八卡 HBM 和设备进程释放。

原配置下的 `8192_tool_call/event` 曾在相同输入、temperature=0、seed=0 时从首 token 分歧。按累计配置顺序测试：全新原配置、关闭图执行、再关闭 shared-expert multistream、再去掉 MTP、再关闭 async scheduling、最后 max-num-seqs=1。固定 revision、W8A8、TP=8、DSA CP 和全部输入保持一致。每个候选冷启动后执行五轮 E→C→C→E（20 请求），保存冷启动响应、完整 payload、输入 token IDs/哈希、输出 token IDs、首分歧及全部错误。另执行四次 first-token top-5 logprobs 探针，结果与验收请求分开，报告 top-2 gap；探针不决定稳定性是否通过。诊断不在首个差异处停止，也不反复重试直到成功。

首次通过 20 次诊断及完整 48 次 trace-off 的配置保存为 `selected-config.json`；完整基线失败则保留证据并进入下一个候选。trace-off/on 使用同一镜像和同一选定命令，采集阶段仅增加采集环境和只读采集源码挂载（hook 与准确 prompt 长度的 CPU 元数据传递）。不使用旧原服务的响应作为修改配置的基线。全部候选失败则报告失败并恢复服务。`--diagnostic-only --execute` 只执行配置稳定性诊断及恢复，不执行 48 次基线、采集和回放；默认为完整验证。

独立 watchdog 在 worker 异常退出后清理本任务容器并恢复原服务。采集正常或失败后都恢复原容器 ID，检查配置、`/health`、`/v1/models` 和八卡健康。CPU 回放在恢复完成后执行。同步失败重试三次，服务/HTTP 等待上限 900 秒，任务上限六小时；恢复有独立时间预算。

输入从原 DeepSeek 12 对生成新副本，先等长再追加相同中性后缀，保留身份、边界和前缀。实际 `/tokenize` 及每个 API 响应校验完整 token IDs/哈希，trace-off/on 各 48 请求、并发 1。原始请求与响应完整保留，逐请求检验重复稳定和输出等价；失败报告保留请求身份、完整输入哈希和首个不同 token。

八卡原生记录并集覆盖边界前一位置及后 32 个位置、21 个 C4 层（2、4、…、42）。hook 保存实际 rank、chunk 和 CP 分片区间。校验真实查询归属、缺行、跨 rank 重复、512-wide 选择及因果范围，不填补原生行。回放按位置顺序只处理拥有查询的 rank；普通规划仅使用共同边界前 selected set、历史学习和查询归属，oracle 单独读取未来选择。

C4 native ID i 的支持为 `[max(0,4i−4),4(i+1))`，ready position 为 `4(i+1)−1`。NPU probe 比较 CPU reference、逐 token 扰动依赖、非整组前缀、跨 chunk compressor state 和实际 slot mapping。合同失败即停止。模型、kernel、slot mapping 和缓存 spec 均保存源码哈希和行号。Sidecar 使用可精确展开的 `c4_overlap_v1` 支持映射，列出每 rank 每层完整生成库存、准确 API token IDs 和逻辑版本。regions 为 `other`，保持 `synthetic: true`，单独记录真实 NPU 选择来源。

C4 attention payload 固定 `512×BF16=1024` 字节/native unit/层；page128 为 128 KiB。两种预算口径（`per_rank` 独立容量及每卡预取上限，`aggregate` 八卡统一容量及预取上限）各测试 64/128 MiB、每窗口 8 MiB。七种策略 × native/page128 均在每次 attention 前检查 selected KV 就绪。缓存身份保留 rank，报告每卡及合计字节、precision/recall、有效/浪费字节、同步 recall、污染、CPU 规划时间和分组配对区间。8K 训练、32K 工程评估，pair/episode/source 的分支与重复不跨组。

`status.json` 为服务器阶段；`sync-status.json` 为本地同步状态；`startup-evidence.json` 仅证明后台已启动。成功还要求 `report.json`、四组 `replay-*/report.json`、`restoration.json` 和 `final-local-verification.json`。失败也导出报告、完成阶段和缺失产物。最终校验清单、代码与报告单独提交到 `local-archive/final-artifacts.bundle` 并同步回服务器。

证据限于选定实验配置下的合成输入 DeepSeek 工程验证，不声明当前生产配置已通过。实验结束后原服务使用原容器及原配置恢复。稳定候选只能缩小根因范围，不单独证明 MTP 或某一算子是根因。证据不证明真实任务泛化、在线 offload、KV 恢复或 DMA stall 收益。Indexer、SWA、C128、compressor state 和通信开销不计入该回放字节模型。

验证：`python3 -m unittest discover -s m0a -p 'test_*.py'`。

`--repair` 保留上述六个候选，并在全部失败后累计添加 `HCCL_DETERMINISTIC=true`、再添加 worker 设备绑定之后/分布式初始化之前的 `torch_npu.npu.set_deterministic_level(1)`。逐 rank 在通信初始化前后读取固定 torch_npu 的 `_npu_get_deterministic_level()`，记录设备、PID、HCCL 环境及实际等级。八个 rank 都有完整证据才能继续。trace-off/on 使用相同 worker、DSA/runner 修补及选定环境；仅采集环境与 hook 有差异。环境、源码和补丁 SHA-256 保存在 launch/provenance 文件。不开启 `VLLM_BATCH_INVARIANT`。

新 run 的执行入口先检查旧 worker 的归属和退出源码，使用已有的非重试 `TaskDeadline` SIGALRM 路径结束实验；已在恢复时等待。旧 run 最终同步、原容器配置与健康检查完成后才部署。新 worker 的 SIGTERM 使用独立 `BaseException`，绕过 HTTP 请求及候选切换的普通异常捕获，保存中断证据并进入清理/恢复。恢复有独立 1800 秒预算。旧产物保留。

真实 C4 state 合同从固定 `layer.py` 和部署 cache spec 读取 block size、page padding、stride；分别覆盖 attention (512维) 和 indexer (128维)。128 配置的 state block 是 8：attention state page 131072 字节、内层有效 65536 字节，indexer state page 16640 字节、内层有效 16384 字节。使用打乱的物理 block table，保留零号 sentinel，检查跨物理块、非整组 chunk、释放后的非零块、人工投毒块、请求间复用、padding 与 sentinel 不被改写。两种复用情形各重放 20 次，结果对照独立 FP32 reference 及干净完整前缀。

仅当这两个 compressor 的其余合同及独立 reference 通过，而真实布局的复用测试复现问题时，实验才尝试新请求物理 state block 初始化。仅清理 start position=0 的非零拥有块，不改 sentinel/padding；20 次重放和独立 reference 全部通过后才能挂载到模型实验。保存修复前后证据、首个有差异调用的实参/调用前状态及补丁哈希。未复现时不应用初始化补丁。该证明范围是合成 C4 state 复用，模型根因仍需正式接受结果支持。

八个配置全部失败时，在最后配置中单独运行 20 个相同 event 请求。算子诊断记录各层 attention、MoE、compressor/indexer 的输入/输出指纹和实际 schedule/slot 元数据，用实际输入 token、位置及已观察到的生成前缀匹配请求。首次分歧输入不同时继续保留上游追踪证据；实参相同且完整时仅导出该 rank 首个观察分歧的调用前 backing storage，保留别名、stride、offset、CPU/NPU 参数和 RNG。单卡可重放的算子重置状态后重放 20 次。分布式通信、内部 NPU 格式、不完整大缓存或未知 reference 会明确报告未验证。只有带独立 reference 的复现才支持数值归因；模块边界或插桩消失的分歧不能证明根因。诊断从不作为正式接受请求。

实现目前支持有证据的 compressor 新请求块初始化修补；其他算子的未知异步依赖或非确定性实现保留为失败报告中的疑点，不自动替换自定义算子。未修复则恢复原服务并生成失败报告，不能把“诊断完成”当成 48+48 次正式接受通过。

参考：[HCCL 确定性环境](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/envvar/envref_07_0099.html)、[TorchNPU 接口源码](https://github.com/Ascend/pytorch)、[vLLM 可复现性限制](https://docs.vllm.ai/en/stable/usage/reproducibility/)。
