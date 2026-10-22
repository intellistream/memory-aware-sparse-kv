# DeepSeek-V4-Flash：CPU 稀疏注意力数值实验

日期：2026-10-22。默认模型始终为 `/models/DeepSeek-V4-Flash-W8A8`。

## 模拟的是什么

前一版 `issue6_simulator.py` 是**系统时序模拟器**：输入选集 trace，模拟预测成本、缓存、Host–设备搬运和排队。合成 C4 选集来自人为概率模型；旧 GLM 选集来自真实受控采集。它不计算 DeepSeek 的 Q/K/V，也不具备模型回答能力。

本次新增 `deepseek_v4_cpu.py`，是**注意力组件的数值参考实现**。它从现有 DeepSeek 检查点加载真实参数，在 CPU 上实际执行矩阵乘法、门控压缩、位置编码、indexer 打分、top-k 和 attention 输出计算。输出选集由输入张量和权重算出，不再直接抽取预设热点集合。

| 层次 | 输入与执行 | 能验证的内容 |
| --- | --- | --- |
| 先前系统模拟 | 选集 trace + 假设时序 | 缓存/预算/搬运和收益条件 |
| 本次数值组件 | 真实注意力权重 + 明确标注的受控隐藏状态 | 稀疏算子数值、选集变化、缓存召回后的 attention 输出一致性 |
| 完整 DeepSeek 推理 | tokenizer、embedding、全部 43 层、mHC、MoE、输出头和采样 | 对话、工具决策、真实隐藏状态与真实原生选集 |

第三行当前没有执行。本次分别调用第 0、2、3 层注意力组件，**没有把这三层串成模型**，也没有把它们的输出作为自然语言生成结果。受控输入不是前置模型层产生的激活，真实权重本身不足以让这些输入具备真实语义。

## 实现的具体路径

使用模型实际维度：隐藏维 4096，64 个 attention head、每 head 512 维；indexer 为 64×128；Q 低秩维度 1024；输出为 8 组低秩投影；窗口 128；C4 的 top-k 为 512。

```text
受控隐藏状态 x
  ├─ Wq_a → RMSNorm → Wq_b → per-head RMSNorm → RoPE → Q
  ├─ Wkv → RMSNorm → RoPE → 最近 128 个位置的 SWA KV
  ├─ 学习的 Wkv/Wgate + APE → gated pooling → RMSNorm → RoPE → 压缩 KV
  └─ C4 indexer：自己的压缩器、Q 投影、Hadamard、逐 head ReLU 打分加权 → top-512

SWA KV + 选中的 C4 KV / 全部已生成 C128 KV
  → 带 attention sink 的 softmax → 加权 KV → 逆 RoPE → 分组 Wo_a → Wo_b
```

- 第 0 层：SWA，只有最近 128 个位置，包含当前位置。
- 第 2 层：SWA + C4。每四 token 产生一个压缩向量；一个完整 C4 单元使用前后两个四 token 块的不同投影半部，首块屏蔽不存在的前驱。对已生成压缩向量计算 indexer 分数，再选至多 512 个。
- 第 3 层：SWA + C128。每 128 token 产生一个压缩向量，访问所有已经完成的压缩块，不伪造一个 C4-style indexer。

压缩单元 `j` 的 RoPE 位置使用 `j × ratio`；C4 在 token 位置 `p` 的可用单元数为 `floor((p+1)/4)`。未来压缩块即使为了离线计算效率预先存在于 CPU 数组中，也被选择接口严格屏蔽。

indexer 分数是各 head 的 `ReLU(q_h · k_j)` 乘以输入相关权重后求和。真正的 attention 对 SWA 和选中压缩 KV 共同归一化；attention sink 只参与分母，值向量为零。不是用 indexer 分数直接替代 attention 权重。

## 权重与精度

通过 safetensors 索引只读取需要的注意力张量，不读取 MoE 专家，也不加载全模型。每个已读张量记录名字、分片、形状、原始 dtype、字节数和 SHA256；模型配置和索引单独记录哈希。模型目录只读。

两种精度配置明确区分：

1. `dequant_fp32`：INT8 权重按实际每输出通道 scale 解码，在 FP32 中计算；检查量化 offset 为零。压缩 KV 与 SWA KV 存为 BF16。
2. `dynamic_int8_reference`：在上述基础上，对量化线性层输入和 indexer Q/K 加入对称逐向量 INT8 round/clamp/dequantize，检查 top-k 对数值变化的敏感性。

第二种只是 CPU 量化参考；未复刻 Ascend 全部舍入、融合、BF16 中间存储和累加顺序。两者都不能宣称与 W8A8 NPU kernel 逐位相同。时序模拟中的 1024 bytes/C4 单元与本次 BF16 512 维存储一致，但不代表完整设备内存占用。

## 数值验证与实验

15 项单元测试包括：C4 首块 padding/投影半部、C4/C128 整块与逐 token 压缩一致性、屏蔽未来 token、indexer 手工逐 head 计算、top-k 平局规则、Hadamard 保持内积、RoPE 可逆、sink 归一化、稀疏 gather 与独立 dense-mask 计算、遗漏 KV 确实改变输出、容量受限缓存复制与召回。

正式数值实验保留真实维度与权重：

| case | 组件 | 初始长度 | decode 隐藏状态数 | 精度 |
| --- | --- | ---: | ---: | --- |
| c4-4096-stable-fp32 | layer 2，C4 | 4096 | 32 | 解码权重后 FP32 |
| c4-8192-shift-fp32 | layer 2，C4 | 8192 | 32 | 解码权重后 FP32 |
| c4-4096-stable-int8 | layer 2，C4 | 4096 | 32 | 动态 INT8 参考 |
| c128-4096-fp32 | layer 3，C128 | 4096 | 16 | 解码权重后 FP32 |
| swa-4096-fp32 | layer 0，SWA | 4096 | 16 | 解码权重后 FP32 |

隐藏状态生成器使用固定种子的八组潜在向量和逐 token 噪声；shift case 在第 16 个 decode 输入切换潜在向量。这里没有语义标签、真实工具事件、自然语言任务成功率或通用性结论。

每个 query 比较两条实际数值路径：完整 CPU KV 上的显式掩码，以及只 gather 需要的 KV。对三个 C4 case，再用容量 512 的独立 tensor LRU 执行无预取、上一选集、全局频率、阶段频率四种策略；遗漏选集同步召回，要求完整输出投影后的向量与完整缓存结果完全一致。实际复制的 BF16 张量被独立计数，并与原系统模拟器的搬运账本交叉核对。

输出 `trace.jsonl.gz` 保持 `source_kind=synthetic_fixture`，并增加真实权重组件的来源说明；不冒充 `native_trace`。完整 indexer scores 在 `numerics.npz` 单独保存，未来位置没有分数。时序扫描仍使用假设参数，不把 CPU 校验的耗时当作 NPU 延迟。

## 本次已完成的结果

`run-r01` 五组实验全部完成，共读取 58 个真实 checkpoint 张量（563.64 MiB 原始张量数据，非运行峰值内存），未读取 MoE 权重。15 项单元测试通过；保存结果的独立审计检查了 8 个执行源码哈希、128 组输出、96 组 indexer 分数与 top-k、384 次缓存输出比较。

| case | gather / dense-mask 最大绝对差（输出投影前） | 相邻压缩选集平均 Jaccard |
| --- | ---: | ---: |
| C4，4096，FP32 | 1.07e-5 | 0.487 |
| C4，8192，受控切换，FP32 | 2.15e-5 | 0.544 |
| C4，4096，INT8 参考 | 1.10e-5 | 0.487 |
| C128，4096 | 2.03e-6 | 1.000 |
| SWA，4096 | 1.43e-6 | 不适用 |

浮点检查使用 `atol=2e-5, rtol=3e-5`，全部通过；这不是与原生 NPU 的误差比较。384 次有限缓存召回后的完整 attention 投影输出，均与同一 CPU 算法的完整 KV 基线逐值相同，最大差为 0，输出 SHA256 相同。C128 这 16 个 decode 位置尚未完成下一个 128-token 压缩块，所以其压缩选集保持不变；SWA 窗口仍随位置滑动。

8192 case 在第 16 个 decode 输入切换潜在原型；跨切换边界的 C4 选集 Jaccard 为 0.247。4096 case 的 FP32 与动态 INT8 参考选集平均 Jaccard 为 0.9864。这些是本次固定种子、受控输入的观察值，不是真实对话统计或多次运行的置信区间。

选集图见结果目录的 `c4-selected-units.png`：横轴为 C4 压缩单元 ID，纵轴为 decode query，蓝色表示真实权重 indexer 选中的单元；橙线为受控输入切换。

接入旧系统模拟器后，在容量/预算均为 512 单元、16 GB/s、每 128 单元启动成本 5 μs、预测成本 5 μs、提前量 50 μs、attention 假设 20 μs 的条件下：4096 FP32 case 的全局频率策略仅缩短单层模拟服务时间 1.33%；8192 切换 case 的全局频率策略反而增加 7.11%，阶段频率策略增加 1.93%。这说明“有选集重叠”不足以证明预取收益；提前量、缓存污染和额外搬运必须共同计算。所有时间都是假设参数下的系统模拟结果，不能换算成实测 NPU TPOT。当前 `last_selection` 没有额外收益，因为上一选集已经留在 LRU 中。

这些结果为 issue6 补充了真实权重下的选集产生、因果边界和 CPU 数值正确性证据。任务 18–20 所涉及的真实设备异步搬运、原生 kernel 对齐、端到端在线性能仍未验证。

## 复现和接入

运行环境为 Python 3.12、PyTorch 2.10.0+cpu、safetensors 0.8.0。必要依赖记录于 `m0a/deepseek-v4-cpu-requirements.txt`；新环境应先从 PyTorch 官方 CPU wheel 源安装 torch，再安装该文件。结果包的 `environment-packages.json` 保存现场依赖版本，`reference/SHA256SUMS` 固定实际对照的 Ascend 源码快照。

远端目录：`/root/experiments/issue6-dsv4-cpu-20261022/`。复用前一阶段独立 CPU venv，设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免没有设备时加载 NPU 插件。

```bash
cd /root/experiments/issue6-dsv4-cpu-20261022/code
PY=/root/experiments/issue6-simulation-20261022/venv/bin/python
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -m unittest m0a.test_deepseek_v4_cpu -v
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PY" -u -m m0a.run_deepseek_v4_cpu \
  --model-dir /models/DeepSeek-V4-Flash-W8A8 --threads 8 --output ../run-NEW
"$PY" -m m0a.summarize_deepseek_v4_cpu ../run-NEW
```

`AttentionComponent.prepare(hidden, prompt_tokens)` 已将输入隐藏状态与组件执行分离。以后取得真实模型对应层的 attention 输入，能够直接替换受控生成器；必须同时记录捕获层、norm/mHC 位置、token IDs 和输入哈希。然后与 NPU 原生 score/选集/输出对齐，才有完整模型行为方面的证据。

## 源码依据

- [DeepSeek 官方参考模型](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)：Compressor、Indexer、Attention 的算子关系与压缩规则。通过网页查阅；本机与远端直连下载失败，未声称执行该官方文件做对比。
- [vLLM-Ascend 模型](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/deepseek_v4.py) 和 [CP attention](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/context_parallel/dsa_cp.py)：结合现场安装版本核对 W8A8 布局、Q norm、SWA/压缩注意力和输出逆 RoPE；现场源码封存在 `reference/`。

当前通过的数值一致性是两个 CPU 参考算法及缓存实现之间的一致性。尚未完成 NPU kernel、完整 43 层、MoE/mHC、自然语言输出或真实 Agent 的一致性验证。
