## Context

A0、A1、B2、C0、C7 使用相同的 MMW 15-domain inner-train 3600 条和 inner-validation 900 条样本，backbone 与 64 维 block space 相同，均存在 validation-loss 选择的 `best.pth`。当前 encoder 在内部将 TinyViT 320 维、Radar/GPS 末端 hidden 表示投影到 64 维；`UMaskBeamJEPA.encoder_projections` 在这些 recipe 中均为 identity，prototype bank 再对 64 维 block feature 做 L2 normalize 和 cosine logits，不存在第二个独立 prototype projector。

历史 `T2-NoBPA/seed1` 虽关闭 BPA，但其 4500 条 confirmation train 已包含本轮 900 条 inner-validation，且使用 40 epoch/`last.pth`，因此不能作为本轮 probe validation 上的公平因果对照。诊断必须记录缺少公平 NP，而不能重训或把该 checkpoint 混入主结论。

## Goals / Non-Goals

**Goals:**

- 对五个已训练 checkpoint 复用完全相同的 inner 样本和确定性 clean/corrupt panel。
- 区分 prototype bank collapse、feature collapse、良性 beam semantic compression、quality erasure、cross-modal over-alignment、modality laziness 和 Router observability failure。
- 完成 beam-conditional corruption collapse 的 paired distance、centroid、conditional probe、sensitivity、hidden degradation 与 virtual-gradient 证据。
- 以分片缓存复用 backbone 结果，并使单个统计任务失败不破坏其他任务和已有缓存。

**Non-Goals:**

- 不训练、微调、调参或选择下一代模型，不自动补 no-prototype run。
- 不读取 outer test、channel、CSI、path、beam gain/power vector 或 ray-tracing path。
- 不修改 checkpoint、原实验指标、canonical recipe、正式 claim 或 public package CLI。
- 不以 effective rank、CKA 或 UMAP 单项指标直接宣称 collapse。

## Decisions

### 1. 固定 checkpoint 与样本身份

manifest 固定 A0/A1/B2/C0/C7 的 validation-best checkpoint、resolved config、selection、loss/router 开关、参数量和 `prototype_bank.prototypes`。样本使用 quick protocol 的全部 inner-train 3600 条拟合 probe、全部 inner-validation 900 条评测；按 weather、scene、beam 做确定性排序与审计，不改变原 split。所有 checkpoint 必须通过样本 manifest checksum 一致性校验。

### 2. 复用 PGCD corruption，定义有限 panel

panel 使用 `SensorDegradationGenerator` 的 image blur/occlusion/exposure-noise、LiDAR point/range dropout 与 coordinate jitter、Radar detection dropout/jitter/clutter、GPS drift/jump/white noise，各取 L1-L3；每模态另有一次 L2 stale 和 L4 missing，加 clean 共 45 个 condition。随机性由现有 SHA256 generator 与固定 global seed/sample identity/sensor/type/variant 派生；severity 只改变强度，所有 checkpoint 复用同一 view identity。

### 3. 用 opt-in return 与 hook 表达真实层级

`return_intermediates=False` 保持默认行为。启用时 model 额外返回 `[B,T,M,64]` 的 `F_block/F_postproj/F_proto/Z_block`、模态聚合、router 和 fused 状态。抽取器只在各 encoder 最后输出线性层注册 forward hook，捕获其输入作为 `F_enc/F_preproj`；不复制 encoder 或 model forward。

layer manifest 明确：`F_enc` 与 `F_preproj` 是同一真实 tensor，`F_postproj` 与 `F_block` 是同一真实 tensor；这些是架构 alias，不重复解释成独立层。不存在的 tensor 写 `not_available`。

### 4. 分片缓存保持最小充分信息

每个 checkpoint/split/batch 写一个 NPZ shard。clean view 保存四模态 pre-projection tensor；corrupt view只保存受影响模态的 pre-projection tensor，完整保存 64 维 block/prototype/logit/router/fused 状态，并由 sample/condition identity 与 clean shard 配对。大特征用 float16，统计转 float32；metadata、shape、dtype、checksum 和 alias 写入 index，缓存不得序列化原输入、future beam path 或任何 channel/path 字段。

### 5. 一个共享实现加附件要求的薄入口

公共统计、流式 shard 读取、固定 probe、metric 与 CSV 写入集中在 `analysis/prototype_collapse_diagnostics.py`。附件点名的 D1-D6、抽取和聚合文件只提供独立 subcommand 入口，避免复制逻辑。probe 统一使用 train-only StandardScaler 和固定预算的线性 SGD/logistic/ridge；不同输入维度不可公平合并时输出 `not_comparable`，不补零伪造。

### 6. Modality dependence 区分精确、近似与 unavailable 证据

缓存已保存完整模型在 clean、单模态 missing 和 corruption view 下的 `Z_fused/Z_modality/router_weights`。D5 的 joint 与 LOMO 使用完整模型缓存的精确输出，single 报告联合模型的 unimodal head；shuffle 与 zero/mean/wrong replacement 使用统一的 cached-logit ablation，并在 CSV 的 `inference_mode` 中明确标记，不能冒充动态 router 的精确反事实。缓存是 detached tensor，未保留 backward graph，因此 feature-gradient sensitivity 写 `unavailable` 及原因，不伪造梯度。LiDAR dominance 必须由 LOMO 之外的 shuffle、unimodal 或 weight 等独立证据共同支持；单个 missing drop 只能判为 missing sensitivity。

### 7. 组合判定而非单阈值

聚合器按附件 A-H 与 BC7 的联合证据生成 `diagnostic_summary.md`，同时写支持、反对和 unavailable evidence。只有 encoder 可分、prototype 距离/quality probe 收缩、task degradation 上升、prototype sensitivity 低、beam probe 保持且 router 未降权同时成立，才判定有害 beam-conditional corruption collapse；否则区分良性 invariance 或证据不足。最终只能选择一个主方向。

## Risks / Trade-offs

- [完整 45-condition panel 计算和缓存较大] -> 先预处理固定模型输入，再在用户授权的 GPU0-5 并行抽取；按 batch 分片，已完成 shard checksum 命中时跳过。
- [encoder pre-projection 维度跨模态不同] -> beam/quality probe 按模态独立；modality-ID 在不可比层明确标记 unavailable，CKA 仍可处理不同维度。
- [hook 位置随 encoder 实现变化] -> 启动前按 encoder 类型 fail closed 校验最后线性层路径与输出 shape，并写 layer manifest。
- [NoBPA checkpoint 有 validation contamination] -> 不纳入主分析，manifest 明确 `fair_causal_control=false` 和原因。
- [GPU 状态变化] -> runner 启动前重新记录 `nvidia-smi`，只使用用户范围内空闲设备，不终止其他进程。
- [单项诊断失败] -> runner 独立记录 PID/return code/stdout/stderr，继续其他任务；summary 显式列 unavailable，不能默认为通过或失败。

## Migration Plan

1. 增加 opt-in intermediate return 与 synthetic 测试，确认默认 forward/state dict 不变。
2. 生成 checkpoint/layer/sample manifest 并运行小样本 cache/metric 测试。
3. 预处理固定 inner 输入，并在 GPU0-5 抽取五个 checkpoint 的完整 inner cache，随后以受限线程在 CPU 运行 D1-D6/BC1-BC6。
4. 聚合 CSV、图表与唯一主方向；保留全部产物在 ignored `outputs/prototype_collapse_diagnostics/`。
5. 回滚只需删除 opt-in 源码与本地输出；历史模型、配置和 checkpoint 不受影响。
