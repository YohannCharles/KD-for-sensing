## Context

PR-SQDF cache 已从冻结 C0 `qv_c0_corrupt_global_prior` 保存 inner-train 3,600、inner-validation 900 个 clean sample。每个样本包含 `[5,4,64]` clean block logits、`[5,4]` availability、64 类 beam label、weather、scene 和原模型 full fused logits；每个 shard 另保存同一个 `[5,4]` learned global prior。C0 对 availability mask 后的 20 个 prior logits 做 softmax，再线性加权 block logits，因此本轮无需重新训练或再次缓存 backbone feature。

本轮是 fresh local analysis，所有 checkpoint、CSV、图表和报告只写入 ignored `outputs/missing_evidence_probe/`。结果是 single-seed、inner-only、claim-ineligible evidence，不修改正式 claim。

## Goals / Non-Goals

**Goals:**

- 验证剩余三模态的 clean modality evidence 对缺失模态 evidence 和 full-minus-missing residual 的可预测性。
- 用 no recovery、mean、train-only nearest neighbor、linear、MLP 和 oracle 同口径比较最终 beam Top1、Top3、Top5、Within-3、MAE 与 oracle-gap recovery。
- 在正式 probe 前以全部 inner-validation cache 和固定原模型子集验证 full/missing 融合一致性。
- 对四个缺失模态、weather 和 8 个连续 beam sector 完整报告，并给出唯一后续方向。

**Non-Goals:**

- 不重新训练 backbone、prototype bank 或完整 fallback adapter，不修改 C0 checkpoint 或 canonical recipe。
- 不使用 outer test、multi-seed、corrupted view、channel、CSI、path、channel gain、beam power vector 或原始传感器特征作为 probe 输入。
- 不增加 Transformer、MoE、额外 probe 类型或超参数搜索，不把本轮结果升级为正式 claim。

## Decisions

### 1. 只读取 train/validation 的 clean NPZ

runner 从 `cache_manifest.json` 的 `splits.train/validation.batches[*].clean_path` 加载 sample id、weather、scene、beam label、clean block logits、availability 和 clean fused logits，并读取 shard global prior。它拒绝 `eval` split、forbidden manifest、非 C0 protocol、outer-test 标记、split 重叠和非全可用 clean mask。

备选是复用 development E5 missing view，但该路径来源于历史 development/test CSV，不符合本轮 inner-validation 限制，因此不采用。

### 2. modality evidence 保留 C0 两级等价分解

对模态 `m`，先在该模态五个时间块上对 C0 prior 条件归一化，得到 `E_m=sum_t softmax_t(prior[t,m])*Z_block[t,m]`。full-observation modality mass 等于该模态五个 block prior mass 之和，四个 mass 加权 `E_m` 与原 20-block full fusion 等价。missing logits 则在移除该模态五个 block 后对剩余 prior 重新 softmax。

这种分解允许 evidence predictor 只输出 64 维 `E_hat_m`，再按 full-observation modality mass 加回。真实 `E_m` 的 oracle add-back 与 cache-reconstructed Full 数值一致；residual oracle 使用 `Delta=Z_full-Z_minus_m`，按构造精确等于 Full。

### 3. 最小输入与固定四任务

输入只拼接剩余三个 `E_j`，固定为 192 维；不附加 entropy、margin、metadata 或 embedding。每个 missing modality 顺序训练 Linear Evidence、MLP Evidence、Linear Residual、MLP Residual。Linear 是单层 `Linear(192,64)`；MLP 严格使用 `LayerNorm -> Linear(256) -> GELU -> Dropout(0.1) -> Linear(128) -> GELU -> Linear(64)`。

四任务共享 seed=1、train/validation sample identity、train-fit input mean/std、epoch=30、batch=512、AdamW、学习率 `1e-3`、weight decay `1e-4`、每 epoch 固定 batch order 和 patience=5。checkpoint 只按最低 inner-validation recovery loss选择，不按 Top1选择。

### 4. 损失直接复用现有 topology

Evidence loss 使用 SmoothL1、teacher-to-student KL 和 C0 cyclic beam topology 的 distribution transport expectation。Residual loss使用 SmoothL1、corrected-logit CE 和现有 topology soft-label cross entropy。beam label只进入 residual监督和离线指标，不进入模型输入。

### 5. preflight 分成 cache 全量和原模型固定子集

全部 900 个 inner-validation sample用于 cache full reconstruction。固定 manifest-order 子集由冻结 C0 分别执行 full 与四种 modality missing mask forward，和 cache block/prior 重建结果比较。full Top1 agreement低于 0.999或任一 missing低于0.995时，runner fail closed且不创建 probe checkpoint。

静态测试另验证 residual identity、input schema、split隔离、train-only normalization/NN、oracle add-back、共同代码路径、参数量和固定 seed；输出保存到 `preflight_tests.txt`。

### 6. 每模态独立运行，最后单进程汇总

GPU0--3 分别映射 image、lidar、radar、gps；设置物理 `CUDA_VISIBLE_DEVICES` 后程序内部只使用 `cuda:0`。每个 worker只写自身 checkpoint、状态和 `results/<modality>.json`，避免并发写共享 CSV；所有 worker完成后由 `--summarize` 生成主表、预测性表、weather/sector 分层和可行性总结。单任务失败不终止其他任务，但汇总拒绝缺失方向。

## Risks / Trade-offs

- [float16 block cache带来微小 logit误差] -> 先报告 max/mean absolute error与 Top1/Top3/Within-3 agreement，并以 Top1 gate fail closed。
- [MLP 用 validation recovery loss选择后仍可能无 beam收益] -> 可行性只由最终 beam指标、oracle gap与分层方向判断，不以相关性代替任务收益。
- [NN距离计算占用显存] -> 对 validation 分批执行 `torch.cdist`，检索库只包含 train evidence。
- [四进程重复加载约数十 MiB clean cache] -> 数据规模仅 4,500 sample，换取独立状态和简单失败隔离；不引入共享服务。
- [原模型 fixed-subset preflight需要一次 raw validation forward] -> 只在 cache缺失 missing原输出时执行，冻结全部参数，不保存原始张量或空间 feature。

## Migration Plan

1. 新增 OpenSpec、独立 analysis runner、GPU launcher和 synthetic tests，不修改 package public surface。
2. 运行测试与 preflight；gate失败则停止并保留 cache检查报告。
3. 通过 gate后运行四个 modality worker，再统一汇总本地结果并停止。
4. 删除 opt-in runner/launcher即可回滚；C0、PR-SQDF cache和默认训练评估不受影响。

## Open Questions

- 无。输出 evidence 采用 raw logits，不额外标准化；只对 192 维输入使用 train-fit mean/std。
