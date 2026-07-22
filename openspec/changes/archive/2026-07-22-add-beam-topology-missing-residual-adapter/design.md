## Context

B2 standalone-quality validation-best checkpoint 可用，但其 state dict 只保存样本级 `pcer_router`，没有 train-fit global-mean replacement 或 learned global/static block prior。既有 B2 D1 replacement 在 evaluation 脚本中从 validation loader 收集，不符合本轮禁止 validation 统计融合权重的约束。因此基础模型按预注册顺序回退到 C0 `qv_c0_corrupt_global_prior`：其 validation-best checkpoint、resolved config、train-fit GPS normalization、`[5,4,64]` clean block evidence 和 learned `[5,4]` global prior 均已保存，且既有 online/cache fixed-subset reconstruction gate 已通过。

本轮只使用 inner-train 3,600 与 inner-validation 900 个 clean sample，结果是 single-seed、development、claim-ineligible。源码只增加独立 analysis surface；所有训练与报告产物写入 ignored `outputs/missing_residual_adapter/`。

## Goals / Non-Goals

**Goals:**

- 在同一冻结 C0-static 和同一 paired cache 上比较 no recovery、train-mean、plain linear、topology linear、shared linear 和 MLP residual。
- 保证 full、missing radar/GPS、双缺失和其他非精确 single-image/single-LiDAR mask逐元素 bypass。
- 仅从剩余三个 modality-level 64维 beam evidence预测 64维 residual，并使用 train-only normalization、校准和统计。
- 统一报告 beam、oracle-gap、天气、sector、错误距离、动态替换、参数量、显存、吞吐和 success gates。

**Non-Goals:**

- 不解冻或重训 backbone、prototype projector/bank、beam head或静态融合，不修改 canonical recipe或默认模型 forward。
- 不训练 radar/GPS adapter，不研究双模态同时缺失，不读取 channel、CSI、path、gain、power、原始图像/点云或 metadata作为 adapter输入。
- 不运行 outer test、multi-seed、下一轮训练或正式 claim更新。

## Decisions

### 1. B2 不合规时确定性回退 C0-static

预处理器在 `base_model_manifest.yaml` 记录 B2 checkpoint/config与拒绝原因，并固定 C0 checkpoint/config SHA。C0-static logits只由 cached prototype block evidence和 learned global prior在 availability mask 后 softmax融合；不调用任何动态 Router。所有 R0-R5 在 manifest SHA 不一致时 fail closed。

备选是重新遍历 B2 inner-train计算 global mean。附件只允许使用 B2 已保存的 train-fit global mean或 learned static prior；重新设计 B2静态化会扩大范围且无法复用现有 cache，因此不采用。

### 2. 将现有 clean cache转换为最小 paired residual cache

`analysis/precompute_missing_residual_cache.py` 只读取 PR-SQDF manifest中 train/validation的 clean NPZ及 global prior，构造 modality evidence、`z_full`、`z_minus_image/lidar`、两个192维 remaining-evidence输入和 residual identity。输出每 split 一个压缩 NPZ；feature/logit用 float16，训练加载时转 float32，label/identity/weather/scene/8-sector保留离散类型。

预处理器拒绝 forbidden field、outer-test标志、split重叠、非全可用 clean source和来源 SHA漂移。它比较量化前后 full/missing logits与 residual identity，并引用已通过的 C0 online fixed-subset gate；任一 Top1 agreement未达到 full 0.999、missing 0.995时不发布 cache manifest。

### 3. 一个 wrapper表达真实 bypass 契约

`MissingResidualAdapter` 接受基础 logits、四个 modality evidence和 `[B,4]` availability。只有 availability 恰为 missing image 或 missing LiDAR时执行相应 adapter；full、missing radar/GPS、双缺失及任意其他 mask直接返回原 tensor。wrapper维护 forward计数，adapter输入由固定 modality index选择，因此不可能读取缺失 evidence。

analysis runner训练缓存上的相同 adapter，并保存可由 wrapper加载的 state dict。默认 T2/C0 forward不修改；这是 claim-ineligible实验的显式组合边界。

### 4. 六组实验共享一个训练/评测实现

R0不训练；R1只用 inner-train teacher residual均值且 alpha固定1；R2/R3各有 `LayerNorm(192)+Linear(192,64)` 的 image/LiDAR分支；R4使用一个 `LayerNorm(200)+Linear(200,64)` 并拼接8维 missing-modality embedding；R5各有 `LayerNorm(192)+Linear(192,256)+GELU+Dropout(0.1)+Linear(256,64)`。learned alpha为每缺失模态一个 sigmoid标量，初始化0.5。

learned实验以相同 seed、epoch、optimizer、batch order和 modality-paired batch训练。R2损失为 SmoothL1、full-to-corrected KL和CE；R3-R5额外使用现有 cyclic beam soft-label topology loss与 teacher-student topology transport。固定 train batches上的 zero-residual raw loss用于一次 train-only量级校准，目标份额固定为50/15/20/10/5%，所有实验保存相同 calibration artifact，validation只用于最低 total adapter loss选 checkpoint。

### 5. inner-validation同时作为唯一 evaluation split

最终指标只在同一900个 inner-validation sample计算。R0-R5的 full、missing radar/GPS和非目标 mask共享同一基础 logits，S3 macro/worst由四种 single-missing构造；其他 S0-S5 若 clean block cache可按既有 deterministic mask contract重建则报告，否则明确标为未运行，绝不读取 outer test补齐表格。

天气、8-sector和 circular beam distance均从 label/metadata离线分组；这些字段不进入 adapter。D0-D3替换测试、teacher纠错/新错、Pearson/Spearman/cosine、残差方差和 success gates由同一预测快照汇总。

### 6. GPU launcher隔离六个任务

GPU0-5分别运行 R0-R5。launcher先记录 `nvidia-smi`，为每个任务写 PID/status/log，等待所有进程且不因单任务失败终止其他任务；`--summarize` 拒绝缺失结果。修复后可用 `--experiments` 只重跑失败任务，汇总结束即停止。

## Risks / Trade-offs

- [C0而非首选B2导致基础Full较低] -> manifest明确记录 fallback证据，本轮结论只适用于 C0-static，不外推B2。
- [float16 cache改变边界样本排序] -> 发布前执行逐字段误差与 Top1 agreement gate，失败则改用float32而不训练。
- [900个validation样本使天气/sector方差较大] -> 报告sample count与方向，不形成正式claim。
- [cache wrapper不自动嵌入默认模型forward] -> 保存同一adapter state并以 online-composition unit test验证bypass和输入选择；只有方向通过后才另行提议正式集成。
- [并行GPU任务非常轻，进程启动占主要时间] -> 保持一卡一实验以符合预注册映射，不增加分布式框架。

## Migration Plan

1. 创建并严格校验 OpenSpec change；实现 cache转换、runner、launcher和synthetic tests。
2. 运行聚焦测试与预处理 gate；gate失败即停止，不启动R0-R5。
3. 通过后检查GPU0-5并运行六组任务；只重跑失败实验，统一汇总后停止。
4. 删除独立 analysis/launcher即可回滚；C0、B2、PR-SQDF cache与默认训练评估不受影响。

## Open Questions

无。基础模型按审计结果固定为 C0-static；inner-validation是本轮唯一evaluation split。
