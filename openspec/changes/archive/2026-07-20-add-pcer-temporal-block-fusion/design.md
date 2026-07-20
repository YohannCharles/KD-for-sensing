## Context

MMW 当前主路径固定为 image、radar、gps、lidar 四模态，历史长度 5，`UMaskBeamJEPA` 在 encoder 后形成 `latent_sequence[B,T,M,64]`。现有 prototype head 持有 `prototype_bank.prototypes[64,64]`，BPA 用 beam-index topology soft target 约束 fused/modal features；旧 Router 则在 masked temporal pooling 后根据 confidence、entropy、prototype margin、reliability 和 modality identity 产生 `[B,M]` 权重。

现有 `temporal_missing` 能在输入张量、valid mask、masked pooling 和模态 softmax 上阻断缺失单元，但其 600-entry schedule 不表达本轮六类语义场景。MMW prepared CSV 逐时刻保存真实资源路径和 frame id，当前 loader 没有 forward/backward fill，也没有 `source_frame_id[m,t]` batch 字段；实现必须接受可选复制帧分组，但不能伪造数据中不存在的 group identity。

## Goals / Non-Goals

**Goals:**

- 以一个 opt-in T2 分支完成 A0--A3，保持 encoder、prototype bank、BPA、optimizer 和数据预算一致。
- 让缺失块在输入、池化、evidence fusion、Router softmax 和归一化分母中严格为零。
- 用同 backbone 完整视图监督缺失视图，用缓存块证据产生向量化反事实 Router target。
- 在 MMW 15-domain、seed1、16 epoch 的 claim-ineligible quick-validation 中完成固定六场景评测和机制诊断。

**Non-Goals:**

- 不重建 RGB、Radar、GPS 或 LiDAR，不加入模糊、噪声、稀疏或杂波退化。
- 不新增独立 teacher、模型注册名、训练循环、第三方依赖或 canonical recipe。
- 不运行多 seed、LOS/NLOS、双模态全缺失、单模态生存或正式 outer evidence。

## Decisions

### 1. 新 mask generator 复用现有应用边界

新增 `TemporalBlockMaskGenerator`，生成器接口使用 `availability_mask[B,M,T]`，并按 sample id、mask type、variant、seed 做 SHA256 派生随机数。训练接入 `apply_training_temporal_missing` 时只做一次 `permute(0,2,1)`，继续复用 `apply_modality_temporal_mask_to_batch` 清零输入并写出 runtime mask。

新增 `pcer_curriculum` mode。前 10% 为 full/sparse 各 0.5；10%--30% 为 full 0.30、sparse 0.30、burst2 0.25、missing-modality 0.15；后 70% 使用请求中的完整分布。训练按 sample identity 采样，评测按 global eval seed、sample identity、mask type 和 variant 固定生成。若传入 `source_frame_ids[B,M,T]`，同一模态内相同非空 id 同组 mask；未传入时不猜测。

### 2. PCER 是 UMask 内部 opt-in 分支

`model.primary.pcer.mode` 只允许 `evidence_static` 或 `counterfactual_router`；字段不存在时不实例化新参数，不改变 current state dict 和 forward 数值。A0 使用现有 `uniform_mean` 静态融合，A1 使用 current supervised Router；A2/A3 才启用块证据分支。

块特征直接使用 `latent_sequence[B,T,M,64]`，因为其维度已与 prototype bank 一致，不增加无效 projector。调用同一 prototype bank 得到 `[B,T,M,64]` cosine logits，再按时间优先展平为 `[B,N,64]`。A2 对可用块等权；A3 的小型共享 Router 消费 detached block feature、evidence confidence/entropy、modality embedding 和 time embedding，输出 `[B,N]`，缺失位置在 softmax 前屏蔽。

### 3. 完整视图复用现有 superset runtime

`temporal_missing.preserve_unmasked_for_superset=true` 保留原始输入。A2/A3 在训练 extension 中用现有 same-model no-grad superset forward 生成 teacher logits，不创建 EMA 或 checkpoint teacher。普通 temperature KL 只对 student mask 严格小于 base mask 的样本计算，full 样本贡献为零。

虽然第二个 forward 比复用块特征更慢，但它已由仓库测试过，能保证完整视图经过同一完整 forward contract；quick-validation 仅 16 epoch。后续只有在 profiling 表明开销不可接受时才将 superset 缓存下沉到 block feature 层。

### 4. 反事实监督只在 evidence tensor 上向量化

对 detached `block_evidence_logits[B,N,K]` 使用可用块等权 reference。一次计算全量 evidence sum，并通过减去每个 block 构造 `[B,N,K]` leave-one-out logits；用当前 BPA topology soft target 得到逐块 `loss_without_i-loss_with_all`。贡献先按可用块中心化并可配置 clamp，再经 temperature softmax 形成 target；缺失块不参与 target 或预测分布。

Router loss 使用 target 到 predicted weights 的 KL。训练和评测均复用同一 target helper，以输出 Pearson、Spearman、top1 agreement、entropy、样本间标准差和 modality/time 均值。

### 5. 四配置只改变目标消融项

- A0：prototype/BPA、统一 mask、`uniform_mean`，无旧 Router、consistency 或 route loss。
- A1：prototype/BPA、统一 mask、current supervised Router 和现有 oracle Router loss，无 PCER。
- A2：prototype/BPA、块 evidence 等权、完整/缺失 KL，无旧或反事实 Router。
- A3：prototype/BPA、块 evidence、完整/缺失 KL、反事实 Router KL，无旧 Router。

共同使用 seed1、MMW 15-domain、group-safe inner train/validation、历史 h5p1 strict test、batch32、16 epoch、AdamW/cosine 配置和相同六类训练 mask。DataLoader 统一使用 12 workers、prefetch factor 1；该值由同配置 5% 子集的 4/12 worker 实测确定，避免四任务共享预处理吞吐导致 GPU 长时间空闲。h5p1 test 已是历史开发资料，因此结果只用于本轮判断，不升级 formal claim，也不读取冻结 `outer_evidence`。

### 6. best checkpoint 为显式 opt-in

训练 runtime 在 `training.checkpoint_selection=best_validation_loss` 时逐 epoch验证并额外发布 `checkpoints/best.pth`；默认仍只发布 `last.pth`，现有 40-epoch协议不变。quick evaluator 只接受 `best.pth`，并记录选择 epoch、validation loss 和 checksum。

## Risks / Trade-offs

- [A2/A3 的逐块 evidence 会改变 A0/A1 的融合位置] → A0/A1 明确保留现有静态/旧 Router 路径，A2 用于单独识别 evidence-space 收益。
- [完整视图 forward 增加显存和时间] → teacher 使用 `no_grad`，先完成 batch32 单步 smoke；四组 effective batch 保持一致。
- [反事实贡献在证据近似下不等于重跑 backbone] → 该近似是预注册机制，只声明块 evidence 层贡献，不解释为传感器因果效应。
- [复制帧 identity 不完整] → 接口支持 group ids，实际 CSV 路径审计写入 notes；无法可靠恢复时明确标记限制。
- [单 seed 结果有方差] → 仅按三项 quick gate 判断是否值得继续，绝不升级正式 claim。

## Migration Plan

1. 保持 `pcer` 字段缺省，验证 current T2 数值和 state-dict 兼容。
2. 添加 mask、model/loss 与 opt-in best checkpoint 聚焦测试。
3. 生成四组 resolved config，完成配置、单 batch forward/backward 和 GPU smoke。
4. GPU4--7 并行训练；单个任务失败只修复并重跑失败任务。
5. 用共同 fixed-mask evaluator 汇总后停止，不自动启动下一批实验。

回滚只需移除 opt-in 配置和本 change 新代码；canonical recipe 与历史 checkpoint 不受影响。

## Open Questions

- 若 A2 已通过且 A3 未优于 A2，后续优先保留静态 evidence fusion，不继续扩大 Router。
- 若实际资源路径审计发现同一窗口内复制帧，后续 change 应把 per-modality source ids 正式纳入 dataset batch contract；本轮不临时猜测。
