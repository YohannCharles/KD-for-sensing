## Context

当前 Scene31 next-round 是 manifest-backed local/manual workflow：`scripts/generate_scene31_next_round.py` 生成 run manifest 和本地 YAML，训练仍通过 `kd-sensing-train --config <generated-yaml>` 执行。`proto_sampler_uniform_es40` 的缺失模态 pattern exposure 已经通过 `loss.u_mask_beam_jepa` / training 字段接入 `UMaskBeamJEPATrainingExtension.before_forward`，该 extension 负责按 pattern 生成 `missing_mask` 并传给共享 runtime。主 beam loss 由 `BatchStepRunner._compute_base_loss` 调用 `context.task_criterion` 计算，loss registry 已有 hard/soft target CE 基础。

## Goals / Non-Goals

**Goals:**

- 在不改变默认 baseline 行为的前提下新增 adaptive pattern-balanced sampler。
- 新增 beam-neighborhood hard/soft CE 混合 loss，并保留普通 label smoothing 对照。
- 扩展 Scene31 BC 实验矩阵、launcher、fresh eval/summary 和 sanity checks。
- 将 AMR-Net 与 AMBER full 作为需要训练的 baseline 纳入 BC launcher 的 baseline group。

**Non-Goals:**

- 不继续扩展 condBTAPA、weakKD、BTAPA tau1 或新 imputation/Transformer 模块。
- 不恢复旧入口、旧实体 generated YAML 或旧兼容聚合层。
- 不改变已完成 run、已有 checkpoint 或默认 CE/focal loss 行为。
- 不把 sampler 写死到 Scene31；Scene31 只负责生成本地实验矩阵。

## Decisions

1. **adaptive sampler 放在 U-Mask training extension 内。**
   现有 uniform sampler 已经通过 `sample_pattern_balanced_mask` 和 `missing_pattern_sampler=uniform` 控制 per-batch mask，因此最小实现是在同一 extension 中维护 pattern EMA state，并把 adaptive probability 传给同一个 mask helper。这样不会触碰 DataLoader、dataset 或模型 forward contract。

2. **EMA 更新使用训练 step 的 batch loss 和当前 batch pattern names。**
   每个 batch 可能混合多个 pattern，因此更新时按当前 batch 出现的 pattern 都接收同一个 detached beam task loss；这是最小可审计实现。`gap_to_full` 使用 full pattern 的 EMA 作为 reference；full EMA 缺失时 fallback uniform 并 warning。`acc_gap` 先保留配置与 warning，因为当前训练 step 没有可靠 pattern-wise acc。

3. **概率计算每个 epoch 开始固定一次，epoch 结束落 CSV。**
   warmup epoch 使用 uniform。warmup 后按 `q=(1-alpha)q_uniform+alpha softmax(score/temperature)` 计算，再 clip/renormalize。epoch 内固定概率可减少 step-level 抖动，也让日志和 sanity check 更可读。

4. **beam-neighborhood loss 做成 registry loss，而不是复用 DBA-aware helper。**
   DBA-aware helper 已存在，但本任务需要明确的 `loss.type: beam_neighborhood_ce`、`mix_ce` 含义和 label smoothing baseline。新增 loss 类可在 `build_task_criterion` 原路径直接使用，不改 validation/evaluation hard-label 指标。

5. **Scene31 配置继续由 generator/manifest 表达。**
   仓库当前不跟踪 generated YAML；新增 run names 写入 generator 和 manifest，测试通过 temp 目录生成实体 YAML 检查字段。`scripts/run_scene31_bc_next.sh` 按 manifest 找 config，不存在时可自动生成。

## Risks / Trade-offs

- **Risk:** per-batch 混合 pattern 时同一个 loss 更新多个 EMA，难以精确归因。  
  **Mitigation:** 记录 `num_samples` 和 pattern counts；后续如果需要更精确，可在 batch 内按 pattern mask 拆 loss。

- **Risk:** `gap_to_full` 在 warmup 后仍缺 full EMA。  
  **Mitigation:** fallback uniform、打印 warning、继续训练，不 crash。

- **Risk:** beamsoft 提升 MAE/within-3 但 Top1 不升。  
  **Mitigation:** summary 保留 Top3/Top5/within_3/MAE（当 fresh eval 输出存在）并仍以 hard-label Top1 主指标排序。

- **Risk:** BC launcher 同时训练 AMR/AMBER baseline 时配置路径和 output root 与 Scene31 generated configs 不同。  
  **Mitigation:** baseline group 显式映射到已有 `configs/fusion/amr_net_supervised.yaml` 与 `configs/fusion/amber_full_architecture.yaml`，不要求它们出现在 Scene31 next-round manifest。

## Migration Plan

1. 添加 OpenSpec delta 和 focused tests。
2. 实现 loss 与 adaptive sampler opt-in 字段，默认配置不变。
3. 扩展 Scene31 generator/manifest 与新增 launcher/summary。
4. 运行 OpenSpec validate、loss/sampler focused tests、Scene31 generator/summary tests 和必要 CLI/config smoke。
5. 回滚时删除新增 opt-in config、loss class、adaptive sampler state 和 BC launcher；未 opt-in 的旧 run 不受影响。

## Open Questions

- `acc_gap` 是否需要真正启用取决于后续是否在训练 loop 中引入 pattern-wise accuracy 聚合；本次先 warning fallback。
- fresh eval 是否新增 `within_3`/`mae` 取决于现有 apples-to-apples 输出字段；summary 会在字段存在时保留，避免大改 eval 逻辑。
