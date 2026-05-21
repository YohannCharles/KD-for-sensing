## Context

`add-csi-hardening-experiment-matrix` 已经完成了 CSI hardening 矩阵、pilot dual-view CSI encoder 扩展和分析脚本。但当前 TensorBoard 结果显示：除了 `A0_clean_full_teacher`，大多数 A/B/C/D 变体都停在 `accuracy_val≈0.13~0.15`、`val_adba≈0.50`。其中 `C1_view_gate_warmup` 和 `C2_no_internal_gru` 理论上不应改变输入 CSI，`A1_mild_pilot_estimation` 也不应把可学习性直接打到随机水平。

这说明首要问题不是继续扩大 sweep，而是验证非 A0 run 是否真正继承了 A0 的数据、模型、训练和 CSI encoder 路径。本 change 将排查逻辑固化为 OpenSpec 契约和实施任务，先生成 A0 clone 与最小 debug 矩阵，再增加必要日志和不变量检查。

## Goals / Non-Goals

**Goals:**

- 确认 `A0_clone_generated` 通过当前 sweep/config 生成器后是否与原始 A0 在关键配置上等价。
- 在 run 启动时保存并打印 resolved config、模型结构、CSI encoder 关键开关、参数规模和 trainable params by module。
- 在 CSI forward 首个 train/val batch 记录 hardening/pilot/view/tokenizer/fusion/GRU 的 shape、统计量和 feature norm。
- 对 pilot disabled、hardening shape/stat、view gate warmup、`use_internal_gru`、训练梯度与参数更新建立明确诊断标准。
- 只运行 5 个 10~20 epoch debug run，先定位配置继承、CSI encoder 新路径、pilot estimator、hardening 或训练连接问题。

**Non-Goals:**

- 不继续新增完整 CSI hardening 变体或解释当前完整 sweep 的实验结论。
- 不改变 G2D 的理论设定，也不把 hardening 目标改成 destructive degradation。
- 不重新设计 beam label、CSI 数据集或训练任务定义。
- 不引入新的外部日志系统；优先复用 run metadata、训练日志和 TensorBoard。

## Decisions

1. 先做 generated A0 clone，而不是直接修某个可疑模块。

   当前异常模式是非 A0 变体集体失效，最可能是配置继承或新生成路径整体偏离 A0。`A0_clone_generated` 必须通过当前 sweep/config 生成器产生，但关闭所有新选项：`csi_hardening.enabled=false`、`csi_degradation.enabled=false`、pilot noise disabled、`use_internal_gru=true`、`view_fusion=symmetric_gate`、warmup epoch 为 0、`representation_core=single_gru`。如果 clone 也低到 0.14，后续无需分析 hardening 强度，先修配置生成。

   备选方案是逐个检查 B/C/D 配置。该方式会被多个变量交织影响，无法最快判断“新路径整体错误”。

2. resolved config diff 使用 allowlist 机制。

   A0 original 与 generated clone 的 diff 必须允许 `run_name`、`output_dir`、`seed`、日志目录等运行身份字段不同，但 optimizer、scheduler、loss、数据 split、`seq_len`、`num_pred`、`num_classes`、CSI encoder、representation core、beam head、normalization 和 train RMS path 等关键字段必须一致。diff 结果写入 run artifact，关键字段不一致时 debug run 应标记为 failed parity。

   备选方案是人工目视比较 YAML。该方式容易漏掉默认补全、alias 复制和 CLI override 后的差异。

3. CSI forward debug 只记录首个 train batch 和首个 val batch。

   需要检查的指标包括 raw/after_hardening/after_pilot 的 shape、dtype、abs_mean、abs_std、abs_max、real_mean、imag_mean、nan_count、zero_ratio，freq/delay view 的 shape、mean、std、nan_count，以及 freq_feat、delay_feat、fused_feat、gru_out、final csi feature norm。只在首个 batch 记录可以避免训练日志爆炸，同时足以发现 shape 广播、全零、NaN、pilot 噪声量级和 warmup 融合错误。

   备选方案是在每个 batch 记录完整 tensor 统计。该方式对性能和日志体积不划算。

4. pilot 和 hardening 的诊断使用物理量级约束。

   当 `pilot_estimator.enabled=false` 时，`max_abs(h_hat - h)` 必须为 0 或浮点精度内的 0。A1 的 25~35 dB estimation SNR 应产生约 `0.003~0.0003` 的 `noise_power/signal_power`，明显接近 1 或大于 1 时视为噪声方差计算错误。hardening 后必须保持 `[B,T,Nsc,Nant]`，finite，且 abs_mean/abs_std 相对输入不应无解释地漂移超过 20%。

   备选方案是只看最终 accuracy。该方式无法区分 slow learning 与数据流损坏。

5. 训练健康指标同时看 grad norm 和 param delta。

   只看 `requires_grad` 或 trainable parameter count 不能证明 loss 真的接到了模块。每个 epoch 记录 `grad_norm_csi_encoder`、`grad_norm_representation_core`、`grad_norm_beam_head` 以及对应 param delta，能发现冻结、optimizer 参数组遗漏、SMP/屏蔽逻辑误伤、loss 未连接等问题。

   备选方案是只打印 trainable params。该方式无法发现梯度为 0 或 optimizer 没更新。

## Risks / Trade-offs

- [Risk] debug logging 改变训练性能或污染正式 sweep 日志。→ Mitigation：通过显式 debug flag 启用，默认关闭，并限制为首个 train/val batch 与 epoch 级标量。
- [Risk] config diff 误报运行身份字段。→ Mitigation：使用明确 allowlist，只把 run identity、seed 和输出目录视作可接受差异。
- [Risk] hardening 的 20% 统计漂移阈值对某些 gain scaling 配置过严。→ Mitigation：仅作为 debug 警告或失败原因记录，允许明确标注 gain scaling 的配置放宽该检查。
- [Risk] param delta 计算增加一次参数快照开销。→ Mitigation：只在 debug run 启用，并按模块聚合，不保存完整参数差异。
- [Risk] 10~20 epoch debug run 不能证明最终 ceiling。→ Mitigation：本 change 的目标是定位明显路径错误；clone parity 和 feature/grad health 通过后再恢复长 sweep。

## Migration Plan

1. 增加 debug run 配置或生成器入口，生成 `A0_original`、`A0_clone_generated`、`A0_clone_generated + pilot disabled through new path`、`C1_view_gate_warmup_only`、`C2_no_internal_gru_only`。
2. 在配置解析与 run metadata 中保存 `resolved_config.yaml`，并实现 A0 original/clone diff artifact。
3. 在 CSI encoder 中加入受 debug flag 控制的首 batch 统计和 pilot/hardening 不变量记录。
4. 在 trainer 中加入 run-start 模型摘要、module trainable params、epoch grad norm 和 param delta 记录。
5. 运行单元测试和 5 个短 debug run；只有 `A0_clone_generated` 接近 `A0_original` 后，才恢复 hardening 矩阵分析。

Rollback 策略：关闭 debug flag 并移除 debug run 配置即可回到当前训练行为；默认禁用的日志与诊断不应影响现有实验。

## Open Questions

- 当前项目中原始 A0 配置的权威入口是实体 YAML、生成后的 resolved config，还是已跑 run 的 artifact？实现时需要优先选择最接近实际训练的 artifact 作为 diff 基准。
- “接近 A0”的短跑判据是使用 10~20 epoch 的 accuracy 曲线相对趋势，还是要求达到某个绝对阈值？首版建议把绝对阈值作为人工判断，把 parity 和数据/梯度健康作为自动 gate。
