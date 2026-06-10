## 1. Objective 与配置入口

- [x] 1.1 在 objective metadata/registry 中新增 `gps_conditioned_jepa`，设置默认 metric `val_jepa_loss`、mode `min`、available metrics、history fields、TensorBoard scalar 和 runtime metadata。
- [x] 1.2 调整配置 normalization/validation，使 `experiment.objective: gps_conditioned_jepa` 可解析，并拒绝 JEPA 下不可用的 beam/occlusion/position/LOS/link early stopping metric。
- [x] 1.3 增加 JEPA canonical smoke 配置，启用 image RGB/ImageNet profile、GPS relative-polar、`model.primary.type: gps_conditioned_jepa` 和输出目录 metadata。
- [x] 1.4 增加 paper-split 风格 full low-memory 配置，训练拼接 DeepSense6G scenes 32-34，验证/监控 scenes 31-34，并保留 scene31 仅作 smoke/诊断口径。

## 2. JEPA 模型与采样组件

- [x] 2.1 新增 `src/kd_sensing/models/jepa.py`，实现可注册 `gps_conditioned_jepa` 主模型和轻量 visual patch/token encoder。
- [x] 2.2 实现 context encoder 与 EMA target encoder 初始化、冻结、checkpoint state dict 兼容和 `update_target_encoder()` 方法。
- [x] 2.3 实现 GPS conditioner，至少支持 `film` 和 `concat_mlp`，并校验 GPS feature 维度。
- [x] 2.4 实现 predictor，使 conditioned context latent 预测 target latent，并输出 `predicted_target_latent`、`target_latent`、mask 和 diagnostics。
- [x] 2.5 实现 JEPA mask sampler，支持 `random` 与 `gps_angle_biased`，保证 context/target 非重叠且可由 seed/epoch/step 复现。
- [x] 2.6 将 JEPA 模型加入默认组件导入和 `kd_sensing.models` 延迟导出，不扩大 registry 轻量导入依赖面。

## 3. Loss、训练扩展与验证

- [x] 3.1 新增 JEPA latent loss helper，支持 masked MSE、可选 latent normalization、SmoothL1/Huber 变体和 NaN/空 mask 防护。
- [x] 3.2 扩展 model output/runtime，使 `gps_conditioned_jepa` objective 下可处理 self-supervised payload，而不要求 beam logits。
- [x] 3.3 实现 JEPA training extension，提供 base loss、diagnostics，并在 optimizer step 后触发 EMA target encoder update。
- [x] 3.4 为 `TrainingExtension` 增加 `after_optimizer_step` hook，并在 AMP 与非 AMP 路径下均按 optimizer step 后顺序调用。
- [x] 3.5 扩展 validation/evaluation pass，使 JEPA objective 只计算 `val_jepa_loss` 和 `val_loss`，不产生 beam Top-K/DBA 指标。
- [x] 3.6 更新 artifact writer 或 runtime metadata，记录 JEPA mask sampler、EMA decay、context encoder artifact key 和 pretraining kind。

## 4. 测试与文档

- [x] 4.1 添加模型单元测试：registry build、forward shape、缺 GPS 报错、target latent detach、EMA 参数更新。
- [x] 4.2 添加 mask sampler/loss 单元测试：random 与 GPS-biased mask 非重叠、可复现、masked loss 有效、空 mask 防护。
- [x] 4.3 添加 objective/config 测试：JEPA objective metadata、early stopping metric 校验、canonical config 可加载且不含 KD 字段。
- [x] 4.4 添加训练 smoke 测试：使用 synthetic 或小 batch 运行 1 epoch，验证 forward/backward/optimizer/EMA/validation/checkpoint 和 `val_jepa_loss`。
- [x] 4.5 更新 README 或实验矩阵文档，说明 JEPA 预训练入口、配置路径、输出产物和 checkpoint 复用边界。
- [x] 4.6 更新 DataFactory/metadata，使 DeepSense6G 多场景 full 配置可拼接训练集并记录 source scene 信息。

## 5. 验证

- [x] 5.1 运行 `openspec validate add-gps-conditioned-jepa-pretraining --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.3 运行 JEPA 相关单元测试，例如 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_objective_metadata.py -q`。
- [x] 5.4 运行 CLI 快速检查：`conda run -n kd_mm_beam kd-sensing-train --help`。
- [x] 5.5 变更完成后运行最终回归：`conda run -n kd_mm_beam pytest -q`。

## 6. 下游复用验证

- [x] 6.1 增加 `jepa_context_image` supervised image encoder，支持从 JEPA `best.pth` 和 `last.pth` 抽取 `context_encoder.*` 初始化。
- [x] 6.2 增加 image+GPS supervised A/B 配置：baseline、JEPA random best、JEPA random last、JEPA GPS-biased best。
- [x] 6.3 增加单元测试覆盖 JEPA context encoder checkpoint 加载和 `[B,T,D]` 输出契约。
- [x] 6.4 使用 tmux 在 GPU0/2/3 启动 baseline 与 JEPA 初始化下游训练，并记录 session 名称。
- [x] 6.5 OOM 后将下游 A/B 配置切换为低内存运行：batch 4、`num_workers=0`、不叠同一 GPU 并发，并重新启动 lowmem run。

## 7. BeamBench-fair 下游复核

- [x] 7.1 增加 supervised image+GPS fair low-memory 配置族，训练 scenes 32-34，final test scenes 31-34，并将 JEPA checkpoint 路径更新为 `outputs/deepsense6g_gps_conditioned_jepa_full_s32_s34_lowmem` 与 `outputs/deepsense6g_gps_conditioned_jepa_gps_biased_s32_s34_lowmem`。
- [x] 7.2 增加内部 validation-from-train split，训练过程优先用 validation split 做 early stopping/checkpoint selection，训练结束后单独加载 best checkpoint 在 test split 上汇报 final metrics。
- [x] 7.3 增加 BeamBench linear DBA 距离模式，并在 fair 配置中设置 `evaluation.dba_distance_mode: linear`，保留 circular DBA 为默认行为。
- [x] 7.4 将 fair 配置的 scheduler 设置为 `none`，避免 warm-restart 学习率对论文表格比较产生额外变量；保留 `num_pred: 1` 且不修改 `seq_len`。
- [x] 7.5 增加测试覆盖 linear DBA 和内部 validation split 的 train-only GPS scaler 复用。

## 8. 2604.05668 对齐下游复核

- [x] 8.1 增加 DeepSense6G `stratified_80_10_10` split protocol，支持按 scene 合并官方 train/test CSV 后用 `future_beam1` 标签分层切分 train/validation/test。
- [x] 8.2 确保该 split protocol 的 GPS scaler 只在 80% train 子集上拟合，并复用于 validation/test。
- [x] 8.3 增加 image+GPS 2604 对齐配置族：baseline、JEPA random best、JEPA GPS-biased best，使用 S32/S33/S34、`seq_len: 5`、`num_pred: 1` 和 linear DBA。
- [x] 8.4 增加测试覆盖 2604 split protocol 的样本数、metadata 和 train-only GPS scaler 复用。
