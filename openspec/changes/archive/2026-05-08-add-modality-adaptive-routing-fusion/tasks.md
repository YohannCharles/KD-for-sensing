## 1. 模型与模块结构

- [x] 1.1 新增 `src/kd_sensing/models/fusion/marf.py`，注册 `marf_fusion`，并在 fusion/model 包导出，保持 `experiment.task: fusion` 的现有 forward 参数契约。
- [x] 1.2 复用或抽取 CRAF encoder、feature projection、mask、confidence helper，确保 MARF 暴露 `encoders: nn.ModuleDict`、`supports_force_modality_mask = True` 和 `supports_marf_routing = True`。
- [x] 1.3 实现 `ModalityRouter`，输出 sample-wise、horizon-wise 的 anchor/residual logits 与 weights，支持 modality mask、temperature、prior bias 开关和 prior scale。
- [x] 1.4 实现 `AnchorFusion`，使用 horizon query 对按 anchor 权重加权的模态 token 做 cross-attention，并正确应用 token padding mask。
- [x] 1.5 实现 `ResidualAdapter`，支持共享 cross-attention、模态 embedding、residual scale 和 `enabled: false` ablation。
- [x] 1.6 实现 `MARFFusionNet.forward()` 输出 dict diagnostics，确保 logits 形状为 `[B, num_pred + 1, num_classes]`，并能被 `adapt_model_output()` 正确解析。

## 2. Teacher Registry 与冻结边界

- [x] 2.1 为 MARF 实现 `set_reliability_prior()` 或等价兼容入口，使现有 `apply_teacher_priors()` 能写入 router prior。
- [x] 2.2 验证现有 `load_teacher_encoders()` 可加载 MARF `encoders`；如需调整 key 映射，仅做兼容扩展，不破坏 CRAF。
- [x] 2.3 确保 `teacher.load_encoders: true`、`teacher.freeze_encoders: true` 时 MARF encoder 冻结，router、anchor fusion、residual adapter、projection 和 head 仍可训练。
- [x] 2.4 扩展训练 runtime metadata，记录 MARF prior、每模态 encoder load summary、frozen 状态和 trainable parameter count。

## 3. MARF Loss 与 Subset-Aware Training

- [x] 3.1 新增 MARF loss helper，覆盖 residual norm、anchor prior regularization、anchor entropy 和 all-to-subset KL，并处理 ignore index。
- [x] 3.2 新增 prior-driven `ModalitySubsetSampler`，支持 `all`、`top_prior`、`single_best_prior`、`random`、`random_with_top_prior` 和 `drop_one`，不得写死 GPS/mmWave。
- [x] 3.3 在 trainer 中接入 `training.subset_training`，仅当模型支持 MARF routing 且配置启用时执行 subset forward。
- [x] 3.4 subset forward 必须通过 `force_modality_mask` 控制可用模态，subset CE 使用同一 task criterion，subset KD 使用 all-modal logits 的 detach soft target。
- [x] 3.5 扩展 epoch 日志和 TensorBoard，记录 `marf/anchor_mean/*`、`marf/residual_mean/*`、`marf/anchor_h*/*`、subset CE、subset KD、residual norm、prior regularization 和 anchor entropy。

## 4. Validation 与评估脚本

- [x] 4.1 将 validation subset 定义改为 prior-driven，并保留 `strong_only`、`weak_only` 作为兼容别名，日志中记录实际模态列表。
- [x] 4.2 确保 `all` subset 使用全部启用模态，`val/subset/all/top1` 与官方 `accuracy/val` 在同一 checkpoint、dataloader 和 criterion 下保持一致。
- [x] 4.3 新增 `scripts/debug_eval_consistency.py`，使用 `conda run -n kd_mm_beam python scripts/debug_eval_consistency.py ...` 可输出 official/subset-all Top-1、差值、样本数、batch 数、logits/labels 形状和首批预测一致性。
- [x] 4.4 新增 `scripts/eval_modality_subsets.py`，使用现有 config、checkpoint、dataloader 和 validator 输出各 subset 指标与实际模态列表。
- [x] 4.5 新增 `scripts/eval_modality_perturbation.py`，支持 `shuffle_<modality>` 和 `zero_<modality>`，输出 clean 与扰动后的 Top-1/ATop/ADBA 指标。

## 5. 配置与消融

- [x] 5.1 新增 `configs/fusion/scene32_marf.yaml`，默认 teacher encoder 初始化、冻结 encoder、`cross_entropy` + `label_smoothing: 0.03`、低权重 beam soft 和 prior regularization。
- [x] 5.2 新增 `configs/fusion/scene32_marf_subset_training.yaml`，启用 subset-aware training，默认每 batch 采样 `top_prior` 与 `random_with_top_prior`。
- [x] 5.3 新增 `configs/fusion/scene32_marf_no_residual_ablation.yaml`，只关闭 residual adapter，不改变其它实验条件。
- [x] 5.4 新增 `configs/fusion/scene32_marf_no_prior_bias_ablation.yaml`，只关闭 router prior bias，不改变其它实验条件。
- [x] 5.5 新增 `configs/fusion/scene32_marf_no_subset_training_ablation.yaml`，只关闭 subset-aware training，不改变其它实验条件。

## 6. 自动化测试

- [x] 6.1 新增 MARF forward 单元测试，覆盖 logits、anchor/residual weights、`h_anchor`、`h_final` 和 `residual_delta` 形状。
- [x] 6.2 新增 anchor softmax 与 mask 测试，验证 anchor 权重按可用模态归一、被 mask 模态 anchor/residual 权重为 0。
- [x] 6.3 新增 teacher prior 与 encoder freeze 测试，验证 MARF prior 写入、encoder 加载/冻结和 CRAF 既有测试不回退。
- [x] 6.4 新增 subset sampler、subset loss 和 validation all consistency 测试。
- [x] 6.5 新增配置加载测试，覆盖 MARF 主配置与三个 ablation 配置。
- [x] 6.6 运行 `conda run -n kd_mm_beam pytest tests/test_craf_fusion.py tests/test_teacher_prior_craf.py`，确认旧 CRAF 行为不回退。
- [x] 6.7 运行 `conda run -n kd_mm_beam pytest tests/test_marf_fusion.py tests/test_marf_training.py`，确认 MARF 新行为通过。

## 7. 实验执行顺序

- [x] 7.1 先运行 `conda run -n kd_mm_beam pytest tests/test_marf_fusion.py -k "forward or mask or prior"`，确认模型 shape、mask 和 prior 行为。
- [x] 7.2 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/scene32_marf.yaml`，确认 clean all-modal 表现接近 `teacher_init_no_prior`。
- [x] 7.3 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/scene32_marf_subset_training.yaml`，确认 subset training 不明显损害 clean Top-1。
- [x] 7.4 运行 `conda run -n kd_mm_beam python scripts/debug_eval_consistency.py --config configs/fusion/scene32_marf_subset_training.yaml --ckpt <best_checkpoint>`，确认 official validation 与 subset all 一致。
- [x] 7.5 运行 `conda run -n kd_mm_beam python scripts/eval_modality_perturbation.py --config configs/fusion/scene32_marf_subset_training.yaml --ckpt <best_checkpoint>`，评估各模态 shuffle/zero 扰动。
- [x] 7.6 依次运行 no residual、no prior bias、no subset training ablation，并与 `scene32_teacher_init_no_prior_ablation`、`scene32_stage2_teacher_init_prior_residual` 对比。
