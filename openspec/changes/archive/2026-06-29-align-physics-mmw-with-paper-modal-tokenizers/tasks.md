## 1. 前端与模型结构

- [x] 1.1 在 `pinn_multimodal_beam` 中增加可配置 frontend 类型，保留旧 stats frontend 作为兼容和 ablation。
- [x] 1.2 实现 paper-style modality tokenizer wrapper，统一输出 `[B, T, K, hidden_dim]` 或 `[B, T, hidden_dim]` token，并记录每个模态 tokenizer metadata。
- [x] 1.3 将 image tokenizer 接到 `jepa_context_image`，默认使用非 GPS context pooler，并确保 forward 不传 `gps_condition_features`。
- [x] 1.4 将 CSI/RF tokenizer 优先接到 `pilot_dual_view_csi` 或 Linear + LayerNorm，并支持 `csi_observation_mask` 作为可选输入/metadata。
- [x] 1.5 将 radar、lidar 和 GPS tokenizer 复用现有 `radar_cnn`、`lidar_cnn`、`gps_mlp` 或薄 Linear + LayerNorm wrapper。
- [x] 1.6 增加 modality embedding、time/position embedding、共享 Transformer fusion 和 token pooling/horizon adapter，输出 `[B, num_pred, hidden_dim]` latent。
- [x] 1.7 保持 direct head、path head、可微信道合成器、physics logits、hybrid logits 和 `ModelOutput` 适配兼容。

## 2. 输入边界与 metadata

- [x] 2.1 确认模型 forward 只消费 `csi_input` 或等价受限无线观测，不消费 `physics_targets.csi_target`。
- [x] 2.2 在 run/model metadata 中记录 frontend type、tokenizer type per modality、checkpoint/freeze policy、uses_gps_context、restricted wireless input、oracle flag 和 channel target scope。
- [x] 2.3 对 `Nsc=1` 的 CSI target 标记 `channel_target_scope=narrowband_array_channel`，避免报告完整宽带 CSI 重构。
- [x] 2.4 对缺少 JEPA checkpoint 的 debug/smoke 配置标记 `formal_experiment_eligible=false`，正式配置缺 checkpoint 时 fail fast。

## 3. 配置与文档入口

- [x] 3.1 新增 paper-style sparse-pilot multimodal physics MMW 配置，使用 `pinn_multimodal_beam`、paper-style frontend、`jepa_context_image` 和 sparse/restricted CSI input。
- [x] 3.2 新增最小 debug/smoke 配置，允许 synthetic 或随机初始化路径但明确不可进入正式结论。
- [x] 3.3 如需 oracle upper-bound 配置，必须要求 `data.allow_oracle_full_csi_input=true` 并写入 oracle metadata。
- [x] 3.4 更新主线实验文档或模型目录中对应条目，说明该 baseline 只声明窄带阵列信道重构。

## 4. 测试与验证

- [x] 4.1 扩展 `tests/test_physics_informed_mmw.py`，覆盖 paper-style frontend 的 registry build、synthetic forward、loss/backward 和 output adaptation。
- [x] 4.2 增加 JEPA image tokenizer 不使用 GPS context 的 focused test。
- [x] 4.3 增加 sparse pilot / restricted CSI 输入与 `csi_target` 监督分离的 focused test。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py -q` 验证模型和 loss。
- [x] 4.5 使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q` 验证配置和架构边界。
- [x] 4.6 使用 `openspec validate align-physics-mmw-with-paper-modal-tokenizers --strict` 验证 OpenSpec change。

## 5. 实验顺序

- [x] 5.1 先跑 debug/smoke，确认 forward、loss、CSI NMSE/path metrics 可记录。
- [x] 5.2 再跑 sparse-pilot multimodal 正式小规模实验，确认内存和速度可接受。
- [x] 5.3 最后跑对照矩阵：stats frontend、paper-style frontend、no-physics-loss、no-CSI-reconstruction、oracle upper-bound。
- [x] 5.4 汇总 Top-K、NBG/beamspace、CSI NMSE、path metrics，并按 metadata 排除 debug/oracle run 的主结论资格；当前结论记录为 sparse CSI 提供主要多模态增益、task-aligned PINN 仅小幅提升 Top-1、raw CSI reconstruction 为负贡献。
