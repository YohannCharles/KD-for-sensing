## 1. 模型与注册

- [x] 1.1 梳理现有 fusion token 组件和 encoder helper，决定新增共享 helper 还是独立实现，避免新模型依赖 CRAF 私有细节。
- [x] 1.2 新增 `cls_token_transformer_fusion` 模型模块，实现可配置的 `modalities`、`feature_size`、`d_model`、`num_heads`、`num_layers`、`dropout`、`max_seq_len` 和 `num_pred` 参数。
- [x] 1.3 为启用模态构建现有 encoder/projector，输出统一为 `[B, T, d_model]`，并校验 batch/time 维一致。
- [x] 1.4 将新模型注册到 `MODELS`，并更新默认组件导入流程，确保 `import_default_components()` 后可构建 `cls_token_transformer_fusion`。

## 2. CLS-token Transformer 融合实现

- [x] 2.1 实现 token 序列化：将 `[B, K, T, d_model]` 按时间优先顺序转换为 `[B, T*K, d_model]`，五模态时每个时间步保留五个独立模态 token。
- [x] 2.2 实现可学习 CLS token，并将其前置到 Transformer 输入序列，形成 `[B, 1 + T*K, d_model]`。
- [x] 2.3 实现 token-type embedding，覆盖所有模态 token 和独立 CLS token 类型。
- [x] 2.4 实现 time embedding，确保不同历史时间步的相同模态 token 可区分，且 CLS token 不绑定错误的历史时间。
- [x] 2.5 实现 Transformer Encoder 融合层，使用多头自注意力和前馈网络处理完整 token 序列。
- [x] 2.6 实现 horizon prediction head，从 CLS hidden state 输出 `[B, num_pred, num_classes]` logits，并保持 `select_prediction_slots()` 兼容。
- [x] 2.7 实现 `force_modality_mask` 支持，通过 attention padding mask 排除被屏蔽模态 token，并在无可用模态时抛出清晰错误。
- [x] 2.8 返回 `logits`、`input_features`、`output_features`、`token_features`、`modalities`、`effective_modality_mask` 和 `fusion_memory` diagnostics，确保 `adapt_model_output()` 与 G2D diagnostics 可解析。

## 3. 默认配置与兼容入口

- [x] 3.1 更新推荐/default fusion student no-KD 配置或 canonical 生成规则，使默认混合方式使用 `cls_token_transformer_fusion`。
- [x] 3.2 更新推荐/default fusion logits KD 和 RKD 配置或 canonical 生成规则，使可训练 student 使用 `cls_token_transformer_fusion`，teacher 继续使用明确的 teacher baseline。
- [x] 3.3 增加或更新五模态默认 fusion no-KD 配置，启用 image、radar、GPS、LiDAR 和 mmWave 数据字段，并设置 CLS-token Transformer 默认超参数。
- [x] 3.4 确认显式 legacy early-concat、模块化 `early_concat_gru`、CRAF、MARF 和 G2D 配置不被默认行为覆盖。
- [x] 3.5 更新 README、扩展指南或配置说明，说明默认 fusion 方法已切换为 CLS-token Transformer，并列出 legacy/advanced baseline 的显式入口。

## 4. 测试

- [x] 4.1 新增模型单元测试，覆盖五模态 forward shape：`logits [B, H, C]`、`token_features [B, K, T, D]` 和 CLS/Transformer memory diagnostics。
- [x] 4.2 新增任意模态子集测试，覆盖 image+gps、radar+gps、单个强模态等合法组合。
- [x] 4.3 新增 `force_modality_mask` 测试，验证被屏蔽模态 token 不参与 attention，且空可用模态抛出清晰错误。
- [x] 4.4 新增 registry 测试，使用 `conda run -n kd_mm_beam pytest ...` 验证默认组件导入后可构建 `cls_token_transformer_fusion`。
- [x] 4.5 新增配置加载测试，使用 `conda run -n kd_mm_beam pytest ...` 验证默认 fusion 配置选择新模型，显式 legacy/CRAF/MARF 配置保持原模型。
- [x] 4.6 新增 G2D diagnostics 或模型输出适配测试，验证 `adapt_model_output()` 能解析新模型输出并按模态拆分 token features。

## 5. 验证

- [x] 5.1 运行定向单元测试：`conda run -n kd_mm_beam pytest tests/test_cls_token_transformer_fusion.py tests/test_component_registry.py tests/test_student_configs.py`。
- [x] 5.2 运行 fusion 相关回归测试：`conda run -n kd_mm_beam pytest tests/test_marf_fusion.py tests/test_craf_fusion.py tests/test_g2d_smp.py tests/test_fusion_image_feature_extractor.py`。
- [x] 5.3 运行 OpenSpec 校验：`openspec validate add-cls-token-transformer-fusion-default --strict`。
- [x] 5.4 进行一次小规模配置 smoke run 或 dry-run，使用 `conda run -n kd_mm_beam python scripts/train.py --config <默认fusion配置> ...` 验证配置可构建、forward 可运行、输出目录写入完整 `final_config.yaml`。
