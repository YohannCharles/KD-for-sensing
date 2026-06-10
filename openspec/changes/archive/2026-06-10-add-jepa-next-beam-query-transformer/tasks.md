## 1. Core 实现与契约测试

- [x] 1.1 在 `src/kd_sensing/models/modular.py` 新增并注册 `NextBeamQueryTransformerCore`，支持 `d_model`、`modality_count`、`num_heads`、`num_layers`、`dropout`、`max_seq_len` 和 `output_dim` 配置。
- [x] 1.2 为 `NextBeamQueryTransformerCore` 实现 modality embedding、time embedding、learned next-beam query token、Transformer 编码和 `[B, 1, D_out]` 输出。
- [x] 1.3 为 core 增加输入校验：拒绝非 `[B, K, T, D]` 输入、错误 `K/D`、超过 `max_seq_len` 的时间维，并提供清晰错误信息。
- [x] 1.4 添加 focused 单元测试，覆盖 core forward shape、query 输出时间维、错误输入和 `REPRESENTATION_CORES.build()` 注册构建。
- [x] 1.5 使用 `conda run -n kd_mm_beam pytest tests/test_cls_token_transformer_fusion.py tests/test_modular_sequence_next_query_transformer.py -q` 或等价新增测试文件运行 core 级验证。

## 2. ModularSequenceModel 集成

- [x] 2.1 确认 `ModularSequenceModel` 多模态 stack 路径可直接向 `next_beam_query_transformer` 传入 `[B, K, T, D]`，并保持 `input_features`、`output_features`、`modalities`、`modality_features` 和 `encoder_features` 输出字段。
- [x] 2.2 添加集成测试：构建 image/GPS synthetic `modular_sequence`，使用 `next_beam_query_transformer` 和 `beam_head`，验证 logits 形状为 `[B, 1, num_classes]`。
- [x] 2.3 添加集成测试：确认现有 `early_concat_gru`、`snapshot_frame` 和 `token_transformer` core 行为不被新增 core 破坏。
- [x] 2.4 使用 `conda run -n kd_mm_beam pytest tests/test_cls_token_transformer_fusion.py tests/test_snapshot_next_frame_baselines.py tests/test_gps_conditioned_jepa.py -q` 运行模块化模型相关回归。

## 3. JEPA 下游配置矩阵

- [x] 3.1 新增或调整 JEPA downstream 配置，覆盖 `jepa_gru`、`jepa_snapshot`、`jepa_plain_token_transformer` 和 `jepa_next_query_transformer` 四组 ablation。
- [x] 3.2 四组配置 MUST 尽量复用相同 JEPA checkpoint、`jepa_context_image` encoder、GPS MLP、projectors、beam head、beam label space 和训练 recipe。
- [x] 3.3 为 `jepa_snapshot` 配置显式设置 `seq_len=1` 和 `num_pred=1`，避免违反 `snapshot_frame` 契约。
- [x] 3.4 为 `jepa_next_query_transformer` 配置显式记录 `representation_core.type=next_beam_query_transformer`、time embedding、modality embedding、query token 和 ablation 名称。
- [x] 3.5 若新增配置位于 `configs/fusion/` 根目录，更新对应 inventory/architecture guardrail；否则放入明确实验子目录并保持 README/测试引用一致。

## 4. Metadata 与配置验证

- [x] 4.1 更新 run metadata 或 final config 写出逻辑，使 JEPA downstream 运行记录 ablation 名称、core 类型、JEPA checkpoint、freeze image encoder、time embedding、modality embedding 和 next-query 启用状态。
- [x] 4.2 添加配置加载测试，确认四组 JEPA downstream ablation 配置可解析，且 objective 为 supervised beam prediction/default beam objective。
- [x] 4.3 添加 forward smoke 测试，使用 synthetic image/GPS batch 构建四组 ablation 模型并验证 logits shape。
- [x] 4.4 确认新增路径不引用 HiST/Hist、KD distillation、teacher_no_kd、student_no_kd、no_kd、logits_kd 或旧兼容入口。
- [x] 4.5 使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_retired_encoder_entries.py tests/test_gps_conditioned_jepa.py -q` 运行配置和退役入口回归。

## 5. 验收

- [x] 5.1 运行 `openspec validate add-jepa-next-beam-query-transformer --strict`。
- [x] 5.2 运行 `openspec status --change add-jepa-next-beam-query-transformer`，确认任务状态和 artifact 状态可读。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认配置面或 inventory 未产生新漂移。
- [x] 5.4 运行新增和相关 focused tests：`conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_gps_conditioned_jepa.py -q`。
- [x] 5.5 如实现触及训练/评估公共路径，补充运行 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_evaluation_pass.py -q`。
- [x] 5.6 汇总新增 core、四组 ablation 配置、metadata 字段和未运行的长耗时真实训练实验。
