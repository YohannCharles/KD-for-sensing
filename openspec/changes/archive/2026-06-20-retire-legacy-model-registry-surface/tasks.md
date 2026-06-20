## 1. 退役清单与基线审计

- [x] 1.1 生成并提交实现期退役清单 fixture，列出每个旧名、registry、退役批次、迁移目标和错误提示文本。
- [x] 1.2 用引用扫描确认第一批名称无 current config 依赖：`modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer`、`safe_residual_reranker`、`point_cloud_mlp`、`radar_feature_extractor`、`lidar_feature_extractor`、`mmwave_feature_extractor`。
- [x] 1.3 用引用扫描确认第二批名称的实体 config 依赖范围：`image_strong`、`image_lightweight`、`radar_strong`、`radar_lightweight`、`gps_strong`、`gps_lightweight`、`lidar_strong`、`lidar_lightweight`、`mmwave_strong`、`mmwave_lightweight`、`fusion_lightweight`、`fusion_strong`。
- [x] 1.4 明确 deferred 保留清单并写入实现说明：`cls_token_transformer_fusion`、`token_transformer_fusion`、`vision_position_late_fusion`、`vision_position_transformer_fusion`、`gps_sequence_baseline` 如未退役，必须说明保留理由。

## 2. Canonical 配置迁移

- [x] 2.1 将 `configs/radar/strong.yaml` 和 `configs/radar/supervised.yaml` 迁移到 `model.primary.type: modular_sequence`、`encoders.radar.type: radar_cnn`、`representation_core.type: single_gru`、`heads.beam.type: beam_head`。
- [x] 2.2 将 `configs/radar/lightweight.yaml` 迁移到同一 modular radar 路径，并用配置参数表达 lightweight 差异。
- [x] 2.3 将 `configs/gps/strong.yaml`、`configs/gps/supervised.yaml` 和 `configs/gps/ablation_relative_polar.yaml` 迁移到 `modular_sequence`、`encoders.gps.type: gps_mlp`、`single_gru`、`beam_head`。
- [x] 2.4 将 `configs/gps/lightweight.yaml` 迁移到 modular GPS 路径，并用配置参数表达 lightweight 差异。
- [x] 2.5 将 `configs/mmwave/strong.yaml` 和 `configs/mmwave/supervised.yaml` 迁移到 `modular_sequence`、`encoders.mmwave.type: mmwave_mlp`、`single_gru`、`beam_head`。
- [x] 2.6 将 `configs/mmwave/lightweight.yaml` 迁移到 modular mmWave 路径，并用配置参数表达 lightweight 差异。
- [x] 2.7 将 `configs/fusion/radar_gps_supervised.yaml` 迁移到 `modular_sequence`，启用 `radar_cnn`、`gps_mlp`、projectors、`early_concat_gru` 和 `beam_head`。
- [x] 2.8 检查 `configs/image/{strong,lightweight,supervised}.yaml` 和 `configs/lidar/{strong,lightweight,supervised}.yaml` 已使用 modular path，清理残留旧注册字段或 misleading 注释。

## 3. Registry 退役实现

- [x] 3.1 将 `MODELS` 中的 `modular_sequence_model`、`gps_only_neural_baseline`、`radar_feature_extractor`、`lidar_feature_extractor`、`mmwave_feature_extractor` 改为 removed guard 或从 current 注册路径移除并在合适位置登记 `register_removed(...)`。
- [x] 3.2 将旧单模态 whole-model 注册名 `image_strong`、`image_lightweight`、`radar_strong`、`radar_lightweight`、`gps_strong`、`gps_lightweight`、`lidar_strong`、`lidar_lightweight`、`mmwave_strong`、`mmwave_lightweight` 改为 removed guard。
- [x] 3.3 将旧 fusion whole-model 注册名 `fusion_lightweight` 和经确认可退役的 `fusion_strong` 改为 removed guard。
- [x] 3.4 将 `REPRESENTATION_CORES` 的 `jepa_token_transformer` 别名改为 removed guard，并指向 `token_transformer` 或 `token_aware_transformer`。
- [x] 3.5 将 `HEADS` 的 `safe_residual_reranker` 别名改为 removed guard，并指向 `safe_residual_beam_reranker`。
- [x] 3.6 处理 `ENCODERS.point_cloud_mlp`：若无 current 依赖则改为 removed guard；若保留则从 current inventory 表中移出并标为 experimental/deferred，补测试证明其边界。
- [x] 3.7 确保 `import_default_components()` 后 current `MODELS.list()`、`ENCODERS.list()`、`REPRESENTATION_CORES.list()`、`HEADS.list()` 不包含已退役名称。

## 4. 模型构建与 forward 测试

- [x] 4.1 新增或更新 registry removed guard tests，覆盖每个退役名称的 registry、错误类型、请求名称和迁移提示。
- [x] 4.2 新增 modular radar config build/forward smoke，使用 synthetic radar batch 验证 logits shape、input/output feature contract 和 `ModelOutput` 适配。
- [x] 4.3 新增 modular GPS config build/forward smoke，使用 synthetic GPS batch 验证 logits shape、GPS input dim 和 future label 对齐不回归。
- [x] 4.4 新增 modular mmWave config build/forward smoke，使用 synthetic mmWave batch 验证 64 维输入和 logits shape。
- [x] 4.5 新增 modular radar+GPS fusion config build/forward smoke，验证只需要 radar/GPS batch 字段并输出 beam logits。
- [x] 4.6 更新 image/LiDAR focused tests，确认旧注册名退役后 canonical image/LiDAR configs 仍可加载并构建 modular 模型。
- [x] 4.7 更新 model architecture summary tests，确认 removed 名称不出现在 current summary/inventory 输出中，且 migrated config 可生成架构摘要。

## 5. 配置、边界和治理测试

- [x] 5.1 更新 `tests/test_config_load_characterization.py`，覆盖迁移后的 radar/GPS/mmWave/fusion root configs。
- [x] 5.2 更新 `tests/test_architecture_boundaries.py` 的 model registration allowlist 和 removed guard 断言。
- [x] 5.3 更新任何仍期待旧 `*_strong`、`*_lightweight`、`*_feature_extractor` `MODELS` 名称的 focused tests。
- [x] 5.4 确认 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 通过。
- [x] 5.5 确认 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 通过，或仅因已知无关工作树治理漂移失败并在最终说明记录。

## 6. 文档和索引收口

- [x] 6.1 更新 `docs/model_architecture_inventory.md`，从 current 表中移除退役名称，并新增 legacy migration 表。
- [x] 6.2 更新 `docs/project_surface_inventory.md`，记录本 change 对模型 registry surface 的收口、保留例外和 deferred cleanup。
- [x] 6.3 更新 `docs/extension_guide.md`，删除直接新增/使用旧 whole-model strong/lightweight 的示例，强调 `modular_sequence` 路径。
- [x] 6.4 更新 `docs/maintainer_context_index.yaml` 的 model registration allowlist、retired route guard 和验证命令。
- [x] 6.5 更新 README 或相关 quickstart 文档中仍引用旧 registry 名称的段落，保留 config 路径但说明主模型已是 modular。

## 7. OpenSpec 与 lifecycle 同步

- [x] 7.1 检查本 change 的 spec delta 与 current specs 不再冲突，特别是 GPS/radar/LiDAR/mmWave teacher/student wording。
- [x] 7.2 运行 `openspec validate retire-legacy-model-registry-surface --strict`。
- [x] 7.3 如实现过程中发现 `token_transformer_fusion`、`vision_position_*` 或 `gps_sequence_baseline` 也应退役，先更新本 change proposal/design/spec/tasks，再继续代码修改。
- [x] 7.4 如实现过程中决定保留 `fusion_strong` 或 `point_cloud_mlp`，必须更新 design 的 Open Questions、spec delta 和 tasks，说明保留理由和后续治理边界。

## 8. 最终验证与交付说明

- [x] 8.1 运行 `conda run -n kd_mm_beam pytest tests/test_model_architecture_summary.py -q`。
- [x] 8.2 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py -q` 或相关 modular focused tests。
- [x] 8.3 运行迁移涉及的新增/更新 focused tests，例如 radar/GPS/mmWave/fusion config forward tests。
- [x] 8.4 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`，确认未影响 CLI discoverability。
- [x] 8.5 汇总最终退役清单、保留例外、迁移后的 config 路径、验证命令结果和剩余 caveat。
