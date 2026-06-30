## 1. 模型组件

- [x] 1.1 在 `src/kd_sensing/models/` 中新增或拆出 AMBER full core owner，注册 `amber_full_adaptive_mask_transformer` representation core。
- [x] 1.2 实现 image/LiDAR/radar/GPS/projected feature 与历史 beam embedding 的 token 组装、time/modality positional embedding 和 learnable fusion token。
- [x] 1.3 实现缺失模态 availability mask 到 attention mask 的转换，确保 fusion token 不 attend 到 unavailable modality tokens。
- [x] 1.4 实现 modality-specific transformer branch 和 modality-fusion transformer branch，并保持主输出可被现有 beam head 消费。
- [x] 1.5 为 AMBER full core 提供 `training_strategy_metadata()`，记录 scope、模态、mask strategy、CMA、loss weights 和 missing metadata consumption。

## 2. 训练输出与损失

- [x] 2.1 扩展 AMBER full forward auxiliary payload，包含 modality-specific features、fusion features/token、CMA embeddings/logits 和 mask provenance。
- [x] 2.2 在现有 loss/objective 扩展点中新增 AMBER L2/alignment loss 与 CMA contrastive loss helper，不复制训练循环。
- [x] 2.3 接入加权 total loss 标量记录，保留 beam focal loss 主路径，并在 AMBER auxiliary payload 缺失时早失败。
- [x] 2.4 验证普通 non-AMBER 配置不要求 CMA、L2 或 missing-modality auxiliary payload。

## 3. 配置与文档

- [x] 3.1 新增 AMBER full 本地配置，默认使用 `kd-sensing-train --config`、scratch/local weights 和 `outputs/analysis/local_baselines/amber_full_architecture/` 输出边界。
- [x] 3.2 更新 AMBER-lite 与 AMBER full 的 metadata/summary helper 或 claim row，使 `amber_lite_local` 和 `amber_full_local` scope 不混淆。
- [x] 3.3 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md`，记录 AMBER full pending/local caveat。
- [x] 3.4 如实现新增模型 owner 或文档生命周期条目，更新 `docs/project_surface_inventory.md` 的对应 current/local baseline 说明。

## 4. 测试与验证

- [x] 4.1 新增 synthetic forward tests，覆盖 AMBER full registry/config build、beam logits shape、training/eval mode auxiliary payload 和 `adapt_model_output`。
- [x] 4.2 新增 mask attention tests，验证 single-missing、多模态 missing 和 all-but-one available 情况下 fusion token 屏蔽 unavailable tokens。
- [x] 4.3 新增 loss tests，覆盖 L2/alignment、CMA contrastive、加权 total loss 和 missing payload 早失败。
- [x] 4.4 新增 metadata/architecture summary tests，验证 AMBER full 参数统计、component role、scope 和 ordinary baseline 不消费 AMBER metadata。
- [x] 4.5 运行 `openspec validate reproduce-amber-full-architecture --strict`。
- [x] 4.6 运行 `conda run -n kd_mm_beam pytest tests/test_amber_lite_missing_modality.py -q` 和新增 AMBER full focused tests。
- [x] 4.7 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py -q`。
