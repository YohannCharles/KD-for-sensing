## 1. 规格与边界确认

- [x] 1.1 对照本 change 的 specs，确认删除范围只覆盖旧模态子集/扰动脚本入口、fusion KD virtual alias 和 no-KD 配置中的 KD-only 字段。
- [x] 1.2 确认保留能力清单：`evaluation.modality_subsets`、`force_modality_mask`、viewer manifest、Raymobtime s008、CSI hardening、HiST-Beam/MMW LOSO、单模态 legacy KD 实体配置和 distillation tensor-level helper。
- [x] 1.3 运行 `openspec validate prune-obsolete-modality-imbalance-surface --strict`，先确认 OpenSpec artifact 合法。

## 2. 脚本入口与 inventory 收窄

- [x] 2.1 删除 `scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py`，或将同等能力迁入包内 evaluate 路径后移除脚本入口。
- [x] 2.2 更新 `tests/test_architecture_boundaries.py` 的 Python entrypoint allowlist，拒绝旧脚本回流。
- [x] 2.3 更新 `docs/project_surface_inventory.md`，移除旧模态子集/扰动诊断脚本分类，并保留当前主线入口说明。
- [x] 2.4 确认 `evaluation.modality_subsets` 的共享 evaluation pass 仍有测试覆盖；必要时新增 focused test。

## 3. KD virtual 配置与 no-KD 配置瘦身

- [x] 3.1 调整 `src/kd_sensing/config/canonical.py`，从 canonical fusion virtual modes 中移除 `logits_kd` 和 `rkd`。
- [x] 3.2 调整 `src/kd_sensing/config/canonical_recipes/fusion.py`，停止为 fusion virtual config 生成 legacy KD overrides，并保留 no-KD 主线 recipe。
- [x] 3.3 为请求不存在实体 YAML 的 fusion `logits_kd` / `rkd` 路径增加清晰错误，说明 legacy KD fusion virtual alias 已退役。
- [x] 3.4 清理 no-KD 实体 YAML 和 virtual no-KD 输出中的 KD-only 字段：`temperature`、`alpha`、`rkd_pairs_per_anchor`、`rkd_distance_weight`、`rkd_angle_weight`。
- [x] 3.5 保留单模态 `configs/{image,radar,gps,lidar,mmwave}/{logits_kd,rkd}.yaml` 的 legacy lineage，确认运行时 metadata 仍标记为 supplemental baseline。

## 4. 测试更新

- [x] 4.1 更新 `tests/test_student_configs.py`，移除 fusion virtual KD 成功加载断言，新增 fusion KD virtual alias 失败断言。
- [x] 4.2 更新 no-KD 配置测试，断言 no-KD 主线无需 KD temperature、alpha 或 RKD 权重字段。
- [x] 4.3 更新 `tests/test_config_load_characterization.py`，不再把 fusion KD virtual config 作为加载性能样本。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_config_load_characterization.py -q`。
- [x] 4.6 运行保留 subset/evaluation focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py tests/test_subset_specs.py -q`。

## 5. 文档更新

- [x] 5.1 更新 README，移除旧模态诊断脚本推荐，强调当前主线为少样本跨场景 adaptation。
- [x] 5.2 更新 `docs/experiment_matrix.md`，将 legacy KD、snapshot、occlusion、position 和 multitask 标记为 optional/supporting，而非默认主线步骤。
- [x] 5.3 更新 `docs/extension_guide.md`，说明新增 fusion 配置时默认只扩展 no-KD 主线，legacy KD baseline 需独立提案或显式实体配置。
- [x] 5.4 更新 `docs/research_notes.md`，记录本轮进一步收窄模态失衡表面积和 KD virtual alias。

## 6. 最终验证

- [x] 6.1 运行 `openspec validate prune-obsolete-modality-imbalance-surface --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-hist-beam-loso --help`。
- [x] 6.3 运行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 6.4 按风险决定是否运行最终回归 `conda run -n kd_mm_beam pytest -q`；若未运行，记录原因和已完成的 focused 验证。
