## 1. Baseline 与配置族分类

- [ ] 1.1 枚举 `configs/scene31/`、`configs/fusion/experiments/rbma_missing_workflow/`、`configs/fusion/experiments/rbma_missing_workflow_strong_encoders/` 和 `configs/fusion/experiments/jepa_image_gps/` 的 tracked YAML。
- [ ] 1.2 对每个 YAML 或配置族标记 lifecycle：`canonical/current`、`paper_workflow_reproduction`、`claim_evidence_input`、`diagnostics_manifest`、`local_manual_overlay`、`generated_recipe_backed`、`historical` 或 `delete_candidate`。
- [ ] 1.3 检查 README、docs、OpenSpec current specs、tests、scripts、claim provenance 和 config loader 对候选 YAML 的引用。
- [ ] 1.4 记录每个保留 YAML 的 owner、真实数据/本地 checkpoint 依赖、默认输出边界、保留理由和删除触发条件。

## 2. Generator/Manifest 等价验证

- [ ] 2.1 为可由 generator、template、manifest 或 recipe 替代的配置族列出替代 owner、base config 和 manifest schema。
- [ ] 2.2 补齐或更新 generator focused tests，覆盖 run name、seed、epoch、sampler、loss weights、missing pattern、dataset split、model primary、output boundary 和关键 overrides。
- [ ] 2.3 对允许差异字段建立说明，例如 run identity、输出目录、timestamp 或显式非行为字段。
- [ ] 2.4 确认 generator/virtual config 不生成或接管 retired KD、BGAM、viewer、Hist、Raymobtime、AMR-Net_gps_image 或 JEPA-MSAC 路径。

## 3. 配置删除与降级

- [ ] 3.1 删除可无损重建且无 current evidence/docs/spec/test 依赖的实体 YAML，保留 generator、manifest、base config 和 focused tests。
- [ ] 3.2 将仍需人工复跑但不支撑 promoted claim 的 YAML 登记为 `local_manual_overlay` 或 `historical`，并写明 checkpoint/outputs 依赖。
- [ ] 3.3 保留 current claim、paper/workflow reproduction、diagnostics manifest 或 focused test fixture 所需 YAML；若改为 generated input，先更新 claim provenance。
- [ ] 3.4 更新 scripts 默认 config path，避免 current runner 或 diagnostic 指向已删除 YAML。

## 4. 文档与 Doctor 同步

- [ ] 4.1 更新 `docs/project_surface_inventory.md`，同步配置族 lifecycle、保留理由、删除项、generator/manifest 复跑路径和本地产物边界。
- [ ] 4.2 更新 `docs/mainline_model_catalog.md`、`docs/experiment_matrix.md` 和 `docs/result_claims_registry.md`，确保 current/pending/local/manual/historical 状态和配置路径一致。
- [ ] 4.3 更新 `tests/test_config_load_characterization.py`、generator focused tests、architecture boundary 或 config/surface doctor，使 stale config references 和 recipe migration candidates 能被发现。
- [ ] 4.4 确认删除配置后 current docs/OpenSpec/tests/scripts 不指向不存在路径；historical/archive 引用必须明确不是 current entry。

## 5. 验证

- [ ] 5.1 运行 `openspec validate shrink-experiment-config-families --strict`。
- [ ] 5.2 运行 `openspec validate --all --strict`。
- [ ] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。
- [ ] 5.4 运行受影响 generator 的 focused tests；最终说明必须列出实际 test 文件名和命令。
- [ ] 5.5 运行 `conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope configs --scope scripts --format markdown --fail-on warning`。
- [ ] 5.6 最终说明列出删除、保留、生成化、historical/local-manual 的配置清单，未运行验证原因，以及与 `right-size-public-entrypoint-surface` 并行实现时的 docs/inventory 合并注意事项。
