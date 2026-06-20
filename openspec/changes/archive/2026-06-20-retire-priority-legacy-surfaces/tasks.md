## 1. 前置状态与退役基线

- [x] 1.1 运行 `openspec status --change retire-legacy-model-registry-surface`，确认 legacy registry change 的剩余任务、已完成范围和未归档状态。
- [x] 1.2 运行引用扫描，确认 legacy registry 名称、AMR-Net、JEPA-MSAC、MMW 旁支脚本和非 CSI shell runner 的 current 引用范围，并把结果记录到实现总结。
- [x] 1.3 若 `retire-legacy-model-registry-surface` 尚未完成 registry guard、config migration 或 docs/tests 收口，先完成或明确阻塞；本 change 不重复修改同一批 registry 代码。
- [x] 1.4 确认本 change 不读取真实 `dataset/`，不生成训练输出、cache、logs 或 checkpoint。

## 2. MMW GPS v2 旁支脚本收口

- [x] 2.1 对比 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py`、`scripts/mmw/visualize_prediction_error_label_distribution.py` 与 `kd-sensing-plot-mmw-town-gps-v2` / `kd-sensing-compare-mmw-town-gps-v2` 的输出职责。
- [x] 2.2 将仍属 current spec 的必要图表或不可用说明迁入 `src/kd_sensing/cli/plot_mmw_town_gps_v2.py` 或窄 helper；不迁移历史 exploratory 图。
- [x] 2.3 删除或退役三个 `scripts/mmw/visualize_gps_*` 旁支脚本，并从 `docs/maintainer_context_index.yaml` 的 `python_allowlist` 移除。
- [x] 2.4 更新 README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md` 和 MMW v2 相关说明，使当前图表/对比命令只指向 package CLI。
- [x] 2.5 更新 `tests/test_architecture_boundaries.py` 和必要 focused tests，防止三个旁支脚本回流。
- [x] 2.6 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q`，确认 MMW GPS v2 runner/plotter/comparison 行为未回归。

## 3. AMR-Net_gps_image 退役

- [x] 3.1 删除 `pyproject.toml` 中的 `kd-sensing-run-amr-net-gps-image` console script，并更新 CLI help 测试预期。
- [x] 3.2 删除 `src/kd_sensing/cli/run_amr_net_gps_image.py`、`configs/baselines/amr_net_gps_image.yaml` 和 `src/kd_sensing/baselines/amr_net_gps_image/` current workflow 实现。
- [x] 3.3 删除或改写 `tests/test_amr_net_gps_image.py`：不再测试 mock/report runner，改为验证 AMR config、CLI 和 module path 不作为 current 入口存在。
- [x] 3.4 更新 `src/kd_sensing/config/normalization.py`、`src/kd_sensing/config/validation.py` 或其它 config hook，移除只服务 AMR-Net preset 的 current branch；必要时加入 retired config guard。
- [x] 3.5 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`、README 和 `docs/project_surface_inventory.md`，删除 AMR current 行，只保留 metadata conflict tombstone 或历史说明。
- [x] 3.6 更新 `docs/maintainer_context_index.yaml` entrypoint allowlist 和 retired route guard，确保 AMR 不再作为 package CLI。
- [x] 3.7 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，确认 AMR CLI/config 退役边界生效。

## 4. JEPA-MSAC 退役

- [x] 4.1 运行引用扫描，确认 `jepa_msac`、`jepa_msac_pretraining`、`val_jepa_msac_loss`、`JepaMsacModel` 和 `jepa_msac_stage2_losses` 是否只服务 JEPA-MSAC workflow。
- [x] 4.2 删除 `pyproject.toml` 中的 `kd-sensing-run-jepa-msac` console script，并更新 CLI help 测试预期。
- [x] 4.3 删除 `src/kd_sensing/cli/run_jepa_msac.py`、`configs/pretraining/jepa_msac_s32_smoke.yaml`、`configs/pretraining/jepa_msac_s32_paper.yaml` 和 `src/kd_sensing/baselines/jepa_msac/`。
- [x] 4.4 删除或 retired-guard `src/kd_sensing/models/jepa_msac.py` 的 whole-model registry surface；若保留 removed guard，错误信息必须指向当前 GPS-conditioned JEPA 或 JEPA benchmark workflow。
- [x] 4.5 删除 `src/kd_sensing/losses/jepa_msac.py` 及 `src/kd_sensing/losses/__init__.py` 中的导出；若有通用 loss 逻辑被复用，先迁到非 JEPA-MSAC 命名 helper。
- [x] 4.6 清理 `src/kd_sensing/engine/objectives/history.py`、`registry.py`、`metadata.py`、`src/kd_sensing/config/normalization.py` 和 `src/kd_sensing/config/validation.py` 中只服务 JEPA-MSAC 的 objective/config branches。
- [x] 4.7 删除或改写 `tests/test_jepa_msac.py`：不再测试 current smoke workflow，改为验证 JEPA-MSAC CLI/config/model/loss/module path 不作为 current 入口存在。
- [x] 4.8 更新 README、mainline catalog、experiment protocols、claim registry、experiment matrix、project inventory 和 maintainer index，删除 JEPA-MSAC current/pending/mock-smoke 行。
- [x] 4.9 运行 `conda run -n kd_mm_beam pytest tests/test_objective_metadata.py tests/test_prediction_objectives.py -q`，确认 objective 元数据和当前预测目标未回归。

## 5. 非 CSI shell orchestration 退役

- [x] 5.1 删除 `scripts/run_deepsense_gps_circular_soft_label.sh` 和 `scripts/run_mmw_gps_circular_soft_label_ablation.sh`，并从 docs/index 中移除 current shell_allowlist。
- [x] 5.2 删除 `scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 和 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh`，并从 docs/index 中移除 current shell_allowlist。
- [x] 5.3 保留 `scripts/run_csi_hardening_matrix.sh`，并确认 README/inventory 仍将其标为 current CSI hardening matrix runner。
- [x] 5.4 更新 README、`docs/experiment_matrix.md` 和 `docs/project_surface_inventory.md`，把被删 shell runner 标为 retired/historical，不再提供 current 推荐命令。
- [x] 5.5 更新 `tests/test_architecture_boundaries.py`，拒绝被删 shell runner 重新加入 allowlist 或源码 current scripts。

## 6. 文档、索引和 OpenSpec 收口

- [x] 6.1 更新 `docs/maintainer_context_index.yaml` 的 package CLI、python allowlist、shell allowlist、retired route tokens、health check 命令和 owner metadata。
- [x] 6.2 更新 `docs/project_surface_inventory.md` 的 capability lifecycle、脚本入口分类、配置生命周期和退役说明。
- [x] 6.3 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md`，确保 current 表不含 AMR-Net、JEPA-MSAC 或退役脚本命令。
- [x] 6.4 更新 README，保留短索引和关键 caveat，不复制完整退役账本；当前 MMW GPS v2 只指向 package CLI。
- [x] 6.5 检查本 change 的 spec delta 与 current specs 不冲突，必要时调整 REMOVED/ADDED requirement。

## 7. 验证与交付

- [x] 7.1 运行 `openspec validate retire-priority-legacy-surfaces --strict`。
- [x] 7.2 若 `retire-legacy-model-registry-surface` 仍未归档，运行 `openspec validate retire-legacy-model-registry-surface --strict` 并记录状态。
- [ ] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 7.5 运行本 change 触碰范围对应的 focused tests：MMW 触碰时运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q`；objective 触碰时运行 `conda run -n kd_mm_beam pytest tests/test_objective_metadata.py tests/test_prediction_objectives.py -q`。
- [x] 7.6 汇总退役清单、保留入口、迁移路径、验证结果、未运行项和剩余风险。
