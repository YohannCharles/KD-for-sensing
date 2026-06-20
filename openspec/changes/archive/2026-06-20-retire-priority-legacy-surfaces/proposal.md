## Why

当前项目已经退役了 BGAM、viewer、Hist、Raymobtime 和多条 residual/KD 路线，但仍有四类“看起来还活着”的表面积会继续增加维护噪音：legacy model registry 收尾、MMW GPS v2 旁支诊断脚本、blocked/mock reproduction 入口，以及少数 shell orchestration 入口。现在已有主线模型目录、结果 claim registry、maintainer index 和 retired-route guard，可以把这些优先退役点收成一组可审计的波次，而不是继续让它们以 current 入口形态漂在 README、pyproject、scripts 和测试里。

## What Changes

- **前置收口**：完成或验证 `retire-legacy-model-registry-surface`，把 legacy whole-model/alias/feature-extractor registry 名称从 current 可发现面移出，只保留明确迁移提示的 removed guard。此项不在本 change 重复实现 registry 迁移代码，但本 change 会要求文档、入口和后续退役波次以该 change 的完成状态为前置条件。
- **BREAKING** 退役 MMW GPS v2 旁支研究脚本：
  - 删除或降级 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py`、`scripts/mmw/visualize_prediction_error_label_distribution.py`。
  - 保留 `kd-sensing-mmw-town-gps-v2`、`kd-sensing-plot-mmw-town-gps-v2` 和 `kd-sensing-compare-mmw-town-gps-v2` 作为唯一当前 MMW GPS v2 运行/绘图/对比入口。
  - 将脚本功能中仍有价值的图表口径并入 package CLI 或明确标记为 historical，不再作为 `scripts/` allowlist current research diagnostic。
- **BREAKING** 退役 blocked/mock reproduction 公开 CLI 入口：
  - 退役 `kd-sensing-run-amr-net-gps-image`、`configs/baselines/amr_net_gps_image.yaml` 和 `src/kd_sensing/baselines/amr_net_gps_image/` 的可运行 mock/report 入口；保留 IEEE 11282996/Scenario 23 metadata conflict 作为 retired tombstone 或文档说明。
  - 退役 `kd-sensing-run-jepa-msac`、`configs/pretraining/jepa_msac_s32_{smoke,paper}.yaml` 和 `src/kd_sensing/baselines/jepa_msac/` 的 current workflow 入口，除非实现阶段明确选择降级为 archived source-audit only；不得继续以 `mock/smoke` current mainline 行出现。
  - 更新 claim registry，使 AMR-Net 与 JEPA-MSAC 不再作为 current pending/local-ready claim 占位。
- **BREAKING** 退役低价值 shell orchestration 入口：
  - 删除或归档 `scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`。
  - 保留 CSI hardening matrix shell runner，除非实现期另有用户确认；本 change 不退役当前 CSI hardening 主矩阵。
  - 将对应 README、experiment matrix、maintainer index、architecture boundary tests 和 CLI help tests 收口到 package CLI、当前 diagnostics 或 retired tombstone。
- 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml` 和 README，使 current 表只列仍推荐维护的入口，不把 mock/blocked/historical 入口写成当前主线。
- 扩展架构边界测试和配置加载测试，拒绝被退役的 CLI、script、config、module path 和 claim 行回流。

## Capabilities

### New Capabilities

- 无。本 change 不新增模型、数据或诊断能力，只收缩当前支持面并强化退役边界。

### Modified Capabilities

- `project-architecture`: 当前 CLI、script allowlist、package entry point 和 retired-route guard 需要移除上述脚本/CLI，并禁止它们以兼容 wrapper 或 legacy alias 回流。
- `experiment-workflow`: 当前实验矩阵和可运行 workflow 需要把 AMR-Net/JEPA-MSAC mock reproduction、MMW GPS v2 旁支脚本和低价值 shell runner 从 current workflow 移到 retired/historical 边界。
- `mainline-experiment-documentation`: 主线目录、协议表和结果 claim registry 需要删除 AMR-Net/JEPA-MSAC current claim 占位，并把 MMW GPS v2 说明限制在 package CLI。
- `mmw-town-gps-adapter-v2`: MMW GPS v2 的当前诊断入口需要收敛为 package CLI，不再要求或推荐额外 `scripts/mmw/visualize_gps_*` 旁支脚本。
- `ieee-11282996-gps-image-reproduction`: AMR-Net_gps_image 从 current source-audit/mock workflow 退役为 tombstone 或 historical note，不再提供 current CLI/config/mock metrics。
- `jepa-msac-reproduction`: JEPA-MSAC Scenario 32 workflow 从 current local-ready/mock-smoke 入口退役为 tombstone 或 historical note，不再提供 current CLI/config/mock metrics。

## Impact

- 受影响入口：
  - `pyproject.toml` 的 `kd-sensing-run-amr-net-gps-image`、`kd-sensing-run-jepa-msac`。
  - `src/kd_sensing/cli/run_amr_net_gps_image.py`、`src/kd_sensing/cli/run_jepa_msac.py`。
  - `scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py`、`scripts/mmw/visualize_prediction_error_label_distribution.py`。
  - `scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`。
- 受影响源码与配置：
  - `src/kd_sensing/baselines/amr_net_gps_image/`、`src/kd_sensing/baselines/jepa_msac/`、`src/kd_sensing/models/jepa_msac.py`、`src/kd_sensing/losses/jepa_msac.py`、与 JEPA-MSAC objective/config validation 相关的窄 helper。
  - `configs/baselines/amr_net_gps_image.yaml`、`configs/pretraining/jepa_msac_s32_smoke.yaml`、`configs/pretraining/jepa_msac_s32_paper.yaml`。
  - `src/kd_sensing/engine/objectives/*` 中只服务 JEPA-MSAC 的 objective key 和 history metadata；若仍被其它 current workflow 消费，必须在实现期拆分确认后再删。
- 受影响测试与文档：
  - `tests/test_cli_help.py`、`tests/test_architecture_boundaries.py`、`tests/test_config_load_characterization.py`。
  - `tests/test_amr_net_gps_image.py`、`tests/test_jepa_msac.py` 预期删除或改为 retired guard tests。
  - README、`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md`、`docs/maintainer_context_index.yaml`。
- 兼容性：
  - 旧 CLI、config 和 script 路径会失败或不存在；用户需改用当前 package CLI、MMW GPS v2 package plot/compare、JEPA visual analysis、GPS shortcut benchmark、CSI hardening matrix或通用训练/评估入口。
  - 不读取真实 `dataset/`，不删除 `outputs/`、`logs/`、cache、checkpoint 或本地历史产物；本 change 只收源码、配置、文档和测试表面积。
