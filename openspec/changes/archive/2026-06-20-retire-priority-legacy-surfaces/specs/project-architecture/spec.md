## ADDED Requirements

### Requirement: 优先退役入口不得作为 current public surface
项目 MUST 将本 change 标记的优先退役入口从 current public surface 移除。被移除的入口 MUST 不再出现在 `pyproject.toml` console scripts、`docs/maintainer_context_index.yaml` entrypoint allowlist、README quickstart、CLI help smoke 或 `scripts/` allowlist 中。历史说明 MAY 保留，但 MUST 标记为 retired、historical、blocked background 或 tombstone。

#### Scenario: 退役 package CLI 不再声明
- **WHEN** 开发者检查 `pyproject.toml` 和安装后的 console script help smoke
- **THEN** 项目 MUST 不声明 `kd-sensing-run-amr-net-gps-image`
- **AND** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`
- **AND** CLI help smoke MUST 不要求这两个命令存在

#### Scenario: 退役 script 不在 allowlist
- **WHEN** 开发者检查 `docs/maintainer_context_index.yaml` 的 `python_allowlist` 和 `shell_allowlist`
- **THEN** allowlist MUST 不包含 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`
- **AND** allowlist MUST 不包含 `scripts/mmw/visualize_gps_prediction_trajectory.py`
- **AND** allowlist MUST 不包含 `scripts/mmw/visualize_prediction_error_label_distribution.py`
- **AND** allowlist MUST 不包含 `scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 或 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh`

### Requirement: 退役入口回流必须被架构边界测试拒绝
项目 MUST 通过架构边界测试防止优先退役入口以同名文件、等价 wrapper、thin alias、compat facade、virtual config 或 console script 形式回流。保留的历史说明 MUST 不要求对应模块可导入或命令可运行。

#### Scenario: 旧模块路径不可导入
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 `kd_sensing.cli.run_amr_net_gps_image` 和 `kd_sensing.cli.run_jepa_msac` 不作为 current CLI 模块存在
- **AND** 测试 MUST 验证 `kd_sensing.baselines.amr_net_gps_image` 和 `kd_sensing.baselines.jepa_msac` 不作为 current workflow package 存在

#### Scenario: 旧脚本路径不回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证被退役的 MMW 旁支诊断脚本和非 CSI shell orchestration 脚本未重新出现在源码树 current allowlist 中
- **AND** 测试 MUST 指向当前 package CLI、MMW GPS v2 plotter/comparison、JEPA visual analysis、GPS shortcut benchmark 或 CSI hardening runner 作为迁移方向

