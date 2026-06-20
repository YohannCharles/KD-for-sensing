## ADDED Requirements

### Requirement: 兼容 facade 收缩后 owner 路径成为当前入口
项目 MAY 删除不再属于 current public surface 的兼容 facade、legacy wrapper 和 re-export 模块。删除前，当前内部源码、README、docs、OpenSpec、tests 和示例 MUST 改用真实 owner 模块、canonical registry 名称、配置路径或 package CLI；删除后不得新增等价 wrapper 恢复旧入口。

#### Scenario: 内部代码迁出 facade
- **WHEN** 内部源码仍通过兼容 facade 导入当前实现
- **THEN** 本 change MUST 将导入改为真实 owner 模块或 registry/config 构建路径
- **AND** 架构边界测试 MUST 拒绝该 facade 重新成为内部依赖

#### Scenario: 外部兼容路径作为 breaking change 删除
- **WHEN** 某个历史 import 路径未被当前 docs、CLI、registry 或配置声明为支持入口
- **THEN** 本 change MAY 删除该路径
- **AND** 变更说明 MUST 将其标记为 breaking change 并给出当前 owner 路径或当前入口类别

## MODIFIED Requirements

### Requirement: models 包级轻量导入
`kd_sensing.models` MUST 保持轻量可导入，但不再 MUST 维持所有历史 package-level 模型符号兼容。该包 MAY 只暴露明确保留的当前公共符号、package metadata 或轻量 helper；当前内部代码、文档和测试 MUST 优先从真实 owner 模块、registry/config 名称或 package CLI 访问模型能力。删除的历史别名和便利导出 MAY 直接产生普通 `ImportError` 或 `AttributeError`，除非本 change 明确保留某个迁移 guard。

#### Scenario: 轻量导入 models 包
- **WHEN** 开发者执行 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入各模型实现模块、训练 runtime、dataset reader 或重依赖视觉/科学计算模块

#### Scenario: 当前模型符号使用 owner 路径
- **WHEN** 当前源码、README、docs 或 tests 需要引用模型实现类
- **THEN** 引用 MUST 使用真实 owner 模块、canonical registry 名称或配置构建路径
- **AND** 不得要求 `kd_sensing.models.__all__` 继续列出历史便利导出

#### Scenario: removed alias 不再强制兼容
- **WHEN** 现有外部代码访问已移除的模型别名或历史 package-level 导出
- **THEN** 系统 MAY 抛出普通导入或属性错误
- **AND** 只有仍被当前迁移文档明确覆盖的别名才需要清晰替代符号提示

### Requirement: 优先退役入口不得作为 current public surface
项目 MUST 将本 change 标记的优先退役入口从 current public surface 移除。被移除的入口 MUST 不再出现在 `pyproject.toml` console scripts、README quickstart、CLI help smoke、当前 structured inventory 或 `scripts/` current allowlist 中。历史说明 MAY 保留，但 MUST 标记为 retired、historical、blocked background 或 tombstone；项目不再 MUST 为这些入口维护 `docs/maintainer_context_index.yaml` 条目。

#### Scenario: 退役 package CLI 不再声明
- **WHEN** 开发者检查 `pyproject.toml` 和安装后的 console script help smoke
- **THEN** 项目 MUST 不声明 `kd-sensing-run-amr-net-gps-image`
- **AND** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`
- **AND** CLI help smoke MUST 不要求这两个命令存在

#### Scenario: 退役 script 不在 current allowlist
- **WHEN** 开发者检查脚本入口健康检查、当前 structured inventory 或保留的脚本 allowlist
- **THEN** current 入口 MUST 不包含 `scripts/mmw/visualize_gps_angle_beam_correspondence.py`
- **AND** current 入口 MUST 不包含 `scripts/mmw/visualize_gps_prediction_trajectory.py`
- **AND** current 入口 MUST 不包含 `scripts/mmw/visualize_prediction_error_label_distribution.py`
- **AND** current 入口 MUST 不包含 `scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 或 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh`
