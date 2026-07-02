# hist-beam-cross-scene-adaptation Specification

## Purpose
定义 HiST-Beam 跨场景自适应方法的模型变体、层次化 beam label、adapter/prototype/residual 适配、训练诊断和评估输出契约，确保快速验证中的 source-only、adapter、adapter+prototype 与 full fine-tuning baseline 可配置、可复现并能被 LOSO workflow 汇总比较。
## Requirements
### Requirement: HiST-Beam 研究线已退役
HiST-Beam/Hist 跨场景适配不再属于当前支持能力。系统 MUST 不再提供 HiST-Beam 模型构建、LOSO 执行、target adaptation、prototype、radio/path variant、history-anchor Hist、image-only Hist probe、V7/V8/V9 Hist variant 或 Hist prediction artifact 生成入口。

#### Scenario: Hist 配置不可运行
- **WHEN** 用户尝试运行 HiST-Beam/Hist 配置或 CLI
- **THEN** 系统 MUST 失败、报告入口已退役或缺失对应配置
- **AND** 系统 MUST 不静默迁移到其它当前 workflow

#### Scenario: 当前源码不包含 Hist 模型实现
- **WHEN** 开发者检查当前 `src/kd_sensing/engine` 和 `src/kd_sensing/models`
- **THEN** 当前支持面 MUST 不包含 `hist_beam_*` engine 实现或 `models/fusion/hist_beam.py`
- **AND** registry MUST 不提供 `hist_beam_fusion`

### Requirement: Hist workflow 已从当前实验入口退役
当前训练、评估、quickstart、CLI help、run metadata 和推荐文档 MUST 不再包含 HiST-Beam/Hist LOSO 入口。旧 Hist 配置路径、console script 和 run plan 不得作为当前 workflow 兼容承诺。

#### Scenario: CLI help 不包含 Hist 保留入口
- **WHEN** 开发者执行当前推荐的 CLI help 验证
- **THEN** `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-jepa-visual-analysis` 和 `kd-sensing-jepa-gps-shortcut-benchmark` MUST 正常退出
- **AND** 验证 MUST 不要求 `kd-sensing-hist-beam-loso` 存在

#### Scenario: 旧 Hist 配置路径失败
- **WHEN** 用户传入 `configs/hist_beam/quick_smoke.yaml` 或其它 `configs/hist_beam/` 路径
- **THEN** 配置加载 MUST 失败或报告路径已退役
- **AND** 系统 MUST 不生成等价 virtual config

### Requirement: Hist 组件注册已退役
组件注册表 MUST 不再注册 HiST-Beam/Hist 专用模型、loss、adapter、prototype 或 workflow 组件。旧注册名 MUST 被识别为已删除或未知名称，并给出当前支持范围。

#### Scenario: hist_beam_fusion 构建失败
- **WHEN** 用户请求构建 `hist_beam_fusion`
- **THEN** registry 或配置构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 说明 Hist/HiST-Beam 研究线已退役

#### Scenario: Hist variants 不作为模型注册名
- **WHEN** 默认组件导入完成后开发者查看 `MODELS` 注册名
- **THEN** 注册名 MUST 不包含 HiST-Beam variants、P3/radio prototype variants、image-only Hist probe variants 或 history-anchor Hist variants
- **AND** 当前主线模型注册名 MUST 继续可用

### Requirement: 已删除组件错误包含 Hist 迁移方向
当用户引用 Hist 旧组件名且该名称仍由当前迁移 guard 覆盖时，错误信息 MUST 区分退役研究线与普通拼写错误，并指向当前推荐 workflow 或说明无兼容迁移。若本 change 删除对应 guard table，Hist 旧组件名 MAY 回落为普通未知名称，但仍 MUST 不注册为 current 可构建组件。

#### Scenario: Hist 旧模型名错误可诊断
- **WHEN** 用户配置 `model.primary.type: hist_beam_fusion`
- **THEN** 系统 MUST 拒绝构建并包含请求名称
- **AND** 若 Hist guard 被保留，错误信息 MUST 提示使用当前 supervised、adapter、GPS candidate、residual fusion 或其它保留 workflow；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误
