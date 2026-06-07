## ADDED Requirements

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

## REMOVED Requirements

### Requirement: HiST-Beam 模型变体配置
**Reason**: HiST-Beam/Hist 研究线已退役，不再提供模型变体配置。
**Migration**: 使用当前 supervised、adapter、GPS candidate、residual fusion、MMW GPS v2、CSI 或 Raymobtime workflow。

### Requirement: 层次化 beam label 与输出契约
**Reason**: HiST-Beam 模型输出契约已随研究线退役。
**Migration**: 当前主线模型按各自 specs 定义输出。

### Requirement: 现代 residual/prototype 表示契约
**Reason**: Hist residual/prototype 表示不再作为支持能力。
**Migration**: 使用当前非 Hist residual 或 candidate workflow。

### Requirement: HiST-Beam 训练 loss
**Reason**: HiST-Beam training extension 和 loss bundle 已退役。
**Migration**: 当前训练流程使用保留 workflow 的 supervised/adaptation loss。

### Requirement: Source prototype artifact
**Reason**: Hist source prototype artifact 已退役。
**Migration**: 仅保留当前主线明确声明的 prototype 或 candidate artifact。

### Requirement: Target adapter adaptation
**Reason**: Hist target adapter adaptation 已退役。
**Migration**: 使用当前主线 adapter 或 candidate workflow。

### Requirement: 无标签与半监督 target adaptation
**Reason**: Hist 半监督 target adaptation 已退役。
**Migration**: 未来需要时必须以非 Hist workflow 重新提案。

### Requirement: HiST-Beam 指标与预测产物
**Reason**: Hist prediction artifact 已退役。
**Migration**: 当前主线评估产物按各自 specs 写出。

### Requirement: HiST-Beam execute run 产物
**Reason**: Hist execute workflow 已退役。
**Migration**: 使用当前保留 CLI 的运行产物。

### Requirement: Adaptation 效率指标
**Reason**: Hist adaptation 效率指标不再作为支持产物。
**Migration**: 当前主线若需要效率指标，由对应 workflow spec 定义。

### Requirement: Quick validation 对比结论
**Reason**: Hist quick validation 结论不再作为当前主线。
**Migration**: 当前主线结论由保留 workflow summary 产生。

### Requirement: Geometry-aware transferable knowledge
**Reason**: Hist geometry-aware branch 已退役。
**Migration**: 当前 geometry/candidate 能力由非 Hist specs 定义。

### Requirement: Scene-private knowledge as explicit refinement
**Reason**: Hist shared/private 场景私有 refinement 已退役。
**Migration**: 使用当前保留 adaptation workflow。

### Requirement: Angular smoothing loss
**Reason**: Hist angular smoothing loss 已退役。
**Migration**: 当前 soft/circular label 能力由非 Hist workflow 明确定义。

### Requirement: Multimodal geometry consistency loss
**Reason**: Hist multimodal geometry consistency loss 已退役。
**Migration**: 当前主线不继承该 Hist loss。

### Requirement: Private prototype alignment must be effective
**Reason**: Hist private prototype alignment 已退役。
**Migration**: 无兼容迁移。

### Requirement: Geometry-aware HiST-Beam 指标
**Reason**: Geometry-aware HiST-Beam 指标已退役。
**Migration**: 当前 geometry diagnostics 由保留 workflow 定义。

### Requirement: HiST-Beam radio-semantic prototype variant
**Reason**: Radio-semantic HiST-Beam variant 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam radio branch diagnostics
**Reason**: HiST radio branch diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam radio-conditioned beam head
**Reason**: HiST radio-conditioned beam head 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam source prototype 按需生成
**Reason**: HiST source prototype 生成已退役。
**Migration**: 无兼容迁移。

### Requirement: Source prototype 进度与耗时诊断
**Reason**: HiST prototype diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: MMW HiST-Beam LOSO stage 内存边界
**Reason**: MMW HiST-Beam LOSO executor 已退役。
**Migration**: 当前 MMW workflow 不通过 Hist LOSO executor 运行。

### Requirement: MMW sensor-assisted HiST-Beam profile
**Reason**: Sensor-assisted HiST-Beam profile 已退役。
**Migration**: 当前 MMW 主线由非 Hist workflow 定义。

### Requirement: Quick validation conclusion 排除不可用于主结论的 run
**Reason**: Hist quick validation conclusion 已退役。
**Migration**: 当前主线 summary 自行定义 eligibility。

### Requirement: History-anchored HiST-Beam 变体
**Reason**: History-anchored HiST-Beam variant 已退役。
**Migration**: Generic history objective 或 GPS window baseline 不因本 requirement 保留 Hist 模型。

### Requirement: Residual beam loss
**Reason**: Hist residual beam loss 已退役。
**Migration**: 当前非 Hist residual losses 由对应 workflow 定义。

### Requirement: Residual shared-private 解耦
**Reason**: Hist residual shared/private 解耦已退役。
**Migration**: 无兼容迁移。

### Requirement: History-anchored few-shot private calibration
**Reason**: Hist history-anchor private calibration 已退役。
**Migration**: 无兼容迁移。

### Requirement: History-anchored HiST-Beam 预测产物
**Reason**: Hist history-anchor prediction artifact 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam 主线默认 supervised/adaptation
**Reason**: HiST-Beam 主线整体退役，不再需要定义默认训练模式。
**Migration**: 当前主线 supervised/adaptation 由非 Hist specs 定义。

### Requirement: KD 不作为可运行 HiST-Beam baseline
**Reason**: HiST-Beam baseline 整体退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam shared/private 语义不依赖 KD
**Reason**: HiST shared/private 语义整体退役。
**Migration**: 无兼容迁移。

### Requirement: V7 shared physical private residual forward contract
**Reason**: V7 Hist variant 已退役。
**Migration**: 无兼容迁移。

### Requirement: V7 source training losses
**Reason**: V7 Hist losses 已退役。
**Migration**: 无兼容迁移。

### Requirement: V7 target private residual adaptation
**Reason**: V7 Hist adaptation 已退役。
**Migration**: 无兼容迁移。

### Requirement: V7 evaluation metrics and artifacts
**Reason**: V7 Hist artifact 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 target-prior HiST-Beam 变体
**Reason**: V8 Hist variant 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 target prior 初始化
**Reason**: V8 Hist target prior 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 target adaptation freeze policy
**Reason**: V8 Hist adaptation policy 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 adaptation loss
**Reason**: V8 Hist loss 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 诊断实验模式
**Reason**: V8 Hist diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: V8 prototype classifier 诊断
**Reason**: V8 Hist prototype diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam prediction histogram artifact
**Reason**: Hist prediction histogram artifact 已退役。
**Migration**: 当前保留 workflow 可自行输出 histogram，不继承 Hist 契约。

### Requirement: Source long-tail de-bias 配置入口
**Reason**: Hist source de-bias 配置入口已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 input-conditioned target adaptation
**Reason**: V9 Hist variant 已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 global prior strength control
**Reason**: V9 Hist prior control 已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 target support prototype logits
**Reason**: V9 Hist prototype logits 已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 anti-collapse regularization
**Reason**: V9 Hist regularization 已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 collapse diagnostics artifact
**Reason**: V9 Hist collapse diagnostics 已退役。
**Migration**: 无兼容迁移。

### Requirement: V9 quick validation experiment modes
**Reason**: V9 Hist quick validation 已退役。
**Migration**: 无兼容迁移。

### Requirement: HiST-Beam image-only variant 输出契约
**Reason**: Image-only HiST variant 已退役。
**Migration**: 当前 image workflows 不继承 Hist probe 契约。

### Requirement: Image source-only baseline
**Reason**: Image-only Hist baseline 已退役。
**Migration**: 使用当前 image supervised/adaptation workflow。

### Requirement: Image-only A2 target linear probe
**Reason**: Image-only Hist target probe 已退役。
**Migration**: 无兼容迁移。

### Requirement: Image-only V8 target prior head
**Reason**: Image-only V8 Hist probe 已退役。
**Migration**: 无兼容迁移。

### Requirement: Image-only V9 sector prototype
**Reason**: Image-only V9 Hist probe 已退役。
**Migration**: 无兼容迁移。

### Requirement: Image-only adaptation 设备与 dtype 稳定
**Reason**: Image-only Hist adaptation 已退役。
**Migration**: 当前保留 workflow 仍需自行保证 device/dtype 稳定。

### Requirement: HiST-Beam 可显式消费 GPS coarse anchor
**Reason**: HiST GPS anchor 条件输入已退役。
**Migration**: GPS anchor 或 candidate 能力由非 Hist workflow 定义。
