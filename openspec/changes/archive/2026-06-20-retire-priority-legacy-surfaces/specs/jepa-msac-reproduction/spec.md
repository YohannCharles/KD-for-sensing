## REMOVED Requirements

### Requirement: JEPA-MSAC workflow 边界
**Reason**: JEPA-MSAC 不再作为 current paper/workflow reproduction 或 mock-smoke 入口维护。
**Migration**: 使用当前 GPS-conditioned JEPA、JEPA visual analysis、GPS shortcut benchmark 或其它 current JEPA downstream workflow。

#### Scenario: JEPA-MSAC CLI 退役
- **WHEN** 开发者检查 console scripts
- **THEN** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`

### Requirement: 论文对齐数据协议
**Reason**: JEPA-MSAC Scenario 32 数据协议不再作为 current workflow 构建目标。
**Migration**: 历史协议说明可保留在 archive/tombstone；当前数据协议以仍维护的 DeepSense6G、BeamBench、JEPA 和 CSI specs 为准。

#### Scenario: JEPA-MSAC 配置不可加载为 current
- **WHEN** 用户加载 `configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml`
- **THEN** 配置 MUST 失败或文件 MUST 不存在

### Requirement: RF 历史字段映射
**Reason**: RF history mapping 只服务已退役 JEPA-MSAC workflow，且不得成为新的 canonical modality。
**Migration**: 当前 RF/mmWave/beam-power 语义由现有 mmWave、beam-power 或 current workflow-local specs 负责。

#### Scenario: RF modality 不因 JEPA-MSAC 保留
- **WHEN** 用户在通用 modalities 中声明 `rf`
- **THEN** 系统 MUST 继续拒绝或不支持该 canonical modality

### Requirement: 多模态 tokenization 与位置编码
**Reason**: 该 tokenizer/positional encoding schema 只服务已退役 JEPA-MSAC model。
**Migration**: 当前 token/query/predictive fusion 使用 `token_transformer`、`next_beam_query_transformer`、GPS-query/predictive JEPA 等保留能力。

#### Scenario: JEPA-MSAC model 不作为 current registry exception
- **WHEN** 开发者检查 current model registry
- **THEN** `jepa_msac` MUST 不作为 current whole-model exception 出现

### Requirement: Temporal block-masked JEPA 预训练
**Reason**: JEPA-MSAC 专属 pretraining objective 不再作为 current objective。
**Migration**: 使用当前 `gps_conditioned_jepa` 或其它保留的 JEPA objective。

#### Scenario: JEPA-MSAC objective 退役
- **WHEN** 开发者检查 objective registry、history metadata 和 config validation
- **THEN** `jepa_msac_pretraining` MUST 不作为 current objective 入口保留

### Requirement: Frozen backbone future latent inference
**Reason**: 该 inference schema 只服务已退役 JEPA-MSAC Stage 2。
**Migration**: 当前 predictive latent/temporal workflows 使用保留的 predictive JEPA 或 downstream modules。

#### Scenario: JEPA-MSAC checkpoint loader 不作为 current API
- **WHEN** 开发者检查 current public model API
- **THEN** 项目 MUST 不要求 `build_frozen_jepa_msac_from_checkpoint` 可用

### Requirement: JEPA-MSAC task heads
**Reason**: localization/beam/RSSI task heads 是 JEPA-MSAC 专属 reproduction surface。
**Migration**: 当前任务 heads 使用 `modular_sequence`、保留 auxiliary heads 或 current prediction objectives。

#### Scenario: JEPA-MSAC task heads 不作为 current output
- **WHEN** 用户运行当前训练或评估
- **THEN** 系统 MUST 不要求 JEPA-MSAC 专属 localization/beam/RSSI heads

### Requirement: 论文指标与报告
**Reason**: JEPA-MSAC paper metrics/report 不再作为 current report schema。
**Migration**: 当前 report 使用仍维护 workflow 的 metrics schema 和 result claim registry。

#### Scenario: JEPA-MSAC report 不作为 current artifact
- **WHEN** 用户运行当前支持的 JEPA 或 benchmark workflow
- **THEN** 系统 MUST 不要求写出 JEPA-MSAC 专属 report

### Requirement: Ablation 与 baseline report
**Reason**: JEPA-MSAC ablation manifest 不再作为 current workflow 产物。
**Migration**: 当前 ablation/benchmark manifest 使用对应 current diagnostics 或 model family specs。

#### Scenario: JEPA-MSAC ablation manifest 退役
- **WHEN** 开发者检查 docs 和 tests
- **THEN** 项目 MUST 不要求 JEPA-MSAC ablation manifest 作为 current output

### Requirement: 运行产物与文档账本
**Reason**: JEPA-MSAC 不再保留 current local-ready/mock-smoke claim 行。
**Migration**: 历史本地产物只能作为 archive 背景或 cleanup manifest 候选。

#### Scenario: JEPA-MSAC claim 行移出 current registry
- **WHEN** 开发者阅读 result claim registry
- **THEN** JEPA-MSAC MUST 不作为 current pending、local-ready、mock-smoke 或 blocked claim 行出现

### Requirement: 验证与测试覆盖
**Reason**: JEPA-MSAC current implementation tests 将退役；保留的是 retired guard tests。
**Migration**: 运行 architecture boundary、CLI/config guard 和仍维护 JEPA workflows 的 focused tests。

#### Scenario: JEPA-MSAC focused smoke 不再要求运行
- **WHEN** 开发者运行 current diagnostics 或 health checks
- **THEN** 验证命令 MUST 不要求 `tests/test_jepa_msac.py` 作为 current workflow smoke 通过

## ADDED Requirements

### Requirement: JEPA-MSAC retired tombstone
项目 MUST 将 JEPA-MSAC Scenario 32 workflow 视为已退役 paper/workflow reproduction。源码、配置、CLI、model registry、loss、objective、tests 和文档账本 MUST 不再把 `jepa_msac` 暴露为 current 可运行入口；历史说明 MAY 保留在 tombstone 或 archived change 中。

#### Scenario: JEPA-MSAC 入口和配置不存在
- **WHEN** 开发者检查 pyproject、包内 CLI、configs、models、losses 和 baseline package
- **THEN** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`
- **AND** 项目 MUST 不保留 `configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml` 作为 current config
- **AND** 项目 MUST 不保留 `kd_sensing.baselines.jepa_msac`、`kd_sensing.models.jepa_msac` 或 `kd_sensing.losses.jepa_msac` 作为 current workflow implementation

