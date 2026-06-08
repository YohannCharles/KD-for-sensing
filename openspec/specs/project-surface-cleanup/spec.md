# project-surface-cleanup Specification

## Purpose
定义项目源码表面、退役研究线和本地运行产物清理的长期边界，确保已退役 Hist/KD 入口不会以兼容 wrapper 或 virtual alias 回流，新的输出目录具备清晰语义，删除本地产物必须经过可审计 manifest。
## Requirements
### Requirement: 退役研究线源码表面清理
项目 MUST 支持按 OpenSpec change 退役整条研究线。退役后，该研究线的 CLI、配置、模型、engine、evaluation、测试和推荐文档入口 MUST 从当前支持面删除，且不得新增旧入口兼容 wrapper、virtual alias 或二级聚合层。

#### Scenario: Hist 研究线退役完成
- **WHEN** 开发者检查当前源码、配置、README、pyproject、tests 和 OpenSpec 当前 specs
- **THEN** 系统 MUST 不再声明 HiST-Beam/Hist CLI、`configs/hist_beam/`、`hist_beam_fusion` 或 Hist variants 为受支持入口
- **AND** 历史 archive MAY 保留旧记录，但 MUST 不作为当前支持契约

#### Scenario: 旧入口不被兼容接管
- **WHEN** 用户引用已退役的 Hist CLI、配置路径或模型注册名
- **THEN** 系统 MUST 失败或给出清晰退役错误
- **AND** 系统 MUST 不通过旧路径自动映射到其它当前 workflow

### Requirement: 输出目录用途分区
项目 MUST 为新的本地运行产物提供用途清晰的输出目录约定。训练 run、analysis artifact、cache、features、cleanup manifest 和 scene-level best checkpoint SHOULD 采用独立分区；当前支持 workflow MUST 不默认向语义不清的 `outputs/other/` 写入新产物。

#### Scenario: 新训练输出写入训练分区
- **WHEN** 当前支持的训练 workflow 未显式覆盖输出目录
- **THEN** 默认输出 MUST 写入可识别 workflow、dataset 或 scene 的训练目录
- **AND** 运行目录 MUST 继续保存 `final_config.yaml`、`resolved_config.yaml`、metrics、checkpoint 和 runtime metadata

#### Scenario: 清理 manifest 写入固定分区
- **WHEN** 用户运行 runtime cleanup dry-run
- **THEN** manifest MUST 写入 `outputs/cleanup_manifests/` 或用户显式指定的 manifest 路径
- **AND** manifest 路径 MUST 不与训练 run、analysis、cache 或 features 混放

### Requirement: 过时输出删除可审计
项目 MAY 删除用户明确要求退役的本地实验产物，但 MUST 先生成 machine-readable manifest，并且删除阶段 MUST 只处理未受保护、未被 git 跟踪、仍位于允许根内且匹配退役规则的候选。

#### Scenario: 删除退役 Hist 输出
- **WHEN** manifest 将 Hist/P3/V8/V9/debug/smoke/plan-check/stale 输出列为未受保护候选
- **THEN** 删除阶段 MAY 删除这些候选
- **AND** deletion report MUST 记录每个已删除、跳过或失败路径的原因

#### Scenario: 保护当前主线输出
- **WHEN** manifest 扫描到当前主线 analysis、features、cache、best checkpoint 或带 sidecar metadata 的复现 artifact
- **THEN** manifest MUST 默认将其标记为 protected 或需要人工确认
- **AND** 删除阶段 MUST 拒绝删除 protected 路径
