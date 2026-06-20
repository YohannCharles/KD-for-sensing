## MODIFIED Requirements

### Requirement: supporting capability 不等于 standalone 当前入口
`supporting` capability MAY 保留支撑代码、数据契约、loss、metric、manifest schema、migration guard 或历史读取逻辑，但 MUST 明确不作为 standalone 当前推荐入口。支撑能力被当前 workflow 消费时，文档 MUST 指向实际 current workflow，而不是恢复旧入口。BGAM 与 viewer manifest 退役后，原先仅服务 BGAM/viewer 的 supporting helper MUST 不再以 supporting 名义保留。

#### Scenario: TopK 支撑代码不因 BGAM 保留
- **WHEN** BGAM 已退役且没有其它 current workflow 消费 TopK candidate manifest、loss 或 metric 支撑代码
- **THEN** lifecycle inventory MUST 将对应 BGAM-only 支撑能力标为 retired-tombstone 或移出当前支持面
- **AND** 文档 MUST 不把旧 Top8 selector、BGAM candidate manifest、训练、plot、compare CLI 或 root config 描述为当前入口

#### Scenario: 通用 helper 保留但历史 workflow 不复活
- **WHEN** 当前源码保留通用 LOSO、metric、cleanup、migration guard 或 artifact reader helper
- **THEN** lifecycle inventory MUST 区分 helper 的 supporting 地位和旧专用 workflow 的 retired 地位
- **AND** README 或 quickstart MUST 指向当前 workflow，而不是历史 workflow 名称

### Requirement: current spec 内部语义一致
`current` lifecycle 的 OpenSpec capability MUST 在同一 spec 内保持当前支持面语义一致。若同一 spec 同时包含 active mainline、retired/supporting、migration guard 或 historical wording，文档 MUST 明确区分其适用范围；不得让旧 active workflow 与当前拒绝边界并存而无解释。

#### Scenario: current spec 不保留旧 active workflow
- **WHEN** current spec 中保留了旧 KD、teacher/student、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual、BGAM、viewer manifest、Gradio viewer 或其它退役路线描述
- **THEN** 对应段落 MUST 标记为 retired、historical、supporting 或 migration guard
- **AND** 文档 MUST 不同时要求实现该旧路线作为当前 active workflow

#### Scenario: current spec 发生语义冲突
- **WHEN** 一个 current spec 内部既要求某路线作为当前默认入口，又在其它段落要求该路线退役或拒绝
- **THEN** 该状态 MUST 被视为规格漂移
- **AND** 维护者 MUST 通过 OpenSpec change 将其收敛为单一 current、supporting 或 retired 叙事

### Requirement: lifecycle 决策优先级
维护者和 AI agent MUST 结合 active change、current specs、lifecycle inventory、README/docs 和源码测试判断当前支持面。若 current spec 与 lifecycle inventory 或 README 当前入口冲突，MUST 优先将其视为待清理的规格漂移，而不是任选一段作为事实。

#### Scenario: current spec 与 inventory 冲突
- **WHEN** lifecycle inventory 将某能力标记为 retired-tombstone 或 supporting，但 current spec 中存在未加限定的 active mainline wording
- **THEN** agent MUST 将其报告为 lifecycle/wording 漂移
- **AND** agent MUST NOT 根据该 active wording 恢复旧 CLI、配置、registry 名称或训练入口

#### Scenario: historical report 与 current docs 冲突
- **WHEN** 历史报告、运行流水账或 archive 中的命令与 README、experiment matrix、mainline catalog 或 current spec 的当前口径冲突
- **THEN** historical report MUST 只作为演进背景
- **AND** 当前推荐入口 MUST 以 current docs、lifecycle inventory 和 current specs 的收敛结果为准

## ADDED Requirements

### Requirement: BGAM 和 viewer manifest 退役 lifecycle 边界
BGAM、GPS pseudo-history BGAM、BGAM-only TopK candidate 支撑、viewer manifest 导出、`kd-sensing-visualize-modalities` alias 和仓库级 Gradio viewer MUST 被标记为 `retired-tombstone` 或从当前 capability inventory 中移除。它们 MUST NOT 被记录为 current、supporting、quick validation、hotspot owner 或 recommended entrypoint。

#### Scenario: lifecycle inventory 标记退役
- **WHEN** 开发者或架构边界测试枚举 OpenSpec capability lifecycle
- **THEN** DeepSense6G/MMW BGAM、GPS pseudo-label BGAM、BGAM-only TopK candidate 和 viewer manifest/Gradio viewer 相关能力 MUST 不属于 current 支持面
- **AND** 若保留墓碑 spec，Purpose 或首个 requirement MUST 明确说明其已退役

#### Scenario: 验证命令不引用退役能力
- **WHEN** agent 读取 `docs/maintainer_context_index.yaml` 或 README 的验证命令
- **THEN** 推荐验证 MUST 不包含 BGAM focused tests、viewer manifest CLI help、`kd-sensing-visualize-modalities --help` 或 Gradio viewer smoke
- **AND** 退役防回流测试 MAY 断言这些入口不存在
