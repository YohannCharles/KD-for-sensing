## ADDED Requirements

### Requirement: 安全边界使用小型静态 guard
项目 MUST 使用小型参数化测试检查 tracked secret、系统配置污染和危险 shell runner，不得为该检查维护通用 surface doctor、inventory renderer 或 JSON report schema。

#### Scenario: 系统配置污染被拒绝
- **WHEN** tracked 文本尝试把训练、清理、GPU queue 或启动命令写入凭证/系统配置语境
- **THEN** 安全 guard MUST 失败并指出文件与规则
- **AND** 检查 MUST 不读取真实系统凭证或修改文件

#### Scenario: 普通源码不触发安全 guard
- **WHEN** 训练命令只存在于正常 CLI、脚本、文档示例或测试 fixture
- **THEN** guard MUST 不把它误报为系统配置污染
- **AND** fixture MUST 覆盖危险与允许样例

## MODIFIED Requirements

### Requirement: Console script surface guardrail
项目健康护栏 MUST 直接比对 `pyproject.toml` console scripts、CLI help smoke、inventory 和 current docs/OpenSpec。新增、删除或降级 CLI 时，guardrail MUST 报告 lifecycle/smoke/stale-reference 漂移；该检查 MUST 不依赖 project surface doctor CLI。

#### Scenario: pyproject 与 help smoke 一致
- **WHEN** 开发者运行 CLI/architecture focused checks
- **THEN** 十个 public entry points MUST 与 help smoke 集合一致
- **AND** 缺少或多余命令 MUST 直接由测试报告

#### Scenario: docs 不引用已删除 CLI
- **WHEN** dashboard、preview 或 doctor command 被删除
- **THEN** static current-doc reference check MUST 发现残留 current 命令
- **AND** 检查 MUST 不调用已删除 doctor

#### Scenario: 新 public CLI 需要锚点
- **WHEN** 后续 change 新增 console script
- **THEN** test MUST 要求 owner、inventory/docs、help smoke 和 output boundary
- **AND** 缺少锚点时 MUST 失败

### Requirement: 大规模表面清理必须有快速验收
项目 MUST 为大规模表面清理提供直接组合 OpenSpec、architecture、CLI help、config characterization 和 compile 的快速验收。所有 Python 命令 MUST 使用 `kd_mm_beam`，并 MUST 不依赖 surface doctor 或其它替代产品面。

#### Scenario: 清理 wave 快速验收
- **WHEN** 一个删除 wave 完成
- **THEN** 对应 OpenSpec 与 focused pytest/compile checks MUST 运行
- **AND** 上一 wave 未通过时 MUST 停止

#### Scenario: CLI 或 script 变更验收
- **WHEN** pyproject、CLI 或 local/manual script surface 改变
- **THEN** CLI help、architecture 和 stale-reference focused tests MUST 运行
- **AND** 检查 MUST 不读取真实 dataset 或启动训练

#### Scenario: Public surface 最终验收
- **WHEN** consolidation 完成
- **THEN** `openspec validate --all --strict` 与 CLI/architecture focused tests MUST 通过
- **AND** 不要求 project surface doctor scope

### Requirement: Post-C2 guardrail 必须检查保护范围
项目健康护栏 MUST 检查 protected inventory，防止 MMW/CSI、主线 YAML/manifest、final C2/U-MaskBeamJEPA 主线和 U-Mask fusion 分支被误删或被文档降级为 retired。检查 MUST 从明确的 protected paths、current configs 和 owner imports 读取事实，不依赖 surface doctor 输出。

#### Scenario: protected path 缺失被发现
- **WHEN** 开发者运行架构边界测试
- **THEN** 检查 MUST 验证 protected MMW/CSI、主线 YAML/manifest、final C2/U-Mask owner 和 U-Mask branch markers 仍存在或有明确替代记录
- **AND** 缺失且无替代记录 MUST 报告 error

#### Scenario: protected docs 不被标成 retired
- **WHEN** README、docs 或 current specs 描述 MMW、final C2、U-MaskBeamJEPA 或 protected mainline config
- **THEN** 健康检查 MUST 不允许这些 protected surface 被描述为 retired、historical-only 或 delete-candidate
- **AND** 若文档只描述后续审计候选，MUST 明确其不属于本 change 删除范围

### Requirement: Stale reference 检查必须覆盖删除波次
Post-C2 清理后，健康护栏 MUST 检查 current README、docs、OpenSpec specs、tests、pyproject 和 scripts 默认路径中是否仍引用已删除入口、config 或 module。历史 archive 中的引用 MAY 保留，但 MUST 不被 current docs 当作推荐入口。

#### Scenario: 删除 CLI 后 docs stale reference
- **WHEN** public console script 被删除
- **THEN** pyproject/current-doc reference check MUST 报告 stale command
- **AND** CLI help smoke MUST 不再要求已删除命令存在

#### Scenario: 删除 config 后 current reference
- **WHEN** YAML 或 manifest 被删除或生成化
- **THEN** 配置引用检查 MUST 能发现 current docs、tests、scripts 或 specs 中指向不存在路径的引用
- **AND** 修复路径 MUST 是恢复配置、更新到 generator/manifest/base config，或把引用改为 historical

## REMOVED Requirements

### Requirement: 项目表面积 doctor
**Reason**: Doctor 与现有结构检查重复且未发现本轮真实 wrapper/语义漂移。
**Migration**: 使用 focused architecture、CLI/config、compile、OpenSpec 和安全 guard。

#### Scenario: Doctor owner 删除
- **WHEN** post-C2 consolidation 完成
- **THEN** doctor module、CLI 和专属 tests MUST 不再存在
- **AND** quick verification MUST 不引用该命令

### Requirement: Doctor 可纳入 quick verify
**Reason**: Project surface doctor 退役，不能继续作为 quick verify 依赖。
**Migration**: Quick verify 直接组合现有 focused checks。

#### Scenario: Quick verify 不调用 doctor
- **WHEN** 开发者运行 quick verification
- **THEN** 命令 MUST 不调用 surface doctor
- **AND** protected behavior 仍由 focused tests 覆盖

### Requirement: Surface doctor 默认输出必须 issue-only
**Reason**: Doctor 输出产品面随 doctor 一并删除。
**Migration**: 测试失败直接使用 pytest/OpenSpec/compile 的原生定位输出。

#### Scenario: 不再维护 doctor renderer
- **WHEN** 检查没有问题或失败
- **THEN** 项目 MUST 使用原生验证命令输出
- **AND** 不再生成 doctor inventory dump

### Requirement: Surface doctor 瘦身不得降低失败可诊断性
**Reason**: 本 change 删除 doctor，而不是继续瘦身其 renderer。
**Migration**: 每个 focused guard MUST 在 assertion 中报告路径和失败原因。

#### Scenario: Focused guard 可定位
- **WHEN** retained guard 失败
- **THEN** 输出 MUST 包含受影响路径和规则
- **AND** 理解失败不得依赖 doctor full dump
