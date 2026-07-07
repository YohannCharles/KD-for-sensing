## ADDED Requirements

### Requirement: Post-C2 guardrail 必须检查保护范围
项目健康护栏 MUST 在 post-C2 表面积清理中检查 protected inventory，防止 MMW/CSI、主线 YAML/manifest、final C2/U-MaskBeamJEPA 主线和 U-Mask fusion 分支被误删或被文档降级为 retired。

#### Scenario: protected path 缺失被发现
- **WHEN** 开发者运行架构边界测试或 project surface doctor 的清理验收 scope
- **THEN** 检查 MUST 验证 protected inventory 中的 MMW/CSI、主线 YAML/manifest、final C2/U-Mask owner 和 U-Mask fusion branch owner 仍存在或有明确替代记录
- **AND** 缺失且无替代记录 MUST 报告 error

#### Scenario: protected docs 不被标成 retired
- **WHEN** README、docs 或 OpenSpec current specs 描述 MMW、final C2、U-MaskBeamJEPA 或 protected mainline config
- **THEN** 健康检查 MUST 不允许这些 protected surface 被描述为 retired、historical-only 或 delete-candidate
- **AND** 若文档只描述后续审计候选，MUST 明确其不属于本 change 删除范围

### Requirement: Stale reference 检查必须覆盖删除波次
Post-C2 清理后，健康护栏 MUST 检查 current README、docs、OpenSpec specs、tests、pyproject 和 scripts 默认路径中是否仍引用已删除入口、已删除 config 或已删除 module。历史 archive 中的引用 MAY 保留，但 MUST 不被 current docs 当作推荐入口。

#### Scenario: 删除 CLI 后 docs stale reference
- **WHEN** public console script 被删除
- **THEN** `kd-sensing-project-surface-doctor --scope cli-surface` 或等价检查 MUST 报告 current docs/specs 中的 stale command reference
- **AND** CLI help smoke MUST 不再要求已删除命令存在

#### Scenario: 删除 config 后 current reference
- **WHEN** YAML 或 manifest 被删除或生成化
- **THEN** 配置引用检查 MUST 能发现 current docs、tests、scripts 或 specs 中指向不存在路径的引用
- **AND** 修复路径 MUST 是恢复配置、更新到 generator/manifest/base config，或把引用改为 historical

### Requirement: 清理验收必须保持无副作用
Post-C2 清理验收 MUST 只读取 tracked source、configs、docs、OpenSpec、tests 和 git metadata，不得读取真实 `dataset/`、启动训练、加载 checkpoint 或写入 `outputs/`、`logs/`、cache。

#### Scenario: 快速验收命令
- **WHEN** implementation 完成一个删除 wave
- **THEN** 验收 MUST 至少考虑 `openspec validate prune-post-c2-nonmainline-surface --strict`、`openspec validate --all --strict`、`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 触碰 CLI/config/script 时 MUST 追加 `make verify-cli-config`、`make verify-compile` 或对应 focused tests

#### Scenario: MMW 周边触碰追加 focused tests
- **WHEN** implementation 虽然保留 MMW 但修改了 MMW docs、configs、CLI lifecycle 或 guardrail
- **THEN** tasks 或最终说明 MUST 追加 MMW/CSI focused validation
- **AND** 验证 MUST 不要求真实 MMW dataset 内容进入源码变更
