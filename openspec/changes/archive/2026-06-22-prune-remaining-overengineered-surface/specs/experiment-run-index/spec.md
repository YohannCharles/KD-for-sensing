## ADDED Requirements

### Requirement: Run index legacy 扫描必须右尺寸化
运行索引 MUST 保持只读、状态分类、过滤、输出和资源快照能力，但不应继续扩张只服务历史研究线考古的默认扫描分支。默认扫描 MUST 以当前 canonical output layout 为中心；历史 archive 或非 run 分区只在显式扫描或 cleanup/organize 需要时处理。

#### Scenario: 默认扫描关注 current layout
- **WHEN** 用户对 `outputs/` 构建 run index
- **THEN** 系统 MUST 扫描 canonical scene、scenegroup、evaluation 和 analysis run 位置
- **AND** 系统 MUST 默认跳过 cache、archive、cleanup manifest 和其它非 run 分区，除非用户显式指定扫描根

#### Scenario: 删除 legacy-only 分支不影响状态分类
- **WHEN** 实现删除某个只服务历史目录命名的 discovery 分支
- **THEN** complete、partial、running、waiting、killed、stale 和 unknown 状态分类 MUST 对 current run 继续可用
- **AND** run index tests MUST 覆盖当前状态分类

### Requirement: Run index 不承担 runtime cleanup 规则库
Run index MUST 提供清理流程需要的结构化摘要，但 MUST 不成为历史删除规则库。清理候选规则 MUST 留在 runtime cleanup owner，run index 只返回 run 状态、artifact、大小和时间摘要。

#### Scenario: cleanup 复用 run summary
- **WHEN** runtime cleanup 生成 manifest 并复用 run index
- **THEN** run index MUST 只返回只读 summary
- **AND** 删除候选规则、保护判断和 action plan MUST 由 cleanup owner 决定
