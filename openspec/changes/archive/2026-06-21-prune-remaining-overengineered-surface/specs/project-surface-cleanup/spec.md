## ADDED Requirements

### Requirement: 剩余低价值项目表面必须分类收敛
项目 MUST 对 ponytail 审计确认的剩余低价值表面建立候选分类，并按删除、合并、保留、归档或后续 change 处理。候选范围 MUST 至少覆盖重复实体 YAML、薄 re-export facade、退役 route guard、无调用 helper、一次性分析脚本、重复小工具和大型治理测试。分类 MUST 记录公开 surface 风险、当前调用方、替代 owner、验证命令和回滚方式。

#### Scenario: 候选删除项有证据
- **WHEN** 开发者准备删除源码、配置、脚本或测试
- **THEN** 候选项 MUST 被证明不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 删除计划 MUST 指向替代 owner、recipe、文档位置或说明无需替代

#### Scenario: 候选保留项有理由
- **WHEN** 某个候选项因 public API、人工样例、diagnostics manifest 或外部迁移风险被保留
- **THEN** inventory 或实现说明 MUST 记录保留理由和未来删除触发条件
- **AND** 项目 MUST 不为了保留该候选项新增兼容 wrapper 或第二套治理表

### Requirement: 一次性研究脚本不得长期占据 current surface
只服务已完成调试结论、历史 sweep 汇总或人工复盘的一次性脚本 MUST 从当前支持面删除、归档为历史文档，或明确标为 local/manual research artifact。保留脚本时 MUST 记录 owner、输入输出边界和仍需运行的 focused 验证；删除脚本时 MUST 保留必要结论和 caveat 到 docs 或报告。

#### Scenario: 删除 CSI hardening sweep analyzer
- **WHEN** `scripts/analyze_csi_hardening_sweep.py` 或等价一次性脚本只服务历史 CSI hardening 调试结论
- **THEN** 本 change MAY 删除该脚本和只服务它的测试
- **AND** 仍有价值的结论 MUST 留在 `docs/research_notes.md`、CSI hardening 文档或对应报告中

#### Scenario: 保留当前诊断脚本
- **WHEN** 某个 `scripts/` 文件仍被 README、docs、OpenSpec current spec 或 package workflow 明确引用
- **THEN** 本 change MUST 不删除该脚本
- **AND** 若脚本保留，inventory MUST 将其分类为 research diagnostic、dataset preparation、figure helper 或 manual/local script

### Requirement: 源码瘦身不得触碰本地产物
项目表面瘦身 MUST 只修改源码、配置、测试、文档和 OpenSpec artifact。实现 MUST 不删除、不移动、不压缩、不重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`All_models/` 历史权重或其它本地运行产物。

#### Scenario: 实现 wave 检查工作树
- **WHEN** 每个源码瘦身 wave 完成
- **THEN** 开发者 MUST 检查 `git status --short`
- **AND** 新增或修改内容 MUST 不包含本地数据、输出、日志、cache、checkpoint 或临时训练产物

#### Scenario: 用户另行要求清理输出
- **WHEN** 用户要求删除 `outputs/`、`logs/`、cache、checkpoint 或 dataset 内容
- **THEN** 该操作 MUST 走 runtime cleanup manifest 或单独显式确认
- **AND** 本 change 的源码瘦身任务 MUST 不把该清理混入同一删除 wave
