## ADDED Requirements

### Requirement: 缺失模态主线清理必须先分类再删除
项目在围绕缺失模态鲁棒性主线清理源码、脚本或配置前，MUST 将候选项分类为 current、secondary/supporting、local/manual、historical、retired、delete-candidate 或 merge-candidate。分类记录 MUST 包含 owner、当前调用方或引用、公开入口风险、替代入口、验证命令和回滚方式。未分类候选 MUST 不得被 README、docs、OpenSpec current specs 或 package CLI 描述为当前推荐入口。

#### Scenario: doctor 候选被分类
- **WHEN** `project_surface_doctor` 报告未分类 scripts、configs 或 hotspots
- **THEN** implementation MUST 删除该候选、合并到现有 owner，或在 inventory/等价生命周期文档中登记分类
- **AND** 最终状态 MUST 不把未分类候选留在 current supported surface

#### Scenario: 当前主线证据链被保护
- **WHEN** 候选项属于 U-MaskBeamJEPA 模型、U-Mask loss、run metadata、migration guard、retired-route guard test、当前 Scene31-34 主线 runner/summary/export 或明确 claim 证据输入
- **THEN** implementation MUST 不直接删除该候选
- **AND** 若仍需缩小，MUST 先记录替代 owner、保持公共语义的验证命令和单独删除触发条件

#### Scenario: 清理不触碰本地产物
- **WHEN** 清理缺失模态主线源码表面
- **THEN** implementation MUST 不删除、移动、重写或纳入 `dataset/`、`outputs/`、`logs/`、checkpoint、cache、TensorBoard event 或本地训练产物
- **AND** 删除验证 MUST 只基于 tracked source、configs、docs、OpenSpec 和 tests

### Requirement: Scene31 与 RBMA 配置表面必须可收缩
Scene31、Scene31-34、RBMA missing-modality、KD/BTAPA/weakKD/tau/seed/PatternFiLM 等配置族 MUST 具有可审计生命周期。当前主线、复现实验或 claim 证据仍需要的实体 YAML MUST 保留并登记；可由 generator、template 和 manifest 无损重建的实体 YAML MUST 从长期源码表面移除或登记删除计划；只服务已完成历史 sweep 的配置 MUST 删除或降级为 historical 说明。

#### Scenario: 可再生成配置不长期堆积
- **WHEN** generator、template 和 manifest 能重建某个 Scene31 或 Scene31-34 YAML
- **THEN** tracked source MUST 优先保留 generator、template 和 manifest
- **AND** 若实体 YAML 继续被跟踪，inventory MUST 说明它不能被无损重建的字段、人工样例价值或当前 claim 证据用途

#### Scenario: stale KD/BTAPA overlay 被移出 current surface
- **WHEN** RBMA/KD/BTAPA/weakKD/tau/seed/PatternFiLM overlay 不再被 current docs、OpenSpec specs、tests、claim 表格或主线实验矩阵引用
- **THEN** implementation MUST 删除该配置或将其标记为 historical/local manual
- **AND** current README、docs 和 OpenSpec MUST 不再把旧 overlay 描述为推荐运行入口

#### Scenario: claim 证据配置保留
- **WHEN** 某个配置仍被论文表格、claim provenance、当前 experiment matrix 或复现文档引用
- **THEN** implementation MUST 保留该配置或先提供等价 manifest/generator 输入
- **AND** 删除前 MUST 更新对应 claim provenance 和验证命令

### Requirement: 清理后 current surface 不得留下未分类漂移
缺失模态主线清理完成后，scripts、configs 和 hotspots 的治理检查 MUST 能发现未分类漂移，并 MUST 对重复入口、已退役路线回流和不存在路径引用给出失败或 warning。实现 MAY 保留 local/manual 研究入口，但 MUST 明确其不是 package CLI、quickstart 唯一入口或长期 canonical config。

#### Scenario: surface doctor 作为验收门
- **WHEN** implementation 完成缺失模态主线表面清理
- **THEN** `project_surface_doctor` MUST 能在 scripts、configs 和 hotspots scope 中通过该 change 设定的 fail-on 门槛
- **AND** 输出 MUST 不再列出本 change 范围内的未分类 Scene31/Scene31-34 helper、未分类实验 YAML 或未登记大 owner

#### Scenario: 删除不创建兼容 wrapper
- **WHEN** implementation 删除或合并 local/manual 历史入口
- **THEN** 项目 MUST 不新增同名 thin wrapper、legacy alias、virtual config、二级聚合层或兼容 fallback
- **AND** 文档 MUST 指向当前主线入口、local/manual 登记项、普通 unknown-name 失败或 retired tombstone
