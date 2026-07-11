## ADDED Requirements

### Requirement: Post-C2 删除候选必须退出真实维护面
满足 retired 或 zero-consumer 删除条件的候选 MUST 从源码、测试、public entrypoint、current docs 和独立 current spec 中一起移除。实现 MUST 不以移动目录、增加 wrapper、登记永久例外或新建治理工具代替删除。

#### Scenario: 高置信候选被完整删除
- **WHEN** 候选没有 current config、package CLI、源码调用方、claim provenance 或 protected inventory 引用
- **THEN** implementation MUST 删除实现和只服务该实现的测试、脚本与 current wording
- **AND** Git/OpenSpec archive MAY 保留历史，不得新建 `legacy` package

#### Scenario: 候选因真实消费者保留
- **WHEN** 引用审计发现候选仍被 current workflow 消费
- **THEN** deletion ledger MUST 记录调用路径、owner、focused validation 和删除触发条件
- **AND** “未来可能使用” MUST NOT 作为 retained-with-evidence 理由

## MODIFIED Requirements

### Requirement: 源码表面积优化必须保持核心 workflow 兼容
删除冗余配置、源码模块或入口后，本 change 明确保留的十个 package CLI、final C2/U-Mask、Scene31-34 final analysis、MMW/CSI 和必要 evidence workflow MUST 保持用户可见语义。已登记删除的 dashboard/preview/doctor/query/旧 Scene31 surface 不属于兼容承诺。

#### Scenario: 十个核心 CLI help 继续可用
- **WHEN** cleanup 完成后运行 CLI help characterization
- **THEN** pyproject 中十个 retained console scripts MUST 正常退出
- **AND** tests MUST 不要求 research dashboard/preview 或 project surface doctor

#### Scenario: 保留 workflow 返回结构稳定
- **WHEN** 用户运行 retained training/evaluation/preprocess/U-Mask/MMW/CSI workflow
- **THEN** public config、payload、metric、checkpoint 和 output boundary MUST 保持
- **AND** 内部删除 MUST 不要求未记录的新命令

### Requirement: Active mainline 与 legacy KD 模块边界
项目 MUST 将 final C2/U-MaskBeamJEPA missing-modality beam prediction 作为 active mainline，将 MMW/CSI、JEPA pretraining/mean-context、AMR/AMBER controls 与 current evidence owners作为 supporting/current surface。Image+GPS GPS-query、Vision-Position、JEPA visual/shortcut、geometry 和旧独立 RBMA/KD sweep MUST 标记 retired；U-Mask 内嵌 RBMA/prototype/full-to-partial teacher 分支明确受保护。

#### Scenario: Mainline 导入不触发 retired runtime
- **WHEN** 开发者导入 retained training/model/evaluation owners
- **THEN** import MUST 不要求 GPS-query/visual-shortcut/geometry/legacy KD runtime
- **AND** current U-Mask opt-in branches MUST 继续可构建

#### Scenario: Retired route 不属于 active mainline
- **WHEN** docs/tests 列举 active mainline
- **THEN** 列表 MUST 不包含旧 query/visual/shortcut/geometry/independent RBMA sweep
- **AND** MMW/CSI/U-Mask owners MUST 不被误标 retired

### Requirement: 表面积 inventory 跟随当前主线
项目 surface inventory MUST 将 final C2/U-Mask、十个 package CLI、Scene31-34 final analysis、MMW/CSI、AMR/AMBER controls 和有真实 consumer 的 supporting owners 作为 current/supporting 表面。GPS-query、JEPA visual/shortcut、Vision-Position、geometry、旧 Scene31 和删除 CLI MUST 只在 retired/historical 行出现。

#### Scenario: Inventory 只列真实 current surface
- **WHEN** 开发者阅读 `docs/project_surface_inventory.md`
- **THEN** current/recommended rows MUST 对应真实 pyproject/config/source owner
- **AND** retired rows MUST 不提供 current command

#### Scenario: Supporting 能力有 consumer 证据
- **WHEN** target-shot、architecture summary、AMR-lite、JEPA mean 或 run-index resources 标记 supporting
- **THEN** inventory MUST 记录 current consumer 和 focused validation
- **AND** 不得因此恢复 standalone CLI

### Requirement: 缺失模态主线清理必须先分类再删除
项目在围绕缺失模态鲁棒性主线清理源码、脚本或配置前，MUST 将候选项分类为 current、secondary/supporting、local/manual、historical、retired、delete-candidate 或 merge-candidate。分类记录 MUST 包含 owner、当前调用方或引用、公开入口风险、替代入口、验证命令和回滚方式。未分类候选 MUST 不得被 README、docs、OpenSpec current specs 或 package CLI 描述为当前推荐入口。

#### Scenario: 候选通过 tracked authority 分类
- **WHEN** tracked source、pyproject、configs、current specs、inventory 或 focused tests 发现未分类 scripts、configs 或 owner
- **THEN** implementation MUST 删除该候选、合并到现有 owner，或在 inventory 中登记 retained-with-evidence
- **AND** 最终状态 MUST 不把未分类候选留在 current supported surface

#### Scenario: 当前主线证据链被保护
- **WHEN** 候选项属于 U-MaskBeamJEPA 模型、U-Mask loss、run metadata、migration guard、集中 retired-route guard、当前 Scene31-34 主线 analysis/export 或明确 claim 证据输入
- **THEN** implementation MUST 不直接删除该候选
- **AND** 若仍需缩小，MUST 先记录替代 owner、保持公共语义的验证命令和单独删除触发条件

#### Scenario: 清理不触碰本地产物
- **WHEN** 清理缺失模态主线源码表面
- **THEN** implementation MUST 不删除、移动、重写或纳入 `dataset/`、`outputs/`、`logs/`、checkpoint、cache、TensorBoard event 或本地训练产物
- **AND** 删除验证 MUST 只基于 tracked source、configs、docs、OpenSpec 和 tests

### Requirement: 清理后 current surface 不得留下未分类漂移
缺失模态主线清理完成后，轻量架构检查 MUST 发现未分类漂移，并 MUST 对重复入口、已退役路线回流和不存在路径引用失败。实现 MAY 保留 local/manual 研究入口，但 MUST 明确其不是 package CLI、quickstart 唯一入口或长期 canonical config。

#### Scenario: 结构检查作为验收门
- **WHEN** implementation 完成缺失模态主线表面清理
- **THEN** architecture、CLI/config 和 compile focused checks MUST 通过
- **AND** 检查 MUST 不要求 project surface doctor 或新的 inventory dump 工具存在

#### Scenario: 删除不创建兼容 wrapper
- **WHEN** implementation 删除或合并 local/manual 历史入口
- **THEN** 项目 MUST 不新增同名 thin wrapper、legacy alias、virtual config、二级聚合层或兼容 fallback
- **AND** 文档 MUST 指向当前主线入口、local/manual 登记项、普通 unknown-name 失败或集中 retired-route summary
