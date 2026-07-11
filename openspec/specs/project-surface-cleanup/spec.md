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

### Requirement: Fusion 配置根目录必须保持可维护
`configs/fusion/` 根目录 MUST 只保留长期 canonical 配置或当前文档明确推荐的入口。实验特化、临时复现、低内存补丁、best/last 对照和一次性矩阵配置 MUST 迁移到明确实验子目录、归档说明或删除，且所有引用 MUST 同步更新。

#### Scenario: 收缩根目录 YAML
- **WHEN** 开发者检查 `configs/fusion/*.yaml`
- **THEN** 根目录 YAML 数量 MUST 回到架构 guardrail 允许范围内
- **AND** 每个保留 YAML MUST 能归入 canonical、当前推荐 workflow、或明确保留的薄入口配置

#### Scenario: 迁移配置后引用一致
- **WHEN** 某个 fusion YAML 被迁移、归档或删除
- **THEN** README、docs、scripts、tests 和 OpenSpec 当前 specs MUST 不再把旧路径声明为当前支持入口
- **AND** 若仍需保留复现路径，文档 MUST 指向新的明确位置或说明该配置已退役

### Requirement: 冗余源码删除必须保守
项目 MAY 删除无当前调用的源码 helper 或模块，但 MUST 先确认其不属于 console script、公开导出、注册入口、README/docs/OpenSpec 声明或测试依赖。无法确认外部依赖时，项目 MUST 优先保留源码并在 inventory 或后续 change 中记录待收敛项。

#### Scenario: 删除孤立 helper
- **WHEN** CodeGraph 或结构检查显示某个 helper 无当前内部调用
- **THEN** 开发者 MUST 进一步检查公开 API、配置注册、CLI、文档和测试引用
- **AND** 只有这些入口均不依赖该 helper 时，源码删除才可进入实现任务

#### Scenario: 保留可能的公共 API
- **WHEN** 某个无内部调用模块可能被外部脚本、公开导出或文档声明使用
- **THEN** 本 change MUST 不直接删除该模块
- **AND** 清理结果 MUST 记录保留原因或提出后续单独退役 change

### Requirement: 实验配置支持面分类
项目 MUST 对当前支持的配置文件进行生命周期分类，至少区分 canonical/root 推荐入口、实验复现配置、debug/smoke 配置、dataset preparation 配置、diagnostics 配置和已退役历史记录。新增、迁移或删除配置时，项目 MUST 同步更新 inventory、引用文档和相关架构 guardrail。

#### Scenario: 实验子目录配置有归属
- **WHEN** 开发者在 `configs/` 下新增实验特化 YAML
- **THEN** 该配置 MUST 位于语义明确的子目录或被 inventory 分类说明
- **AND** README、docs、OpenSpec 或脚本中的引用 MUST 指向真实存在的路径
- **AND** 该配置 MUST 不通过 root `configs/fusion/` 混入长期 canonical 入口，除非 inventory 明确将其列为当前推荐入口

#### Scenario: 配置引用漂移被发现
- **WHEN** 架构边界测试扫描 README、docs、scripts 和当前 OpenSpec specs 中的配置路径引用
- **THEN** 测试 MUST 能发现指向不存在配置文件的当前支持面引用
- **AND** 历史 archive 或明确标记为退役记录的引用 MUST 不被误判为当前入口

### Requirement: 实验配置删除必须保留证据链
项目 MAY 删除重复、生成型或历史 local/manual experiment YAML，但 MUST 保留当前 claim/evidence、paper reproduction、diagnostics manifest 和必要 focused tests 的配置证据链。删除说明 MUST 记录替代 generator/manifest/base config、删除原因、引用同步和回滚方式。

#### Scenario: claim 配置被保护
- **WHEN** YAML 被 `docs/result_claims_registry.md`、`docs/mainline_model_catalog.md`、`docs/experiment_matrix.md`、当前 OpenSpec specs 或 focused tests 引用为证据输入
- **THEN** implementation MUST 保留该 YAML，或先更新 claim provenance 指向等价 generator/manifest 输入
- **AND** 删除后 current docs MUST 不指向不存在路径

#### Scenario: historical sweep 配置可删除
- **WHEN** YAML 只服务已沉淀的历史 sweep、local queue 或被 generator 覆盖的 seed/missing-pattern 组合
- **THEN** implementation MAY 删除实体 YAML
- **AND** 有价值的结论、caveat 或复跑方式 MUST 保留在 docs、inventory 或 result provenance 中

#### Scenario: 清理不触碰 runtime artifact
- **WHEN** implementation 收缩 experiment config family
- **THEN** implementation MUST 不删除、移动、重写或纳入 `outputs/`、`logs/`、checkpoint、cache、TensorBoard event 或真实 `dataset/`
- **AND** generator focused tests MUST 使用临时目录或受控源码配置路径，不写入真实训练产物

### Requirement: 配置表面收缩必须同步引用
当实体 YAML 被删除、迁移、生成化或降级为 local/manual 后，README、docs、OpenSpec current specs、scripts 默认路径、tests 和 inventory MUST 同步更新。健康检查 MUST 能发现 current 引用指向不存在配置。

#### Scenario: stale config reference 被发现
- **WHEN** current docs、scripts、tests 或 OpenSpec specs 仍引用已删除 YAML
- **THEN** architecture/config/surface 检查 MUST 失败或报告 error
- **AND** 修复路径 MUST 是恢复配置、更新引用到 generator/manifest，或将引用标记为 historical

#### Scenario: root/canonical surface 不被实验 YAML 污染
- **WHEN** experiment family shrink 后仍保留 local/manual YAML
- **THEN** 该 YAML MUST 位于语义明确的 experiment/local 目录或被 inventory 分类
- **AND** 它 MUST 不被迁入 root canonical config surface，除非 OpenSpec 明确将其提升为 current canonical entry

### Requirement: Ponytail 审计候选必须有删除证据
项目 MUST 在删除或合并低价值源码、配置、脚本、测试或文档入口前记录最小证据：当前调用方、公开入口风险、替代 owner、是否被 registry/CLI/current docs/OpenSpec 消费、验证命令和回滚方式。没有证据的候选 MUST 保留或降为后续单独 change。

#### Scenario: 删除候选具备证据
- **WHEN** 开发者准备删除 ponytail 审计列出的候选项
- **THEN** change artifact 或实现说明 MUST 记录该候选不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 记录 MUST 指向替代 owner、替代 recipe、普通 unknown-name 行为或说明无需替代

#### Scenario: 保留候选具备理由
- **WHEN** 某个候选因 public API、外部脚本风险、manifest 安全边界或当前 workflow 消费而保留
- **THEN** inventory、任务说明或最终实现说明 MUST 记录保留理由
- **AND** 项目 MUST 不为保留该候选新增兼容 wrapper、二级聚合层或重复治理表

### Requirement: Ponytail 审计表面必须分类收口
项目 MUST 将 ponytail 审计确认的临时配置、一次性脚本、root runbook、薄 facade、重复 helper 和本地工具状态分类为删除、迁移、保留或后续 change。分类 MUST 记录 owner、当前调用方、公开 surface 风险、替代入口、验证命令和回滚方式。未分类项 MUST 不得作为 current README、docs、OpenSpec 或 package CLI 推荐入口。

#### Scenario: 新增脚本被分类
- **WHEN** `scripts/` 或 `tools/analysis/` 下存在新增 Python/shell 脚本
- **THEN** inventory MUST 将其分类为 package_cli、research_diagnostic、dataset_preparation、figure_helper、config_generator 或 local/manual artifact
- **AND** 未分类脚本 MUST 不得被 README、docs 或 OpenSpec 描述为当前推荐入口

#### Scenario: 临时配置不进入 root canonical surface
- **WHEN** 新增配置只服务 Scene31/RBMA queue/fullrun/strong-encoder/seed sweep 或其它本地实验编排
- **THEN** 配置 MUST 位于语义明确的 experiment 子目录、被归档说明，或被删除
- **AND** 配置 MUST 不直接混入 `configs/fusion/*.yaml` 根目录，除非 inventory 将其登记为 canonical/current thin entry

### Requirement: Fusion 根配置与 inventory 必须一致
`configs/fusion/` 根目录的实体 YAML MUST 与 `docs/project_surface_inventory.md` 中的 root canonical/current 分类一致。若 root YAML 集合变化，项目 MUST 同步更新 inventory、引用文档和架构边界测试；若文件不属于 root canonical/current thin entry，项目 MUST 将其迁入 experiment 子目录、归档或删除。

#### Scenario: root YAML 集合被验证
- **WHEN** 架构边界测试扫描 `configs/fusion/*.yaml`
- **THEN** 每个根 YAML MUST 出现在 inventory 的 root 保留分类中
- **AND** 未登记 root YAML MUST 被视为支持面漂移

#### Scenario: 实验 YAML 迁移后引用同步
- **WHEN** root fusion YAML 被迁移到 `configs/fusion/experiments/<family>/`
- **THEN** README、docs、scripts、tests 和 OpenSpec 中的 current 引用 MUST 指向新路径或移除
- **AND** 历史引用 MUST 明确标记为 historical、retired 或 local/manual

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

### Requirement: 入口收敛不得让研究脚本成为核心依赖
保留的 `scripts/` 和 `tools/analysis/` 研究或支持脚本 MUST 不成为核心训练、评估、预处理或 manifest 导出 workflow 的必需依赖。仓库级 `tools/visualization/` viewer support 已退役，核心 workflow MUST 通过包内模块或 package console script 完成。

#### Scenario: 训练入口不依赖研究脚本
- **WHEN** 用户运行 `kd-sensing-train` 或 `python -m kd_sensing.cli.train`
- **THEN** 训练 workflow MUST 不要求调用 `scripts/analyze_*`、`tools/analysis/*` 或 viewer 支持脚本
- **AND** 研究脚本删除或重分类 MUST 不破坏核心训练入口

#### Scenario: viewer manifest 边界清晰
- **WHEN** 用户引用 viewer manifest 导出或 `kd-sensing-visualize-modalities` 兼容入口
- **THEN** 系统 MUST 拒绝该退役入口或不再提供该入口
- **AND** 仓库级 Gradio viewer entrypoint MUST 不再作为当前支持脚本保留

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

### Requirement: 新主线方法不得包含 distillation 配置段
新 current mainline 配置和运行时 MUST 用 `model.primary` 与 supervised/adaptation loss 表达训练。任何 `distillation.*`、`teacher_model_name`、`logits_kd`、`rkd` 或旧 `*_no_kd` 路径 MUST 在配置解析阶段失败并给出迁移建议。

#### Scenario: 当前配置无需 distillation 字段
- **WHEN** 用户加载当前推荐的 supervised/adaptation mainline 配置
- **THEN** 配置 validation MUST 不要求 `distillation.teacher_model_name`
- **AND** 最终配置 MUST 不包含 KD temperature、alpha 或 RKD 权重字段

#### Scenario: 旧 no_kd 字段被拒绝
- **WHEN** 用户加载仍包含 `distillation.type: no_kd` 的历史配置
- **THEN** 系统 MUST 拒绝该配置并提示 strong、lightweight 或 supervised 入口
- **AND** 系统 MUST 不把该 run 作为可运行 baseline

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

### Requirement: Generated Scene31 YAML 必须可清理或有理由保留
Scene31 generated YAML MUST 不静默扩大源码表面。需要长期跟踪的实体 YAML MUST 有 current/local/manual 保留理由；可由 generator 和 manifest 无损重建的 YAML MUST 改为本地生成产物或登记删除计划。

#### Scenario: 可再生成 YAML 不长期堆积
- **WHEN** generator 能从 template 和 manifest 重建 Scene31 YAML
- **THEN** 源码表面积治理 MUST 优先保留 generator、manifest 和 template
- **AND** 实体 YAML 若继续跟踪，MUST 说明不可由 generator 无损重建的字段或人工样例价值

#### Scenario: 清理不触碰运行产物
- **WHEN** 清理 Scene31 源码配置表面
- **THEN** 实现 MUST 不删除、移动或重写 `outputs/scene31*`、`logs/`、checkpoint、fresh eval 结果或本地 cache

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
缺失模态主线清理完成后，轻量架构检查 MUST 发现未分类漂移，并 MUST 对重复入口、已退役路线回流和不存在路径引用失败。实现 MAY 保留 local/manual 研究入口，但 MUST 明确其不是 package CLI、quickstart 唯一入口或长期 canonical config。

#### Scenario: 结构检查作为验收门
- **WHEN** implementation 完成缺失模态主线表面清理
- **THEN** architecture、CLI/config 和 compile focused checks MUST 通过
- **AND** 检查 MUST 不要求 project surface doctor 或新的 inventory dump 工具存在

#### Scenario: 删除不创建兼容 wrapper
- **WHEN** implementation 删除或合并 local/manual 历史入口
- **THEN** 项目 MUST 不新增同名 thin wrapper、legacy alias、virtual config、二级聚合层或兼容 fallback
- **AND** 文档 MUST 指向当前主线入口、local/manual 登记项、普通 unknown-name 失败或集中 retired-route summary

### Requirement: Post-C2 清理必须保护主线与 MMW
项目在 post-C2 表面积清理中 MUST 先建立 protected inventory，再删除源码、配置、脚本、测试或文档入口。protected inventory MUST 至少覆盖 final C2 / U-MaskBeamJEPA 缺失模态主线、当前 claim/evidence 输入、MMW/CSI 数据集与 workflow、主线 YAML/manifest、以及 U-MaskBeamJEPA 已存在 fusion 分支实现。

#### Scenario: protected inventory 阻止误删
- **WHEN** implementation 准备删除任意 `src/`、`configs/`、`scripts/`、`tests/` 或 docs 文件
- **THEN** 删除候选 MUST 先在 protected inventory 或 deletion candidate ledger 中分类
- **AND** 属于 final C2、U-MaskBeamJEPA 主线、MMW/CSI、current claim/evidence 或主线 YAML/manifest 的文件 MUST 不进入删除任务

#### Scenario: MMW 支线保留
- **WHEN** implementation 执行 post-C2 表面积清理
- **THEN** `src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw*.py`、MMW GPS v2、physics-informed MMW、CSI hardening、MMW/CSI configs、MMW/CSI tests 和相关 package CLI MUST 保留
- **AND** 文档 MUST 将 MMW 标记为 future dataset workflow 或 current supporting workflow，而不是删除候选

### Requirement: 主线 YAML 和 manifest 删除必须有证据
项目 MUST 保护仍被主线使用的 YAML、CSV、JSON manifest 或 config reference。删除或生成化配置前，implementation MUST 证明该文件不被 final C2、current Scene31/Scene31-34 evidence、claim registry、experiment matrix、OpenSpec current spec、focused tests 或用户标记的主线输入消费。

#### Scenario: current evidence config 被保护
- **WHEN** YAML 或 manifest 被 claim registry、experiment matrix、mainline docs、OpenSpec current spec、final C2 launcher、focused tests 或用户标记引用
- **THEN** implementation MUST 保留该文件
- **AND** 若后续必须删除，MUST 先把 provenance 更新到等价 generator、base config 或 manifest 输入

#### Scenario: historical generated config 可删除
- **WHEN** YAML 或 manifest 只服务历史 sweep，且可由 generator/template 无损重建，并且不属于 current claim/evidence
- **THEN** implementation MAY 删除实体文件
- **AND** deletion candidate ledger MUST 记录 generator、替代入口、验证命令和回滚方式

### Requirement: U-MaskBeamJEPA fusion 分支本轮不得删除
Post-C2 表面积清理 MUST 不删除或修改 U-MaskBeamJEPA 中已存在的 fusion、router、forward 或 loss 分支实现。未胜出分支 MAY 在 inventory 中标记为后续审计候选，但删除 MUST 通过独立 OpenSpec change 再实施。

#### Scenario: fusion 分支保持源码存在
- **WHEN** implementation 触碰 `src/kd_sensing/models/u_mask_beam_jepa.py` 或 U-Mask loss/config helper
- **THEN** 本 change MUST 不删除 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 或既有 branch/loss 开关
- **AND** focused tests MUST 继续覆盖保留分支的至少 smoke 或 registry/config 行为，除非后续独立 change 明确退役

### Requirement: 删除必须按可回滚波次执行
Post-C2 清理 MUST 分波次实施，每波只处理一个稳定边界：保护扫描、入口/文档收口、一次性脚本删除、非主线源码测试删除、历史配置收缩、guardrail 收口。每波 MUST 记录验证命令和失败回滚方式。

#### Scenario: wave 完成后验证
- **WHEN** 某个删除 wave 完成
- **THEN** implementation MUST 运行该 wave 对应的 focused validation
- **AND** 若验证失败，implementation MUST 能恢复本 wave 删除的文件，而不是继续叠加下一波删除

#### Scenario: 不创建兼容入口
- **WHEN** 某个非主线入口被删除
- **THEN** 项目 MUST 不新增同名 wrapper、stub CLI、virtual config、package facade 或 compatibility alias
- **AND** 文档 MUST 指向保留主线入口、MMW 入口、historical note、普通 unknown-name 错误或 retired tombstone

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

