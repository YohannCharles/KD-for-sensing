# mainline-experiment-documentation Specification

## Purpose
定义当前主线模型目录、实验协议表、结果/claim 账本、baseline 报告分层和跨文档索引规则，使维护者能区分当前可引用事实、local substitute、blocked official、upper-bound、mock/smoke 和 historical ablation。
## Requirements
### Requirement: 主线模型目录
项目 MUST 维护当前主线模型目录，用于集中说明 final C2 / U-MaskBeamJEPA、必要 baseline/control、MMW/CSI supporting workflow 和当前证据 owner 的研究问题、配置入口、数据口径、指标口径、运行状态与结果引用。目录 MUST 位于 `docs/mainline_model_catalog.md` 或等价 current 文档中，并 MUST 不把 retired、historical、supporting-only 或 mock-only workflow 描述为当前推荐入口。

#### Scenario: 主线模型目录覆盖当前支持面
- **WHEN** 开发者阅读主线模型目录
- **THEN** 文档 MUST 覆盖 final C2 / U-MaskBeamJEPA、U-Mask eval matrix、仍用于对照的 AMR/AMBER、MMW GPS v2、physics-informed MMW、CSI hardening 和 current paper evidence owner
- **AND** 每行 MUST 标明 config/入口、数据集/场景、split/target、metric profile、运行状态和 caveat

#### Scenario: 退役路线不得进入 current 主线表
- **WHEN** 主线模型目录提到 Image+GPS JEPA query-pool、Vision-Position、BeamBench、BEV-Fusion、KD、Hist、geometry prior 或旧 Scene31 workflow
- **THEN** 对应行 MUST 标记为 retired、historical 或 compatibility context
- **AND** 文档 MUST 不提供这些路线的 current 推荐训练命令

### Requirement: 实验协议和参数表
项目 MUST 维护实验协议和参数表，用于将主要配置族的正式口径、smoke/debug/lowmem 口径、upper-bound 口径和历史 ablation 口径分开。该表 MUST 位于 `docs/experiment_protocols.md` 或等价 current 文档中，并 MUST 能让读者不打开多个 YAML 也能判断实验是否可横向比较。

#### Scenario: 参数表记录可比性字段
- **WHEN** 文档列出一个实验配置族
- **THEN** 表格 MUST 记录 config path、run status、dataset/scenes、split protocol、selection split、seed、epochs、batch size、learning rate、seq_len、num_pred、target source、GPS feature mode、label space、metric profile、输出目录和 focused validation 命令
- **AND** 对 smoke、debug、lowmem、upper-bound、mock 或 historical ablation 条目 MUST 显式标记状态

#### Scenario: benchmark 配置状态清晰
- **WHEN** 表格列出 JEPA shortcut benchmark、difficulty profile 或其它当前诊断配置
- **THEN** 文档 MUST 标明该配置是正式实验、quick validation、smoke schema check、diagnostic-only、evaluation-only 还是 upper-bound
- **AND** 文档 MUST 标明输出仍写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定目录

### Requirement: 实验文档必须跟随配置生成化
当 experiment config family 的实体 YAML 被删除、生成化或降级为 local/manual 后，主线模型目录、实验矩阵、协议表和 result claim registry MUST 指向真实存在的 current config、generator/manifest 输入、base config 或明确 historical/local/manual 说明。

#### Scenario: experiment matrix 不指向删除配置
- **WHEN** 开发者阅读 `docs/experiment_matrix.md` 或 `docs/mainline_model_catalog.md`
- **THEN** current workflow 行 MUST 不引用已删除实体 YAML 作为可运行入口
- **AND** 若复跑需要生成配置，文档 MUST 指向 generator、manifest、base config 和输出边界

#### Scenario: claim provenance 保持可审计
- **WHEN** claim 或 pending claim 依赖的 YAML 被生成化或删除
- **THEN** `docs/result_claims_registry.md` 或等价 provenance MUST 更新为保留 YAML、generator/manifest 输入或 historical caveat
- **AND** pending/mock/local/manual 状态 MUST 不因删除实体 YAML 而被提升为 promoted claim

### Requirement: 配置族状态必须出现在主线文档
主线文档 MUST 区分 current recommended config、paper/workflow reproduction config、secondary/supporting config、local/manual overlay、generated config 和 historical config。读者 MUST 能从文档判断一个配置是否适合直接运行、是否需要本地 checkpoint/outputs、以及是否支撑当前 claim。

#### Scenario: local/manual 配置不冒充 current
- **WHEN** 文档列出 Scene31、RBMA、strong encoder 或 JEPA image+GPS local/manual 配置
- **THEN** 文档 MUST 标记其 local/manual 或 pending 状态、输出边界和主要 caveat
- **AND** 文档 MUST 不把缺少 evidence 或依赖本地 checkpoint 的配置描述为 promoted mainline result

#### Scenario: generated 配置有复跑路径
- **WHEN** 文档描述一个不再跟踪实体 YAML 的 generated config family
- **THEN** 文档 MUST 给出 generator/manifest/base config 或 package CLI 复跑路径
- **AND** 文档 MUST 说明生成产物默认进入 ignored local output/config directory

### Requirement: 结果和 claim 账本
项目 MUST 维护结果和 claim 账本，用于记录可引用结果的来源、口径和限制。账本 MUST 只保存摘要、配置路径、本地产物路径引用和 caveat；真实 checkpoint、metrics、figures、cache 和 logs MUST 继续作为本地产物留在 ignored 路径。

#### Scenario: 结果账本记录 claim provenance
- **WHEN** 项目文档引用某个实验数值作为当前结论或论文写作支撑
- **THEN** 结果账本 MUST 记录 claim id、model line、config path、run date 或 commit、dataset/split、target source、metric field、数值摘要、checkpoint provenance、claim status 和 caveat
- **AND** claim status MUST 区分 official reproduction、local substitute、local strict-validation、upper-bound、mock/smoke、historical ablation、unverified 或 blocked

#### Scenario: 结果账本不提交运行产物
- **WHEN** 结果账本引用 checkpoint、predictions、figures、tables、cache 或 logs
- **THEN** 账本 MUST 只记录路径、digest、摘要或本地 artifact 名称
- **AND** 源码变更 MUST NOT 要求提交真实 checkpoint、metrics CSV、figures、cache、TensorBoard event 或训练日志

### Requirement: baseline 报告状态分类
项目 MUST 在 current claim、protocol 和历史账本中使用统一状态分类，将 official blocked、local substitute、strict-validation、upper-bound、mock/smoke 和 historical ablation 分开。已失去运行 owner 的 root 复现报告 MUST 在迁移唯一有效结论后删除，不得继续充当 current summary 或命令入口。

#### Scenario: 旧 BeamBench root 报告退出 current surface
- **WHEN** BeamBench package、CLI、配置和脚本已经退役
- **THEN** `README_REPRODUCE.md`、`BASELINE_REPORT.md`、`DATASET_STRUCTURE.md`、`PATCH_NOTES.md`、`TODO_FOR_ATTENTION_MODULE.md` 和 `results/reproduce_baseline.md` MUST 被删除
- **AND** blocked/not-comparable 结论 MUST 迁入 `docs/mainline_experiment_history.md` 与 claim registry，且不得保留旧推荐命令

#### Scenario: Current claim 状态有唯一权威来源
- **WHEN** 开发者查找当前可引用结果或复现 gate
- **THEN** README MUST 指向 `docs/result_claims_registry.md`、`docs/experiment_protocols.md` 和 `docs/mainline_experiment_history.md`
- **AND** 历史输出、checkpoint、metrics 和日志 MUST 继续留在 ignored 本地产物边界

### Requirement: 文档索引和生命周期
README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md` 和 OpenSpec lifecycle inventory MUST 指向主线模型目录、实验协议表和结果账本。新增文档 MUST 被归入 current workflow guide 或 current reproducibility/reporting 文档生命周期，不得漂移成未分类文档。

#### Scenario: README 只保留短索引
- **WHEN** 开发者阅读 README 的文档索引或实验矩阵说明
- **THEN** README MUST 指向主线模型目录、实验协议表和结果账本
- **AND** README SHOULD 保留 quickstart 和关键 caveat，而不是复制完整结果账本

#### Scenario: inventory 记录新增文档
- **WHEN** 新增 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md` 或 `docs/result_claims_registry.md`
- **THEN** `docs/project_surface_inventory.md` MUST 记录这些文档的生命周期和职责
- **AND** OpenSpec lifecycle inventory MUST 将 `mainline-experiment-documentation` 标记为 current

### Requirement: 主线文档不得保留优先退役 claim 行
主线模型目录、实验协议表和结果 claim 账本 MUST 从 current 表中移除 AMR-Net_gps_image 和 JEPA-MSAC 的 pending、mock-smoke、blocked official 或 local-ready current 行。若保留历史背景，MUST 放入 retired/historical/tombstone 说明，不得作为当前可运行或待引用 claim 占位。

#### Scenario: 主线目录不列 AMR 和 JEPA-MSAC current 行
- **WHEN** 开发者阅读 `docs/mainline_model_catalog.md`
- **THEN** 文档 MUST 不把 AMR-Net_gps_image 列为 current model line
- **AND** 文档 MUST 不把 JEPA-MSAC Scenario 32 列为 current model line
- **AND** 如提到二者，MUST 标记为 retired、historical 或 blocked background

#### Scenario: claim 账本不保留 current pending 占位
- **WHEN** 开发者阅读 `docs/result_claims_registry.md`
- **THEN** 账本 MUST 不保留 AMR-Net_gps_image 或 JEPA-MSAC 的 current pending/mock-smoke claim 行
- **AND** 账本 MAY 保留一段退役说明，解释历史 blocked 原因和本地产物只作为 archive 背景

### Requirement: 当前文档只推荐保留入口
README、实验矩阵和协议表 MUST 只推荐仍维护的 current package CLI、config、diagnostic 或 shell runner。被退役入口的命令 MAY 出现在历史说明中，但 MUST 明确不可作为当前 quickstart 或正式复现实验。

#### Scenario: README quickstart 无退役命令
- **WHEN** 开发者阅读 README 的 quickstart、实验矩阵索引或 MMW 小节
- **THEN** README MUST 不提供 `kd-sensing-run-amr-net-gps-image` 或 `kd-sensing-run-jepa-msac` 作为当前命令
- **AND** README MUST 不提供被退役 shell orchestration 脚本作为当前命令

### Requirement: 主线文档记录 AMBER full pending 状态
主线模型目录、实验协议表和结果 claim 账本 MUST 记录 AMBER full architecture reproduction 的本地 pending 状态、配置入口、输出边界、指标口径和 caveat。文档 MUST 区分 AMBER-lite、AMBER full local architecture reproduction 和任何未来 official AMBER reproduction。

#### Scenario: 主线目录包含 AMBER full 条目
- **WHEN** AMBER full 配置和 focused tests 落地
- **THEN** `docs/mainline_model_catalog.md` 或等价 current 文档 MUST 记录其 model line、config、入口命令、数据集/场景、metric profile、run status 和 caveat
- **AND** 该条目 MUST 标记为 local architecture reproduction 或 pending，直到真实严格可比结果存在

#### Scenario: claim 账本不写入未验证数值
- **WHEN** AMBER full 只有 synthetic tests、dry-run 或未完成训练
- **THEN** `docs/result_claims_registry.md` MUST NOT 填入真实性能数值
- **AND** 它 MUST 只记录 pending/unverified 状态、输出路径边界和升级条件

### Requirement: 推荐实验文档保持精简入口
实验工作流文档 MUST 将 README 作为入口地图，而不是完整实验手册。README MUST 指向 canonical config、docs 和 OpenSpec；详细实验矩阵、分析流程和调参说明 MUST 放在 `docs/` 或对应 specs 中。已退役的 G2D、CRAF、MARF 和 Multimodal-NF 内容 MUST 从 README 推荐入口和实验矩阵中删除。

#### Scenario: README 提供最短可运行路径
- **WHEN** 新用户阅读 README
- **THEN** 用户 MUST 能找到安装命令、快速健康检查、训练/评估/预处理/manifest 导出入口和数据产物边界
- **AND** 用户 MUST 能通过链接进入当前保留能力的详细实验矩阵或 viewer 文档

#### Scenario: 长实验说明迁移到 docs
- **WHEN** README 中的某段内容主要描述当前保留的 CSI hardening、MMW、JEPA 或诊断 benchmark 详细实验流程
- **THEN** 该内容 MUST 迁移到对应 `docs/` 文件或 OpenSpec spec
- **AND** README MUST 保留简短摘要和链接

#### Scenario: 退役研究线文档删除
- **WHEN** README、docs 或实验矩阵提到 G2D、CRAF、MARF 或 Multimodal-NF 推荐运行命令
- **THEN** 这些段落 MUST 被删除或改为明确说明该入口已退役
- **AND** 文档 MUST 不再推荐运行对应配置、测试或日志分析流程

### Requirement: 默认实验入口去 KD-first 化
项目默认 quickstart、README 推荐入口、当前主线 quick validation 和新 canonical mainline 配置 MUST 以 supervised/adaptation、JEPA、CSI hardening、baseline/control 或当前诊断工作流为默认。旧 KD、BGAM 和 viewer manifest 配置不得作为当前主线默认实验入口。

#### Scenario: README quickstart 使用当前主线
- **WHEN** 开发者阅读 README 或当前主线运行说明
- **THEN** 推荐的首个训练、评估或诊断命令 MUST 使用当前 supervised/adaptation、JEPA、CSI、baseline/control 或当前诊断配置
- **AND** 文档 MUST 不把 `logits_kd`、`rkd`、Hist/HiST、standalone Top8 selector、GPS residual 或 camera residual 作为当前主线 quickstart

#### Scenario: canonical mainline 配置不要求 teacher checkpoint
- **WHEN** 用户加载当前推荐的 mainline 配置
- **THEN** 配置 MUST 能在没有 teacher checkpoint 的情况下完成解析和 dry-run/smoke 构建
- **AND** 输出 metadata MUST 不记录 KD-enabled lineage

### Requirement: Statistical and stress claim governance
主线实验文档 MUST 记录统计显著性和 stress suite 对 claim 升级的要求。单 seed、smoke、mock、not-comparable 或缺少 stress provenance 的结果 MUST 不写成正式论文结论。

#### Scenario: claim registry 记录统计证据
- **WHEN** 某缺失模态结果升级为 local strict-validation 或 local experimental claim
- **THEN** claim registry MUST 记录 seed_count、baseline、primary metric、mean/std 或 CI、comparability status、stress suite status 和 caveat
- **AND** 缺少任一必要证据时 claim status MUST 保持 pending、unverified、not_comparable 或 mock/smoke

#### Scenario: 实验矩阵区分 smoke 和 formal stress
- **WHEN** `docs/experiment_matrix.md` 或协议表列出 missing-modality stress suite
- **THEN** 文档 MUST 标明该 manifest 是 smoke、quick、formal、diagnostic-only 还是 evaluation-only
- **AND** 文档 MUST 指向 ignored 输出目录，不要求提交真实 stress metrics 或图表

### Requirement: Paper export and literature documentation
主线实验文档 MUST 索引 paper artifact export、当前只读 dataset inspection 和 literature matrix，并说明它们的 claim 状态与本地产物边界。

#### Scenario: 文档索引 paper export
- **WHEN** README 或 `docs/experiment_matrix.md` 给出 paper export 说明
- **THEN** 文档 MUST 说明 export 只消费满足 reviewed gate 的 claim rows
- **AND** pending、mock、historical、unknown 和 candidate-only rows 默认不得进入 main table

#### Scenario: 文档索引 dataset inspection
- **WHEN** current 文档给出数据检查入口
- **THEN** 文档 MUST 只引用 on-disk current script lifecycle 或 public package CLI 中存在的入口
- **AND** inspection MUST 只读、不移动数据，也不代表 official reproduction 已完成

#### Scenario: inventory 记录 literature matrix
- **WHEN** 存在 `docs/literature_matrix.md` 或 `paper/references.bib`
- **THEN** inventory MUST 记录其文档生命周期和职责
- **AND** 文档 MUST 不把本地 PDF 或外部论文下载物纳入源码产物要求

### Requirement: Scene31-34 missing-modality mainline documentation
主线实验文档 MUST 记录 Scene31-34 pooled multi-scene 缺失模态主实验的当前地位、运行入口、指标口径、输出边界和 claim 状态。文档 MUST 明确 `prototype + random subset exposure` 是冻结主方法候选，Uniform 是 ablation，reliability fusion 与 PatternFiLM 不晋升。

#### Scenario: 主线目录记录 Scene31-34 主设定
- **WHEN** 开发者阅读 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 指向 Scene31-34 主实验 runner、summary、missing-count figures、paper tables 和 final conclusion 的本地输出路径
- **AND** 文档 MUST 说明真实 metrics、figures、tables、logs 和 checkpoint 仍属于 ignored runtime artifacts，不纳入源码变更

#### Scenario: 文档不推广 excluded methods
- **WHEN** 文档描述 Scene31-34 缺失模态主实验
- **THEN** 文档 MUST 不把 reliability fusion、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 或 weakKD 写成下一步主线搜索方向
- **AND** AMR/AMBER-lite MUST 只作为可选 multi-scene maskfix external baseline 说明

#### Scenario: 文档记录论文 baseline 与成本补齐
- **WHEN** 文档描述 Scene31-34 主实验的最终论文产物
- **THEN** 文档 MUST mention classifier baselines、AMR/AMBER-lite maskfix external baselines、compute profile table and final all-baseline paper tables as local/manual outputs
- **AND** generated metrics、profile CSV、paper tables and conclusions MUST remain ignored runtime artifacts under `outputs/`

### Requirement: 主实验证据收敛记录
主线实验文档 MUST 在 final C2 / U-MaskBeamJEPA 进入证据收敛阶段时记录 final checklist、缺失 evidence、claim status 和下一步最小动作。文档 MUST 区分继续补证据与新增方法搜索，并在主方法冻结时说明冻结边界。

#### Scenario: Scene31-34 evidence 更新
- **WHEN** Scene31-34 final summary、paper tables 或 claim status 发生变化
- **THEN** mainline history、model catalog、experiment protocols 或 result claims registry 中的对应 current fact MUST 同步更新
- **AND** 真实 metrics、figures、logs 和 checkpoint MUST 继续留在 ignored output root

#### Scenario: Temporal evidence 更新
- **WHEN** H5/P1 temporal matrix 或 final C2 missing-modality evidence 从 smoke 转为可比较结果
- **THEN** 文档 MUST 更新 claim status、config/manifest provenance 和 caveat
- **AND** synthetic smoke 结果 MUST 继续标记为 mock/smoke

### Requirement: 主线文档必须反映 post-C2 边界
主线文档 MUST 将当前默认研究主线描述为 final C2 / U-MaskBeamJEPA 缺失模态波束预测，并明确 MMW/CSI 为保留的 current supporting dataset workflow。非主线历史复现、一次性诊断、research dashboard/preview 和已删除 CLI MUST 不再出现在 current recommended workflow 表中。

#### Scenario: current mainline 表述
- **WHEN** 开发者阅读 README、current research brief、mainline model catalog 或 experiment matrix
- **THEN** 文档 MUST 将 final C2 / U-MaskBeamJEPA、U-Mask eval matrix、claim/evidence gate 和保留 MMW/CSI workflow 描述为当前重点
- **AND** 文档 MUST 不把 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、geometry prior 或旧 RBMA/KD sweep 描述为当前默认主线

#### Scenario: MMW 保留状态清楚
- **WHEN** 文档列出 MMW/CSI 相关入口
- **THEN** 文档 MUST 标明保留原因、数据集用途、入口命令、输出边界和 focused validation
- **AND** 文档 MUST 不把 MMW/CSI 列为 post-C2 删除候选

### Requirement: 删除候选文档必须迁移或降级
被 post-C2 清理删除的历史报告、runbook、README 段落或实验矩阵行 MUST 被迁移为 concise historical note，或从 current docs 删除。保留的历史说明 MUST 不提供当前推荐运行命令。

#### Scenario: 历史报告删除前迁移结论
- **WHEN** implementation 删除历史 Markdown、results log 或一次性报告
- **THEN** 仍有价值的结论 MUST 迁移到 `docs/mainline_experiment_history.md`、`docs/result_claims_registry.md`、inventory historical note 或等价 current 文档
- **AND** 迁移后的说明 MUST 标记为 historical、retired、blocked 或 caveat，而不是 promoted claim

#### Scenario: 当前 docs 不引用删除 runbook
- **WHEN** historical `scripts/` runbook 或 package CLI 被删除
- **THEN** README、experiment matrix、protocol table 和 current OpenSpec specs MUST 不再推荐该命令
- **AND** 若需要复跑历史实验，文档 MUST 指向 preserved generator/base config、final C2 入口、MMW 入口或明确 historical note

### Requirement: Claim provenance 保护主线 YAML/manifest
主线 claim/protocol 文档 MUST 继续指向真实存在且受保护的 YAML/manifest，或指向等价 generator/base config。删除配置不得使 pending 或 reviewed claim 失去 provenance。

#### Scenario: provenance 输入被保护
- **WHEN** `docs/result_claims_registry.md`、`docs/experiment_protocols.md` 或 `docs/mainline_model_catalog.md` 引用 YAML/manifest 作为主线 evidence
- **THEN** implementation MUST 保留该 YAML/manifest 或先更新 provenance 到等价输入
- **AND** claim status MUST 不因删除历史文件而自动升级

#### Scenario: 不确定主线用途时暂缓删除
- **WHEN** implementation 无法确认某个 YAML/manifest 是否会被用户主线使用
- **THEN** 该文件 MUST 标记为 `pending-confirmation` 或 `protected-until-next-audit`
- **AND** 本 change MUST 不删除该文件

### Requirement: 泄漏影响的 temporal evidence 必须降级
主线 claim、protocol 和 history 文档 MUST 将使用逐样本拆分重叠 temporal window 或 test-as-validation 的结果标为 `not_comparable` 或 `invalidated`。此类结果 MUST NOT 进入 reviewed main claim 或 paper main table。

#### Scenario: H5/P1 旧 split evidence
- **WHEN** 文档引用修复前 H5/P1 temporal matrix 结果
- **THEN** claim status MUST 标记为 `not_comparable`
- **AND** caveat MUST 记录 sequence group 与历史/target 帧跨 split 泄漏
- **AND** 文档 MUST 不把旧数值用于方法优劣结论

#### Scenario: Temporal evidence 重新晋级
- **WHEN** 新 H5/P1 结果请求升级为 local strict-validation 或 reviewed claim
- **THEN** provenance MUST 包含 group-safe split artifact、sample/frame identity audit、独立 validation、final test、seed 和 normalization fingerprint
- **AND** 任一字段缺失时 status MUST 保持 pending、unverified 或 not_comparable

### Requirement: Claim registry schema 与外键必须可验证
claim registry MUST 使用固定结构化字段记录 claim id、subject/method、dataset/split、metric/value、status、provenance、caveat、seed_count、baseline、mean/std 或 CI、comparability、stress status、candidate flag 和 upgrade gate。主线 catalog 中每个非空 claim id MUST 唯一引用 registry 中存在的 claim。

#### Scenario: Pending claim 缺少数值
- **WHEN** claim status 为 pending、unverified、not_comparable 或 invalidated
- **THEN** metric value 和统计摘要 MAY 为空
- **AND** provenance、caveat、comparability 和 upgrade gate MUST 解释缺失或 blocker

#### Scenario: Reviewed claim 字段完整
- **WHEN** claim status 请求进入 reviewed paper main table
- **THEN** method、dataset/split、metric/value、seed_count、baseline、统计摘要、comparability、stress、provenance 和 caveat MUST 全部非空且合法

#### Scenario: Catalog claim 外键断裂
- **WHEN** mainline model catalog 引用 registry 不存在或重复的 claim id
- **THEN** architecture test MUST 失败并报告 model id 与 claim id

### Requirement: 仓库主线与数据 campaign 术语必须分离
current README、navigation、catalog、matrix、protocol 和 claim docs MUST 将 final C2 / U-MaskBeamJEPA 描述为仓库默认模型/研究主线，并将 MMW/CSI 描述为 current supporting dataset workflow。MMW MAY 被描述为当前数据实验 campaign，但 MUST NOT 被描述为已替换默认主线，除非 active transition change 同步全部 current contracts。

#### Scenario: Current 文档描述 MMW
- **WHEN** current 文档列出 MMW all-weather、Town GPS 或 BPA/CMA workflow
- **THEN** 文档 MUST 标明 supporting dataset/campaign、入口、输出边界和 claim status
- **AND** final C2 默认主线表述 MUST 保持一致

