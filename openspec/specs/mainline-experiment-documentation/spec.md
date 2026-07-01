# mainline-experiment-documentation Specification

## Purpose
定义当前主线模型目录、实验协议表、结果/claim 账本、baseline 报告分层和跨文档索引规则，使维护者能区分当前可引用事实、local substitute、blocked official、upper-bound、mock/smoke 和 historical ablation。
## Requirements
### Requirement: 主线模型目录
项目 MUST 维护当前主线模型目录，用于集中说明每条当前主线、baseline/control 和诊断 workflow 的研究问题、模型边界、配置入口、数据口径、指标口径、运行状态和结果引用。该目录 MUST 位于 `docs/mainline_model_catalog.md` 或等价 current 文档中，并 MUST 不把 retired、historical、supporting 或 mock-only workflow 描述为当前推荐入口。

#### Scenario: 主线模型目录覆盖当前支持面
- **WHEN** 开发者阅读主线模型目录
- **THEN** 文档 MUST 至少覆盖 Image+GPS JEPA GPS-biased reuse、GPS-query pooling、supervised/random controls、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct 本地 substitute、BEV-Fusion 2604、MMW GPS v2、CSI hardening、JEPA visual analysis 和 GPS shortcut benchmark
- **AND** 每行 MUST 标明对应 config、入口命令或诊断入口、数据集/场景、split/target、metric profile、运行状态和主要 caveat

#### Scenario: 退役路线不得进入 current 主线表
- **WHEN** 主线模型目录提到 KD、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D 或 Multimodal-NF
- **THEN** 对应行 MUST 标记为 retired、historical 或 migration guard
- **AND** 文档 MUST 不提供这些路线的当前推荐训练命令

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
项目 MUST 在 baseline 和复现报告中使用统一状态分类，将 current summary、official blocked、local substitute、strict-validation、upper-bound、mock/smoke 和 historical ablation 分开。历史流水账 MAY 保留，但 MUST 不覆盖 current summary。

#### Scenario: baseline current summary 位于历史流水账之前
- **WHEN** 开发者打开 `BASELINE_REPORT.md`、`README_REPRODUCE.md` 或 `results/reproduce_baseline.md`
- **THEN** 文档开头 MUST 指向当前推荐口径或 current summary
- **AND** 后续历史运行记录 MUST 标明其是否为 historical log、ablation、upper-bound、mock/smoke 或 blocked official reproduction

#### Scenario: 历史命令不可被误认为推荐入口
- **WHEN** 历史流水账包含旧 `future` target、`test_as_validation`、旧 AE 维度、旧校准角或其它已被替换的命令
- **THEN** 该段落 MUST 明确标记为历史或不可作为当前正式结果
- **AND** current summary MUST 给出当前推荐命令或指向当前推荐命令所在文档

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

### Requirement: RBMA ablation documentation
主线实验文档 MUST 记录 RBMA/prototype/KD missing-modality ablation 的 local/pending status、配置入口、推荐运行顺序、比较口径和 claim caveat。文档 MUST 不把未验证本地实验描述为 official reproduction 或已达成数值 claim。

#### Scenario: 文档记录推荐四配置
- **WHEN** RBMA ablation configs 加入仓库
- **THEN** `docs/experiment_matrix.md` 或等价 current 文档 MUST 记录首轮推荐运行 `amber_style_mask_baseline`、`no_jepa_rbma`、`no_jepa_rbma_proto` 和 `no_jepa_rbma_proto_kd`
- **AND** 文档 MUST 说明 `jepa_small_lambda_rbma_proto_kd` 是后续对照而非首轮必跑项

#### Scenario: claim registry 保持 pending/local
- **WHEN** 文档记录 RBMA workflow 结果入口或实验计划
- **THEN** `docs/result_claims_registry.md` 或等价 claim 账本 MUST 将其标记为 local/pending，直到真实评估结果和 provenance 完整
- **AND** 文档 MUST 不声称 AMBER official 数值复现已完成

#### Scenario: 实验协议记录 pattern 口径
- **WHEN** 文档描述 missing pattern evaluation
- **THEN** `docs/experiment_protocols.md` 或等价协议文档 MUST 记录 canonical 模态顺序、pattern definitions、pattern probabilities、hard-label metrics 和输出边界
- **AND** 文档 MUST 明确内部使用 `image` 而不是 `vision` 作为 canonical 模态名

