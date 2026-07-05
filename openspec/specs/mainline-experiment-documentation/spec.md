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

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态 beam prediction、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、GPS v2/adapter、MMW Town GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理和诊断，而不是 KD-first、HiST-Beam-first、Raymobtime-first、Top8/residual-first、BGAM-first、viewer-first 或 GPS coarse-anchor-first 工作流。历史 KD、Hist、Raymobtime、Top8 selector、residual、camera residual、BGAM、viewer manifest 或 GPS coarse anchor 背景可以保留在 archive 或历史说明中，但必须标记为已退役或历史记录。

#### Scenario: pyproject 描述不再 KD Hist 或退役路线 first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation、HiST-Beam、Top8 selector、residual 或 GPS coarse anchor 描述为当前唯一或首要工作流
- **AND** 若提到这些路线，MUST 表达其为 legacy、historical 或 retired

#### Scenario: 文档保留历史说明
- **WHEN** README 或 docs 提到历史 KD、Hist、Top8 selector、residual、camera residual 或 GPS coarse anchor 代码
- **THEN** 文档 MUST 说明对应能力已从当前 active mainline 退役
- **AND** 文档 MUST 不提供当前推荐运行命令

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理和当前诊断。KD baseline、HiST-Beam/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS coarse anchor、residual fusion、camera residual、BGAM、viewer manifest、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、supporting、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、Raymobtime s008、retired Top8 selector/residual/GPS coarse anchor 命令或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、Top8 selector、residual、camera residual、GPS coarse anchor、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/JEPA/CSI 主线

#### Scenario: 当前 workflow 文档声明运行状态
- **WHEN** 文档列出当前实验配置、benchmark manifest 或诊断配置
- **THEN** 文档 MUST 标明该条目是 formal、lowmem、smoke、debug、evaluation-only、upper-bound、historical ablation 还是 mock
- **AND** upper-bound、mock、smoke 或 historical ablation MUST 不得被写成正式结论

### Requirement: 健康检查反映保留入口
快速健康检查 MUST 覆盖当前仍支持的架构边界、包内 CLI、JEPA visual analysis、GPS shortcut benchmark、文档健康和当前主线 focused tests。健康检查 MUST 不要求 Raymobtime s008、已退役的模态失衡诊断脚本、fusion KD virtual alias、BGAM、viewer manifest 或 HiST-Beam/Hist CLI 可用。

#### Scenario: focused validation 不依赖退役入口
- **WHEN** 开发者执行本 change 的 focused 验证
- **THEN** 验证命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 不包含已退役的 Hist CLI、Hist configs 或独立模态诊断脚本
- **AND** 验证 MUST 覆盖配置加载失败、架构边界、registry 和保留 evaluation subset 能力

### Requirement: 当前支持面收敛到 Image+GPS JEPA query-pool
项目 MUST 将当前推荐训练、评估、诊断和实验配置支持面收敛到 Image+GPS JEPA query-pool 主线及其必要对照。保留面 MUST 包含 `jepa_context_image + GPSQueryPool` JEPA downstream、`fair_gps_biased` paired baseline、supervised/random-best 控制组、vision-position baseline suite 和 `jepa_visual_analysis` 论文图/诊断出口。退役路线 MUST 不再作为 README 推荐入口、pyproject console script、架构 allowlist 或当前配置矩阵出现。

#### Scenario: README 展示当前主线
- **WHEN** 开发者阅读 README 的项目定位、主要入口和配置矩阵
- **THEN** 文档 MUST 把 Image+GPS JEPA query-pool、paired baseline/control、vision-position baseline suite 和 JEPA visual analysis 描述为当前主线
- **AND** 文档 MUST 不把 GPS window、DeepVerse/DT31、旧静态 modality visualization 或仓库级 Gradio viewer support 描述为当前入口
- **AND** 文档 MAY 继续保留 BeamBench/Arnold22 Camera AE+GPS Direct 当前入口和复现辅助说明

#### Scenario: 架构测试拒绝退役入口回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 拒绝退役的 viewer support、viewer manifest、BGAM、GPS window baseline、DeepVerse/DT31 workflow、Top8 selector dataset 和旧静态 modality visualization 文件重新出现在当前 allowlist 中
- **AND** 测试 MUST 继续允许 JEPA query-pool、paired control、vision-position baseline、BeamBench/Arnold22 Camera AE+GPS Direct 和 JEPA visual analysis 相关入口

#### Scenario: 配置矩阵只保留必要 JEPA 对照
- **WHEN** 开发者查看 `configs/fusion/experiments/jepa_image_gps/` 和实验矩阵文档
- **THEN** 当前配置 MUST 保留 query-pool、GPS-biased baseline、supervised baseline 和 random-best 控制组
- **AND** scene31-only、非 BeamBench 的 last-checkpoint 和 next-beam downstream ablation 配置 MUST 不再作为当前配置文件维护
- **AND** `beambench_fair` 相关配置 MAY 继续保留用于 Arnold22/BeamBench 口径对照

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
主线实验文档 MUST 索引 paper artifact export、dataset audit 和 literature matrix，并说明它们的 claim 状态边界。

#### Scenario: 文档索引 paper export
- **WHEN** README 或 `docs/experiment_matrix.md` 新增 paper export 说明
- **THEN** 文档 MUST 说明 export 消费已审阅 claim、ledger 或 summary
- **AND** 文档 MUST 说明 pending/mock/historical rows 默认不进入 main table

#### Scenario: 文档索引 dataset audit
- **WHEN** README_REPRODUCE 或主线文档给出数据审计入口
- **THEN** 文档 MUST 使用当前存在的 audit entrypoint
- **AND** 文档 MUST 说明 audit 只读、不移动数据、不代表 official reproduction 已完成

#### Scenario: inventory 记录 literature matrix
- **WHEN** 新增 `docs/literature_matrix.md` 或 `paper/references.bib`
- **THEN** `docs/project_surface_inventory.md` MUST 记录其文档生命周期和职责
- **AND** 文档 MUST 不把本地 PDF 或外部论文下载物纳入源码产物要求

### Requirement: Harvested claim draft governance
主线实验文档 MUST 区分 harvested claim draft 和正式 claim registry。自动生成的 candidate、dashboard summary 或 ledger record MUST 不被描述为已审阅结论。

#### Scenario: candidate 不自动进入 claim registry
- **WHEN** harvester 输出 claim candidate
- **THEN** `docs/result_claims_registry.md` MUST 只有在人工审阅后才新增或更新对应 claim 行
- **AND** candidate 输出 MUST 保留 `draft`、`candidate_only` 或等价状态标记

#### Scenario: README 和实验矩阵引用 dashboard
- **WHEN** 文档新增 research dashboard 或 harvester 入口说明
- **THEN** README 或 `docs/experiment_matrix.md` MAY 指向该入口作为本地研究辅助工具
- **AND** 文档 MUST 说明它不生成正式论文结论、不移动产物、不替代 claim registry

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

