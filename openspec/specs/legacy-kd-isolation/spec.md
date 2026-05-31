# legacy-kd-isolation Specification

## Purpose
定义 legacy teacher-student KD 与当前 no-KD、HiST-Beam、MMW sensor-assisted、history-anchored residual 和 target adaptation 主线的隔离边界，确保历史 KD 仅作为显式选择的复现或补充 baseline。
## Requirements
### Requirement: Legacy KD 隔离边界
系统 MUST 将 teacher-student 知识蒸馏运行时与当前 active mainline 方法隔离。默认训练、评估、HiST-Beam LOSO、history-anchored residual、adapter/prototype/calibration 和 quick validation 路径 MUST 不构建 frozen teacher、不解析 teacher checkpoint、不计算 KD loss，除非用户显式选择 legacy KD 或 optional KD baseline。

#### Scenario: 默认主线不加载 teacher runtime
- **WHEN** 用户运行当前推荐的 no-KD、HiST-Beam、MMW sensor-assisted、history-anchored residual 或 target adaptation 配置
- **THEN** 系统 MUST 不构建 frozen teacher model
- **AND** 系统 MUST 不读取 `teacher_model_name` 或 teacher checkpoint 作为训练必要输入
- **AND** run metadata MUST 记录 `distillation_enabled=false`

#### Scenario: 显式 legacy KD 才启用蒸馏
- **WHEN** 用户运行明确标注为 legacy KD 或 KD baseline 的 `logits_kd`、`rkd` 或等价配置
- **THEN** 系统 MAY 构建 frozen teacher 和可训练 student
- **AND** 系统 MUST 写出 `method_family=legacy_kd` 或等价 metadata
- **AND** 系统 MUST 记录 distillation type、teacher checkpoint 来源和 student model 类型

### Requirement: KD baseline 不参与默认主结论
legacy KD 或 optional KD baseline 的结果 MUST 与当前 few-shot cross-scene 主线 summary 分组隔离。除非某个后续 OpenSpec change 明确将具体 KD 方法纳入主方法，KD run MUST 默认不作为 main conclusion improvement 的证据。

#### Scenario: KD run 默认不可用于主结论
- **WHEN** summary 读取一个 `method_family=legacy_kd` 或 `distillation_enabled=true` 的 run
- **THEN** summary MUST 将该 run 标记为 `main_conclusion_eligible=false`
- **AND** quick validation conclusion MUST 不使用该 run 证明主方法优于 source-only、adapter、prototype 或 calibration baseline

#### Scenario: KD baseline 独立汇总
- **WHEN** 同一 fold、budget、seed 下同时存在 no-KD mainline run 和 legacy KD baseline run
- **THEN** LOSO summary MUST 能按 method family 分组
- **AND** KD baseline 指标 MUST 保留为 supplemental 或 baseline comparison
- **AND** summary MUST 不把 KD improvement 混入默认 mainline 排名

### Requirement: KD 历史代码保留策略
系统 MUST 支持保留历史 KD 代码、配置和测试用于复现，但保留位置和命名 MUST 明确表达 legacy、baseline 或 optional 身份。历史 KD 代码不得作为新主线方法的隐式依赖。

#### Scenario: 历史 KD 配置可追溯
- **WHEN** 仓库保留 `logits_kd`、`rkd` 或 teacher-student 配置
- **THEN** 配置文件、配置 metadata 或相邻文档 MUST 标明该配置属于 legacy KD 或 optional baseline
- **AND** 配置 MUST 不作为 README quickstart 或当前主线 quick validation 的默认推荐入口

#### Scenario: active code path 不依赖 legacy KD 聚合入口
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 拒绝默认 mainline、HiST-Beam、history-anchored residual 或 adapter/prototype 代码新增对 legacy KD runtime 聚合入口的依赖
- **AND** 测试 MAY 允许 legacy KD baseline 自身导入 KD 算法模块

### Requirement: KD 运行时职责不得扩散
KD 算法模块 MUST 专注于张量级 logits/feature/relation distillation 和 schedule 计算。teacher checkpoint 解析、模型构建、dataset 构建、device 选择和 batch 输入准备 MUST 位于 engine runtime 或 legacy baseline adapter 中，并且不得成为 active mainline 的必要依赖。

#### Scenario: 导入 KD 算法不构建运行对象
- **WHEN** 开发者导入 KD loss 或 distillation schedule 工具
- **THEN** 导入 MUST 不构建 model、dataset、optimizer 或 checkpoint registry
- **AND** 导入 MUST 不读取本地 checkpoint、配置文件或数据集

#### Scenario: 新方法不扩写 KD runtime
- **WHEN** 开发者新增 HiST-Beam residual、prototype、calibration 或其它 no-KD adaptation 方法
- **THEN** 主要实现 MUST 位于对应方法模块、loss/objective 或 engine extension 中
- **AND** 实现 MUST 不要求修改 legacy KD teacher-student forward 逻辑

### Requirement: KD 可选增强必须重新提案
LLM teacher 蒸馏、privileged modality distillation、self-distillation regularizer 或其它新 KD 增强若要进入当前研究主线，MUST 通过独立 OpenSpec change 定义目标、数据契约、防泄漏边界、baseline 和 summary eligibility。

#### Scenario: 新 KD 增强不能静默加入主线
- **WHEN** 开发者新增一个非历史 KD 方法并希望参与主结论
- **THEN** 该方法 MUST 有对应 OpenSpec proposal/spec/tasks
- **AND** spec MUST 明确 teacher 信息来源、target label/power 使用边界和 `main_conclusion_eligible` 条件

### Requirement: Legacy KD virtual 配置入口收窄
系统 MUST 将 legacy KD baseline 从 canonical fusion virtual config 主入口中收窄出去。配置加载器 MUST 不再为任意 fusion 模态 slug 自动生成 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml`；显式保留的 legacy KD 实体配置或后续单独 baseline change MAY 继续运行，但 MUST 标记为 legacy/optional/supplemental。

#### Scenario: fusion KD virtual alias 被拒绝
- **WHEN** 用户请求不存在于磁盘的 `configs/fusion/<canonical_slug>_logits_kd.yaml` 或 `configs/fusion/<canonical_slug>_rkd.yaml`
- **THEN** 配置加载 MUST 失败并说明 legacy KD fusion virtual alias 已退役
- **AND** 错误信息 MUST 不静默回退为 no-KD 配置

#### Scenario: 显式 legacy KD 实体配置保留 lineage
- **WHEN** 用户运行仍被源码跟踪的 legacy KD 实体配置
- **THEN** run metadata MUST 记录 `method_family=legacy_kd`、`distillation_enabled=true` 和 `main_conclusion_eligible=false`
- **AND** summary MUST 将该 run 作为 supplemental 或 optional baseline，而不是 few-shot cross-scene 主结论证据

### Requirement: no-KD 配置不携带 KD-only 超参
当前 no-KD、HiST-Beam、MMW sensor-assisted、history-anchored residual 和 target adaptation 主线配置 MUST 不写入 `temperature`、`alpha`、`rkd_pairs_per_anchor`、`rkd_distance_weight` 或 `rkd_angle_weight` 等 KD-only 超参。运行时 MAY 继续接受这些字段用于历史配置，但新生成的 no-KD config 和推荐实体 YAML MUST 使用 supervised/adaptation 命名与最小 distillation 兼容字段。

#### Scenario: no-KD config 最小化
- **WHEN** 用户加载当前推荐 no-KD 或 HiST-Beam 主线配置
- **THEN** `distillation.type` MUST 为 `no_kd`
- **AND** 配置 MUST 不要求 KD temperature、alpha 或 RKD 权重字段存在
- **AND** run metadata MUST 记录 `distillation_enabled=false`

#### Scenario: legacy KD 字段不污染主线日志
- **WHEN** no-KD 主线训练写出 `final_config.yaml`、`resolved_config.yaml` 或 run metadata
- **THEN** 新写出的 artifact MUST 不把无 teacher 的 beam soft target 或 supervised/adaptation loss 记录为 `loss/distillation`
- **AND** KD-only 超参 MUST 不作为主线可调参数出现在推荐文档中

