## ADDED Requirements

### Requirement: 场景条件化多偏移模型契约
系统 MUST 提供一个 opt-in 的场景条件化多偏移 beam prediction 模型能力。该模型 MUST 默认以 `overlap_k16_s8_stage1` 作为 canonical visual/JEPA 基底，在 canonical predictor 基础上计算 `canonical_logits`，再按配置启用 geo、image、fusion、align、radio、object 和 beam-logit offset/adaptation 组件，并 MUST 返回可被 `adapt_model_output` 消费的 `logits` 主输出。

#### Scenario: 默认基底为 overlap k16 s8
- **WHEN** 用户加载 scene meta-offset base、lowmem 或真实数据 recipe 且未显式覆盖 canonical base variant
- **THEN** 配置 MUST 构建 `overlap_k16_s8_stage1` canonical predictor
- **AND** visual tokenizer MUST 解析为 `overlap_patch`、`kernel_size=16`、`stride=8`、`max_tokens=729`
- **AND** pooler MUST 默认为 GPS-query attention frame-level 输出
- **AND** metadata MUST 记录 `base_variant=overlap_k16_s8_stage1`

#### Scenario: hyper all heads forward 输出
- **WHEN** 配置构建启用全部 offset heads 的 scene-conditioned 模型并对 synthetic query batch 执行 forward
- **THEN** 输出 MUST 包含 `logits`、`canonical_logits`、`offsets`、`scene_embedding` 和 `debug`
- **AND** `logits` 与 `canonical_logits` 的 beam class 维 MUST 等于配置的 `num_classes`
- **AND** `offsets` MUST 至少包含启用的 geo、image、fusion、align、radio、object 和 beam-logit 诊断键

#### Scenario: global baseline 不启用 offset
- **WHEN** 配置选择 global/no-adaptation baseline
- **THEN** 模型 MUST 只使用以 `overlap_k16_s8_stage1` 为默认基底的 shared canonical predictor 产生 beam logits
- **AND** 输出 metadata MUST 记录 scene conditioning、hypernetwork 和 offset heads 均未启用
- **AND** 若配置显式选择 patch16 mean、GPS-biased、ResNet+GPS 或 GPS-only control，metadata MUST 标记为 control/fallback 而非默认基底

### Requirement: Scene encoder 与 support set encoder
系统 MUST 支持从 explicit `scene_params`、support set 输入、support labels、unlabeled support 和 scene-id ablation 生成 scene embedding。主方法 MUST 能在没有 scene_id embedding 的情况下依赖 scene params 和 support set 推断场景 embedding。

#### Scenario: zero-shot scene params
- **WHEN** `meta.k_shot=0` 且配置启用 scene params only
- **THEN** scene encoder MUST 只消费 query/support 可用的 `scene_params` 和 domain metadata
- **AND** 训练或评估 MUST 不读取 target query beam label 来生成 scene embedding

#### Scenario: labeled few-shot support
- **WHEN** episode 提供 K-shot labeled support set
- **THEN** support encoder MUST 能将 support 输入特征与允许的 beam label embedding 组合成 scene embedding
- **AND** metadata MUST 记录 support labels 被用于 adaptation 且来源仅限 support subset

#### Scenario: scene-id embedding 仅作 baseline
- **WHEN** 配置选择 scene-id embedding baseline
- **THEN** run metadata MUST 标记该结果只代表 seen-scene 或显式 scene-id ablation
- **AND** summary MUST NOT 将其声明为 unseen-scene 泛化主方法

### Requirement: 多层级 offset heads
系统 MUST 将不同类型的 scene shift 分配给不同层级的 offset/adaptation 组件。BeamLogitOffsetHead MUST 只作为最后层轻量残差，MUST NOT 替代 image、fusion、align、radio、object 或 geo 层级校准。

#### Scenario: image offset 作用于视觉特征
- **WHEN** 配置启用 ImageOffsetHead 且输入包含 image modality
- **THEN** image offset MUST 在 image encoder feature、projection feature 或明确声明的 image adapter 位置生效
- **AND** 测试 MUST 能证明关闭 ImageOffsetHead 会改变 image feature 或 adapter 参数路径，而不只是改变最后 logits bias

#### Scenario: fusion offset 输出模态权重
- **WHEN** 配置启用 FusionOffsetHead
- **THEN** fusion offset MUST 输出 modality gate、reliability weight 或 fused feature residual
- **AND** debug metadata MUST 记录每个启用 sensing modality 的平均 gate 或权重统计

#### Scenario: radio offset 是 sample-dependent
- **WHEN** 配置启用 RadioOffsetHead
- **THEN** radio residual MUST 依赖 query sample feature 与 scene embedding
- **AND** 同一 scene 中不同 query sample 的 radio residual MUST 不被实现为固定场景常数

### Requirement: Hierarchical hypernetwork 小参数生成
系统 MUST 提供 hierarchical hypernetwork，用 scene embedding 生成 geo、image、fusion、align、radio、object 和 beam-logit 子模块所需的小参数。Hypernetwork MUST NOT 生成完整 backbone 权重。

#### Scenario: 生成小参数字典
- **WHEN** hypernetwork 接收 `[B, scene_dim]` scene embedding
- **THEN** 输出 MUST 是按子模块分组的参数字典
- **AND** 参数字典 MUST 只包含 FiLM、adapter、LoRA、gate、bias、low-rank residual 或等价小参数

#### Scenario: 拒绝完整 backbone 参数生成
- **WHEN** 配置请求 hypernetwork 生成完整 image backbone、JEPA backbone、LiDAR backbone 或 fusion backbone 权重
- **THEN** 构建 MUST 失败
- **AND** 错误信息 MUST 指向支持的小参数生成路径

### Requirement: Meta 与 few-shot adaptation
系统 MUST 支持 none、maml、fomaml、anil、hyper 和 hyper_maml 方法。Inner-loop adaptation MUST 根据 `meta.adapt_modules` 白名单限制可更新模块，并 MUST 在 query set 上计算 outer loss 与 metrics。

#### Scenario: FOMAML offset heads only
- **WHEN** 配置选择 `meta.method=fomaml` 且 `meta.adapt_modules=["offset_heads"]`
- **THEN** inner-loop MUST 只更新 offset heads 的 fast weights
- **AND** backbone 与 canonical encoder 参数 MUST 保持未被 inner-loop 更新

#### Scenario: ANIL 只更新 head
- **WHEN** 配置选择 `meta.method=anil`
- **THEN** inner-loop MUST 冻结 backbone、scene encoder 和 fusion encoder
- **AND** 只允许 beam head 或配置允许的 offset head 参数参与 adaptation

#### Scenario: hypernetwork plus gradient adaptation
- **WHEN** 配置选择 `meta.method=hyper_maml`
- **THEN** 系统 MUST 先用 hypernetwork 生成场景特定初始化或小参数
- **AND** 再在 support set 上执行配置步数的 gradient adaptation
- **AND** query metrics MUST 与 support loss 分开记录

### Requirement: Synthetic scenario-hyperbeam sanity
系统 MUST 提供 synthetic scenario-hyperbeam 数据能力，使无真实数据环境也能跑通 dataset、episode、model、loss、training、evaluation 和 logging smoke。Synthetic generator MUST 支持可控 scene shift，用于验证不同 offset heads 的贡献。

#### Scenario: head-specific synthetic shift
- **WHEN** synthetic 配置选择 `shift_mode=geo_only`、`fusion_only`、`image_only`、`radio_only` 或等价模式
- **THEN** 生成样本 MUST 包含与对应 offset head 相关的可学习 shift
- **AND** sanity/test MUST 能在关闭对应 head 时观察 loss、metric 或 debug contribution 的变化

#### Scenario: 无真实 dataset 运行 sanity
- **WHEN** 用户运行 scenario meta-offset sanity 命令或等价 smoke test
- **THEN** 系统 MUST 使用 synthetic dataset 完成 global、hyper_all_heads、maml_offset_heads_only 和 hyper_maml 的最小训练/评估流程
- **AND** 该流程 MUST 不读取真实 `dataset/`、不写入源码控制的 checkpoint 或报告

### Requirement: 防止 target oracle 泄漏
系统 MUST 阻止真实 AoA/AoD、CSI/channel、path gain、真实 beam power vector、target_test label 和 target_unlabeled sensitive targets 作为模型输入或 adaptation 选择依据。Angle、beam power、LOS/path 等字段只能作为允许 split/subset 下的 auxiliary target、loss 或 offline diagnostic。

#### Scenario: query target oracle 输入被拒绝
- **WHEN** 模型 forward 输入包含真实 AoA/AoD、CSI/channel、path gain 或 query beam power vector 作为 sensing input
- **THEN** batch/runtime guard MUST 拒绝该输入
- **AND** 错误信息 MUST 包含字段名、split/subset 和可执行修复提示

#### Scenario: label budget zero 不读取 support label
- **WHEN** `meta.k_shot=0` 或 target adaptation `label_budget=0`
- **THEN** support/query adaptation loss MUST NOT 读取 beam label、beam power、path label 或 radio label 作为监督
- **AND** metadata MUST 记录对应 sensitive usage 字段为 false

### Requirement: 实验矩阵与配置生成
系统 MUST 用 base config + overrides 或等价 recipe 表达 scenario meta-offset 实验矩阵。源码 MUST 只保留少量 base/smoke/example 配置，完整矩阵 MUST 由 generator 输出到 ignored runtime artifact 目录或用户指定目录。

#### Scenario: 生成 80 类实验配置摘要
- **WHEN** 用户运行 scenario meta-offset matrix generator
- **THEN** generator MUST 生成覆盖 baseline、scene info、single/multi offset、adapter、radio、fusion、object、meta、few-shot、generalization、missing modality 和 loss ablation 的配置清单
- **AND** 每个生成配置 MUST 记录 base config、overrides、seed、split protocol 和 output boundary

#### Scenario: 不接管任意缺失 YAML
- **WHEN** 用户加载未声明 recipe 的缺失 scenario meta-offset 配置路径
- **THEN** 配置加载 MUST 抛出清晰缺失文件或未知 recipe 错误
- **AND** 系统 MUST NOT 将任意缺失 YAML 自动解释为可生成配置

### Requirement: 评估、诊断与报告产物
系统 MUST 在评估和诊断中记录 top-k、beam distance、DBA、per-scene/town/weather metrics、few-shot curve、offset magnitude、gate statistics 和 ablation contribution。所有报告产物 MUST 写入 ignored output boundary，并 MUST 保留 resolved config 与 provenance。

#### Scenario: offset contribution report
- **WHEN** 用户对训练完成的 scene meta-offset 模型运行 evaluation
- **THEN** evaluation MUST 能报告 canonical only、canonical + 单个 offset、canonical + offset 子集和 all heads 的指标
- **AND** report MUST 包含每个 offset 的 norm/gate/diagnostic 统计或 unavailable reason

#### Scenario: few-shot curve
- **WHEN** evaluation 配置包含 K-shot sweep `0,1,5,10,20`
- **THEN** evaluator MUST 为每个 K 记录 query metrics、support label usage、seed 和 split artifact provenance
- **AND** target_test 样本 MUST 只参与最终评价
