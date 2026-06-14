# cross-scene-loso-workflow Specification

## Purpose
定义跨场景 leave-one-scene-out 的通用 fold 规划、target adapt/test 防泄漏拆分、few-shot 采样和执行汇总支撑契约。该 capability 只作为 supporting helper 保留；DeepSense6G HiST-Beam 默认矩阵和 `kd-sensing-hist-beam-loso` 执行入口已退役，不属于当前推荐 workflow。
## Requirements
### Requirement: DeepSense6G 31-34 LOSO fold 定义
系统 MUST 提供 DeepSense6G scenarios 31、32、33、34 的 leave-one-scene-out fold 定义。每个 fold MUST 包含三个 source scenes 和一个 target scene，并 MUST 可通过配置或 orchestration 入口选择单个 fold 或全部 fold。

#### Scenario: 生成四个 LOSO fold
- **WHEN** 用户请求 DeepSense6G 31-34 的默认 LOSO folds
- **THEN** 系统 MUST 生成 target scene 34、33、32、31 各一次的四个 fold
- **AND** 每个 fold 的 source scenes MUST 等于其余三个场景

#### Scenario: 选择单个 target scene
- **WHEN** 用户配置 `target_scene: 34`
- **THEN** 系统 MUST 将 source scenes 解析为 `[31, 32, 33]`，除非用户显式覆盖 source scenes
- **AND** fold metadata MUST 记录 target scene、source scenes 和 fold id

#### Scenario: 拒绝 source target 重叠
- **WHEN** 用户显式配置的 source scenes 包含 target scene
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出 source/target scene 不得重叠

### Requirement: Target adapt/test split 防泄漏
系统 MUST 将每个 target scene 的可用 target split 确定性拆分为 `target_adapt` 和 `target_test`。默认比例 MUST 为 20% target_adapt 和 80% target_test。`target_test` MUST 只用于最终测试，不得参与训练、adaptation、early stopping、threshold selection、prototype selection 或 normalizer/scaler fit。

#### Scenario: target split 可复现
- **WHEN** 用户使用相同 target scene、split seed 和 split 配置构建 target adapt/test split
- **THEN** 系统 MUST 产生相同的 target_adapt 和 target_test 样本集合
- **AND** split metadata MUST 记录 split seed、比例、样本数和样本选择口径

#### Scenario: target_test 不参与 adaptation
- **WHEN** 用户运行 source training 或 target adaptation
- **THEN** 系统 MUST 不把 target_test 样本放入训练或 adaptation dataloader
- **AND** 系统 MUST 不使用 target_test label 计算 supervised loss、prototype target 或 early stopping 指标

#### Scenario: adapt/test 样本无交集
- **WHEN** target adapt/test split 构建完成
- **THEN** `target_adapt` 和 `target_test` 的 sample id MUST 无交集
- **AND** 若数据包含 `seq_index`，系统 MUST 优先保证二者的 `seq_index` 无交集

#### Scenario: split metadata 写入运行产物
- **WHEN** LOSO source training、adaptation 或 evaluation 创建运行目录
- **THEN** 系统 MUST 保存 fold metadata 和 target split metadata
- **AND** metadata MUST 足以复现实验使用的 source、target_adapt 和 target_test 样本集合

### Requirement: Source multi-scene 数据加载
系统 MUST 能为 LOSO source training 构建由多个 DeepSense6G scenes 组成的训练数据。每个底层 scene dataset MUST 继续遵守现有模态按需读取、场景 metadata 和本地产物边界。

#### Scenario: 构建三 source scene 训练集
- **WHEN** fold 的 source scenes 为 `[31, 32, 33]`
- **THEN** source train dataloader MUST 覆盖三个 scene 的训练样本
- **AND** batch 或 sample metadata MUST 保留每个样本的 scene id

#### Scenario: normalizer 只从允许训练数据拟合
- **WHEN** source multi-scene dataloader 需要 GPS、LiDAR、mmWave、CSI 或其它 normalizer/scaler artifact
- **THEN** 系统 MUST 只从 source train 或配置允许的训练 split 拟合 artifact
- **AND** 系统 MUST 将 artifact 复用于 target_adapt 和 target_test 读取

#### Scenario: 未启用模态不读取对应资源
- **WHEN** LOSO 配置只启用 `image`、`radar`、`gps`
- **THEN** dataset MUST 不读取 LiDAR、mmWave 或 CSI 资源
- **AND** 缺失未启用模态文件不得阻止该 LOSO run

### Requirement: Few-shot target label budget 采样
系统 MUST 支持 target label budgets `0`、`5`、`10`、`20`、`50`。当 budget 大于 0 时，系统 MUST 从 `target_adapt` 中选择 labeled subset，并 MUST 保留其余 target_adapt 样本作为 unlabeled subset。默认采样 MUST 优先 coarse group stratified sampling。

#### Scenario: label_budget 为 0
- **WHEN** 用户配置 `label_budget: 0`
- **THEN** 系统 MUST 不产生 labeled target subset
- **AND** target_adapt 样本 MUST 只作为 unlabeled adaptation 数据使用

#### Scenario: coarse group 分层采样
- **WHEN** `label_budget` 大于 0 且 target_adapt 中存在多个 coarse group
- **THEN** 系统 MUST 优先选择覆盖不同 coarse group 的 labeled samples
- **AND** labeled sampling manifest MUST 记录每个 labeled sample 的 beam 和 coarse group

#### Scenario: 样本不足时退化
- **WHEN** target_adapt 可用样本数小于请求的 label budget
- **THEN** 系统 MUST 使用全部可用 target_adapt 样本作为 labeled subset
- **AND** sampling metadata MUST 记录 requested budget、actual labeled count 和退化原因

#### Scenario: sampling seed 可复现
- **WHEN** 用户使用相同 fold、budget 和 seed 重复采样
- **THEN** 系统 MUST 产生相同 labeled sample id 集合
- **AND** 不同 seed MAY 产生不同 labeled sample id 集合

### Requirement: LOSO execute preflight
系统 MUST 在启动训练或 adaptation 前执行 preflight。preflight MUST 检查矩阵中涉及的 DeepSense6G scene 数据根目录、CSV、启用模态资源、输出目录写权限和配置合法性。preflight 失败时 MUST 给出明确错误，不得写出表示成功的 summary。

#### Scenario: 无数据时明确失败
- **WHEN** execute 配置引用的 source scene 或 target scene 缺少数据根目录、必要 CSV 或启用模态资源
- **THEN** preflight MUST 失败
- **AND** 错误信息 MUST 包含缺失的 scene、资源类型和路径
- **AND** 系统 MUST 不启动 source training 或 target adaptation

#### Scenario: 输出目录不可写时明确失败
- **WHEN** execute 输出目录不存在且无法创建，或已存在但不可写
- **THEN** preflight MUST 失败
- **AND** 错误信息 MUST 指出不可写的输出目录

#### Scenario: preflight 成功后记录检查结果
- **WHEN** preflight 通过
- **THEN** 系统 MUST 保存 preflight metadata
- **AND** metadata MUST 记录检查过的 scenes、CSV、启用模态、输出目录和 quick validation matrix 摘要

### Requirement: 数据集无关的 LOSO fold 规划
LOSO workflow MUST 在现有 DeepSense6G 31-34 folds 之外支持数据集无关的 fold 规划。对于 MMW，planner MUST 使用 dataset descriptor 和数据可用性 metadata 生成 scenario-level、town-level 或 condition-level source/target folds。

#### Scenario: 生成 MMW scenario fold
- **WHEN** MMW data availability metadata 在请求范围内包含至少两个 ready scenarios
- **THEN** planner MUST 为每个 target scenario 生成一个 fold
- **AND** 每个 fold MUST 记录 dataset family `MMW`、condition、town、target scenario、source scenarios 和 fold id

#### Scenario: 保留 DeepSense6G 默认 folds
- **WHEN** 用户请求现有 DeepSense6G 31-34 LOSO workflow
- **THEN** planner MUST 继续生成四个现有 DeepSense6G folds
- **AND** MMW-specific metadata MUST NOT be required

### Requirement: Single-scene smoke is not LOSO
LOSO workflow MUST 区分 single-scene smoke runs 和 cross-scene adaptation runs。单个 ready MMW scenario MUST NOT 被报告为 LOSO、leave-one-scene-out、cross-town 或 cross-condition evidence。

#### Scenario: MMW 只有一个 ready scenario
- **WHEN** planner sees exactly one ready MMW scenario
- **THEN** planner MUST generate at most smoke or within-scenario sanity runs
- **AND** execution summary MUST mark `cross_scene_claim_allowed: false`

### Requirement: MMW target adapt/test no leakage
对于 MMW folds，target_adapt 和 target_test split MUST 确定性且无泄漏。Split metadata MUST 包含 sample ids、可用时的 sequence ids、scenario/town/condition、split seed、split ratio 和分布摘要。

#### Scenario: target split 无交集
- **WHEN** MMW target split is built
- **THEN** target_adapt and target_test sample ids MUST be disjoint
- **AND** if sequence segment ids are available, the two splits MUST also avoid segment overlap

#### Scenario: target_test 不参与 adaptation 决策
- **WHEN** MMW target adaptation runs
- **THEN** target_test samples MUST NOT be used for supervised loss, prototype selection, threshold tuning, normalizer fitting or early stopping
- **AND** run metadata MUST record this split boundary

### Requirement: MMW few-shot sampling strategy
LOSO workflow MUST 支持 MMW few-shot target sampling，budgets 为 `0`、`5`、`10`、`20` 和 `50`。Sampling MUST 优先覆盖 coarse sector 和 relative azimuth bins；只有当 bin 不可用或样本不足时，才退化为确定性随机采样。

#### Scenario: 分层采样成功
- **WHEN** target_adapt contains multiple coarse sectors and relative azimuth bins
- **THEN** sampler MUST select labeled samples to cover as many sector/bin combinations as possible within the budget
- **AND** sampling manifest MUST record sector/bin for every labeled sample

#### Scenario: 分层字段不可用
- **WHEN** relative azimuth or coarse sector is unavailable for target_adapt samples
- **THEN** sampler MUST fall back to deterministic random sampling
- **AND** sampling manifest MUST record the fallback reason

### Requirement: MMW LOSO summary claim guard
MMW LOSO summary MUST 包含机器可读 claim guard，用于说明某个 run 是否能支撑 cross-scene、cross-town 或 cross-condition 结论。

#### Scenario: summary 输出 claim guard
- **WHEN** MMW smoke, LOSO or adaptation execution completes
- **THEN** summary MUST include `claim_scope` and `cross_scene_claim_allowed`
- **AND** incomplete or single-scene runs MUST set `cross_scene_claim_allowed` to false

### Requirement: Radio-semantic few-shot sampling
LOSO workflow MUST support radio-semantic-aware target labeled subset sampling. When radio-semantic labels are legally available for target_adapt labeled sampling, sampler MUST prioritize radio-semantic stratification, then coarse sector and relative azimuth bin stratification, then deterministic random fallback.

#### Scenario: radio-semantic 分层采样
- **WHEN** `label_budget` 大于 0 且 target_adapt 样本包含合法 `radio_semantic_label`
- **THEN** sampler MUST 优先选择覆盖不同 radio-semantic classes 的 labeled samples
- **AND** sampling manifest MUST 记录每个 labeled sample 的 radio label、beam、coarse sector、relative azimuth bin、seed 和 label source

#### Scenario: radio label 不可用时退化
- **WHEN** target_adapt 缺少合法 radio labels 但存在 coarse sector 或 relative azimuth bin
- **THEN** sampler MUST 退化为 coarse/azimuth 分层采样
- **AND** sampling metadata MUST 记录 radio stratification unavailable reason

### Requirement: Radio-semantic target 防泄漏
LOSO execute MUST enforce radio-semantic leakage boundaries during target adaptation. For `label_budget=0` and unlabeled target_adapt batches, target beam labels, beam_power, q_power and radio_semantic_label MUST NOT be used as supervised training targets or prototype labels.

#### Scenario: 0-label run 记录未使用 target radio label
- **WHEN** LOSO runner 执行 radio-semantic variant 且 `label_budget=0`
- **THEN** adaptation metadata MUST 记录 `used_target_labels=false`
- **AND** metadata MUST 记录 `used_target_beam_power_for_training=false`
- **AND** metadata MUST 记录 `used_target_radio_label_for_training=false`

#### Scenario: target_test 不参与 radio prototype 更新
- **WHEN** target_test evaluation 包含 beam_power 或 radio labels
- **THEN** runner MUST 只将这些字段用于离线 metrics
- **AND** runner MUST NOT 使用 target_test 字段更新 radio prototypes、target-private prototypes、confidence threshold 或 early stopping

### Requirement: Radio-semantic quick validation conclusion
LOSO summary and quick validation conclusion MUST compare coarse prototype and radio-semantic prototype variants with enough diagnostics to judge whether radio semantics contributed beyond adapter-only and coarse prototype baselines.

#### Scenario: V5 vs V6 对比
- **WHEN** 同一 fold、budget 和 seed 下存在 V5 coarse prototype 与 V6 radio prototype metrics
- **THEN** conclusion MUST 比较 Top-1/3/5、coarse accuracy、radio accuracy、power metrics、prototype coverage、trainable ratio 和 adaptation time
- **AND** conclusion MUST 标明 radio prototype 是否优于 coarse prototype

#### Scenario: radio condition off/on 对比
- **WHEN** 同一 fold、budget 和 seed 下存在 radio condition off 与 on 的 V6 runs
- **THEN** conclusion MUST 比较 beam metrics 与 radio assignment diagnostics
- **AND** 若 on/off prediction 完全一致，conclusion MUST 记录 `radio_condition_prediction_delta=0` 或等价诊断

#### Scenario: 缺失 radio 指标时不可判定
- **WHEN** 生成 radio conclusion 所需的 radio label、beam_power、prototype artifact 或 metrics 缺失
- **THEN** conclusion MUST 将对应比较标记为 `inconclusive`
- **AND** conclusion MUST 记录缺失字段和 run path

### Requirement: LOSO 不再绑定 Hist 默认矩阵
当前项目 MUST 不再提供 HiST-Beam 默认 LOSO 矩阵或 `kd-sensing-hist-beam-loso` 执行入口。未来若需要跨场景矩阵，MUST 由当前保留 workflow 通过新的 spec 明确定义配置、CLI、输出和防泄漏边界。

#### Scenario: Hist LOSO 入口不可用
- **WHEN** 用户尝试运行 `kd-sensing-hist-beam-loso`
- **THEN** 系统 MUST 不把该命令作为当前支持入口
- **AND** README 和健康检查 MUST 不要求该命令存在

#### Scenario: 当前 LOSO fold 定义可被未来 workflow 复用
- **WHEN** 未来非 Hist workflow 需要 leave-one-scene-out fold
- **THEN** 新 workflow MUST 显式声明自己的 runner、配置矩阵和输出契约
- **AND** 系统 MUST 不复用已退役 Hist run plan 作为隐式默认
