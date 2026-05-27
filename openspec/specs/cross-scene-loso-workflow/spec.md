# cross-scene-loso-workflow Specification

## Purpose
TBD - created by archiving change add-hist-beam-cross-scene-adaptation. Update Purpose after archive.
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

### Requirement: LOSO 运行编排入口
系统 MUST 提供配置驱动的包内 LOSO orchestration 入口，用于依次运行 source training、source-only target_test evaluation、target adaptation、adapted target_test evaluation 和结果汇总。入口 MUST 使用 `kd_sensing.cli` 或包内模块，不得新增长期维护的根目录脚本。

#### Scenario: 运行单 fold 单 variant
- **WHEN** 用户通过 LOSO 入口指定一个 target scene、一个 variant、一个 seed 和一个 budget
- **THEN** 系统 MUST 只运行对应 fold 和配置组合
- **AND** 输出目录 MUST 包含 source、adaptation、evaluation 和 summary metadata

#### Scenario: 运行默认快速矩阵
- **WHEN** 用户请求默认 HiST-Beam quick verification matrix
- **THEN** 系统 MUST 能遍历四个 LOSO folds、配置的 seeds、配置的 variants 和配置的 label budgets
- **AND** 系统 MUST 允许用户用配置缩小 variants、budgets 或 folds 以进行 smoke test

#### Scenario: 复用已有 source checkpoint
- **WHEN** 指定 fold 和 seed 的 source checkpoint 已存在且配置允许复用
- **THEN** orchestration MUST 能跳过 source retraining 并复用该 checkpoint
- **AND** summary metadata MUST 记录 checkpoint 来源和复用行为

#### Scenario: 不覆盖既有运行产物
- **WHEN** LOSO 入口创建输出目录且目标目录已存在
- **THEN** 系统 MUST 遵守现有输出覆盖和唯一目录规则
- **AND** 未显式 overwrite 时 MUST 不覆盖已有 metrics、checkpoint、predictions 或 prototype artifact

### Requirement: LOSO 结果汇总
系统 MUST 为 HiST-Beam 快速验证输出 source-only、few-shot adaptation 和 efficiency 三类汇总表或等价 JSON/CSV。汇总 MUST 能按 fold、target scene、variant、budget 和 seed 聚合，并 MUST 记录均值与可追溯的单次运行路径。

#### Scenario: 输出 source-only 表
- **WHEN** V0、V1、V2 和 V3 source-only evaluation 完成
- **THEN** 汇总 MUST 包含每个 target scene 的 Top-1、Top-3 和 coarse accuracy
- **AND** 汇总 MUST 包含跨 fold 平均指标

#### Scenario: 输出 few-shot adaptation 表
- **WHEN** source-only V3、full fine-tuning、adapter-only 和 adapter+prototype evaluation 完成
- **THEN** 汇总 MUST 按 label budget 聚合 Top-1、Top-3、Top-5 和 coarse accuracy
- **AND** 汇总 MUST 保留每个 seed 的原始指标路径

#### Scenario: 输出 efficiency 表
- **WHEN** adaptation variants 完成
- **THEN** 汇总 MUST 包含 trainable params、trainable ratio、adapt time per epoch、total adapt time 和 target_test Top-1
- **AND** adapter variants MUST 能与 full fine-tuning baseline 横向比较

#### Scenario: 汇总不伪造缺失指标
- **WHEN** 某个 run 未产生 power metrics 或 prototype metrics
- **THEN** 汇总 MUST 将对应字段标记为不可用或缺失
- **AND** 系统 MUST 不用 0 或其它数值伪造真实指标

