# Clean Data Integrity Specification

## Purpose

定义 MMW 唯一 ID stratified block protocol 的 train/validation/test 数据完整性、train-only 拟合状态与默认 test 封存，并保持 DeepSense6G 独立数据契约。

## Requirements

### Requirement: MMW 只能使用唯一且精确绑定的 block protocol

MMW 训练和开发验证 MUST 通过 protocol `mmw_id_stratified_block_v1` 建立数据域。配置 MUST 显式绑定 manifest 路径/hash、protocol version、split seed、block size、data source hash、window config hash、train/validation/test role 与通过的 audit report。clean-inner、trajectory-disjoint、历史 group-safe、窗口随机拆分和未知 MMW protocol MUST 在创建 dataset 前失败。

#### Scenario: 构建合法 MMW loader

- **WHEN** 配置与受支持 manifest、audit 和 split CSV 完全一致
- **THEN** 系统 MUST 默认只构建 train 与 validation loader
- **AND** 每个 loader 的运行元数据 MUST 记录 protocol、split seed 与 manifest identity

### Requirement: split 间必须隔离 block、基础帧、天气副本与窗口帧

协议校验 MUST 允许并要求同一 `(scene_id,cav_id)` 的不同 block 出现在 train、validation、test，同时拒绝跨 split block、base sample、天气副本和窗口实际引用帧。系统 MUST 审计 sample/target identity、完整 source row、历史/未来 frame、weather binding、block identity 与 split assignment。共享 RSU context MUST 作为 diagnostic overlap 披露，不得用于改变 block assignment。

#### Scenario: 其他 split 的 block 被并入训练输入

- **WHEN** validation/test block 或窗口被加入 train，或任一窗口跨 block
- **THEN** audit MUST 失败
- **AND** 配置不得进入训练或开发评估

### Requirement: 可拟合状态只能来自训练集

GPS scaler、CSI codebook、prototype 统计、contrastive memory/negative queue 与其他 normalization artifact MUST 只由绑定 protocol 的 train loader 拟合。validation/test MUST 不参与 optimizer、scheduler、extension state、训练采样、prototype 初始化、class prior 或可拟合统计更新；checkpoint selection 只能读取 validation loss。test 只有显式 `--evaluate-test` 才可构建，且只能用于最终只读评估。

#### Scenario: 运行开发验证

- **WHEN** 系统对 MMW validation loader 评估
- **THEN** validation MUST 是只读输入
- **AND** 默认不得构建 test dataset，metadata MUST 保持 `test_evaluated=false`

### Requirement: DeepSense6G 不受 MMW protocol 重解释

DeepSense6G MUST 保留 Scene31--34、四模态和 64 类 future-beam 数据契约。它可以有 train/test 或显式 validation CSV，但不得要求 MMW trajectory protocol。

#### Scenario: 加载 DeepSense6G recipe

- **WHEN** recipe 没有 MMW data protocol
- **THEN** 数据工厂 MUST 使用其独立 split 契约

### Requirement: PCPF-T 可拟合风险状态只能来自绑定协议的 train role

PCPF-T 的风险分量 mean/std、静态能力先验、`mean_train_risk`、温度、解析融合参数和 checkpoint selection MUST 只使用所绑定 `mmw_id_stratified_block_v1` seed manifest 的 train role 与只读 validation role。所有可拟合统计 MUST 仅遍历 manifest 声明的 train windows，validation/test MUST 不更新模型、统计、阈值、prototype、memory bank 或 gate；默认运行 MUST 记录 `test_evaluated=false`。

#### Scenario: 准备 Stage 2 或 Stage 3

- **WHEN** trainer 调用 PCPF-T 的 stage preparation
- **THEN** preparation MUST 只接收 train dataset 和 train temporal-missing transform
- **AND** 产物 MUST 记录 protocol、split role、遍历范围和 train-only 状态

### Requirement: 历史 development evaluation 必须显式降级声明

PCPF-T MAY 对 validation 做只读诊断，但配置和全部开发报告 MUST 固定记录 `claim_ineligible: true`。test 只能由独立显式最终评估读取；confirmation、trainval、merged split 或 test-driven gate/融合/模型选择 MUST 被拒绝。

#### Scenario: 评估 validation 或显式 test

- **WHEN** resolved config 运行开发 validation 或显式最终 test
- **THEN** evaluator MUST 保持只读并输出 `claim_ineligible: true`
- **AND** 未显式授权的 test 请求 MUST 在 dataset 创建前失败

### Requirement: PCPF-T 输入必须保持历史 sensing-only

PCPF-T 默认模型、风险 target 和 stage preparation MUST 只消费 canonical image、radar、gps、lidar、temporal availability mask 与未来 beam label。只有配置显式声明 `use_sparse_csi=true` 时，模型 MAY 额外消费从同一样本五帧历史 `csi1..csi5` channel 引用按预注册 2x2 默认选择确定性生成的 sparse CSI；C2 开销筛选及其预注册 J2 三阶段 lineage MAY 使用同一个 4x2 选择。当前/未来 CSI、未来 channel、path、beam power、历史 beam、天气、场景和 corruption metadata MUST 不得进入 forward、风险 target 或可拟合统计；天气与 domain MAY 仅作为评估分组元数据。

#### Scenario: 配置携带禁止字段

- **WHEN** PCPF-T model、loss 或 risk 配置声明任一禁止输入
- **THEN** 严格配置校验 MUST 在模型和 dataset 创建前失败

#### Scenario: 构建 sparse CSI batch

- **WHEN** PCPF-T 配置显式启用历史 sparse CSI
- **THEN** dataset MUST 验证五个 channel 引用分别匹配历史 frame id 且最后历史帧早于 target
- **AND** 生成与编码 MUST 不注入 AWGN、dropout、corruption、当前/未来 CSI 或虚构 SNR
- **AND** metadata MUST 记录 split role、selection/codebook identity 与 `snr_available=false`
