# Clean Data Integrity Specification

## Purpose

定义 MMW 唯一 ID-stratified block protocol 的 train/validation/test 隔离、four-modal topology predictor 的 sensing-only 边界与 TBCP train-only calibration 契约，并保持 DeepSense6G 独立数据契约。

## Requirements

### Requirement: MMW 只能使用唯一且精确绑定的 block protocol

MMW 训练和开发验证 MUST 通过 `mmw_id_stratified_block_v1` 建立数据域。配置 MUST 绑定 manifest path/hash、protocol version、split seed、block size、data source hash、window config hash、split roles 与通过的 audit。旧 trajectory-disjoint、clean-inner、group-safe、随机窗口或未知 protocol MUST 在 dataset 创建前失败。

#### Scenario: 构建合法 MMW loader

- **WHEN** 配置与受支持 manifest、audit 和 split CSV 完全一致
- **THEN** 系统 MUST 默认只构建 train 与 validation loader
- **AND** metadata MUST 记录 protocol、split seed、manifest identity 与 `test_evaluated=false`

### Requirement: split 间必须隔离 block、基础帧、天气副本与窗口帧

协议校验 MUST 允许同一 `(scene_id,cav_id)` 的不同 block 出现在不同 split，同时拒绝跨 split block、base sample、weather copy 和实际窗口帧重叠。共享 RSU context MAY 作为 diagnostic overlap 披露，不得改变 assignment。

#### Scenario: 其他 split 数据进入训练

- **WHEN** validation/test block、base frame、weather copy 或窗口被并入 train
- **THEN** audit MUST 失败且训练不得启动

### Requirement: sensing 模型只能读取四模态历史输入

four-modal topology predictor MUST 只消费 image、radar、gps、lidar、temporal/availability mask 与未来 beam label。CSI、channel、path、beam power、历史 beam、weather、scene、domain、corruption type 或 severity MUST 不得进入 model/loss；weather/domain MAY 仅用于只读评估分组。

#### Scenario: 配置请求旧路线输入

- **WHEN** model、loss 或 dataset 声明 CSI sidecar、stage、risk、learned fusion 或任一禁止输入
- **THEN** strict config validation MUST 在 dataset/model 创建前失败

### Requirement: 可拟合 sensing 状态只能来自 train

GPS scaler、prototype、normalization、memory/queue 与其他可拟合 sensing state MUST 只使用绑定 protocol 的 train role。validation/test MUST 不更新 optimizer、scheduler、prototype、statistics 或 checkpoint state；checkpoint selection MAY 只读取 validation loss。

#### Scenario: 开发验证

- **WHEN** validation evaluator 运行 15-mask matrix
- **THEN** validation MUST 保持只读且 test dataset MUST 不构建
- **AND** report MUST 记录 `claim_ineligible=true`、`outer_test_accessed=false`

### Requirement: probing likelihood 只能使用 train radio ground truth

TBCP MAY 拟合独立 topology likelihood artifact，但只允许读取绑定 train role 的官方 64-beam power 与 argmax label。artifact MUST 记录 train identity/count/hash、source content hash、protocol 与 topology provenance；MUST NOT 进入 sensing model forward、loss、optimizer、checkpoint 或 validation fitting。

#### Scenario: validation/test 尝试更新 likelihood

- **WHEN** fitter 收到非 train role 或 identity/hash 漂移
- **THEN** MUST 在读取完整目标 power 前失败
- **AND** 不得通过 confirmation、trainval、validation replay 或 test sensitivity 更新 artifact

### Requirement: finite probing 必须隔离 evaluation radio ground truth

radio simulator MAY 私有持有 evaluation 样本完整 64-beam power，但 candidate policy MUST 只接收 sensing posterior、train-only likelihood 与已请求 measurements。GT、channel、未请求 power、完整 vector 与 metric denominator MUST 不进入非 oracle policy。

#### Scenario: policy 请求 oracle 信息

- **WHEN** 非 oracle candidate path 尝试读取完整 power、GT 或未请求 beam
- **THEN** API MUST 拒绝且不得生成可用报告

### Requirement: test 默认封存

MMW test 只有独立显式 `--evaluate-test` 才可构建，并只能用于最终只读评估。confirmation、trainval、merged split、validation/test driven model/policy selection MUST 被拒绝。

#### Scenario: 未授权 test 请求

- **WHEN** 开发训练、matrix、probing 或 calibration 没有显式 test 授权
- **THEN** loader 集合 MUST 只有 train/validation 且 metadata MUST 保持 `test_evaluated=false`

### Requirement: DeepSense6G 保持独立

DeepSense6G MUST 保留 Scene31--34、四模态和 64 类 future-beam 契约，不得要求 MMW protocol。

#### Scenario: 加载 DeepSense6G recipe

- **WHEN** 用户加载保留的 Scene31--34 配置
- **THEN** 系统 MUST 使用 DeepSense6G 自身 split 与 train-only normalization 契约
- **AND** MUST NOT 注入 MMW manifest、topology audit 或 probing artifact
