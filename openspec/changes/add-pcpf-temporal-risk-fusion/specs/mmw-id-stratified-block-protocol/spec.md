## MODIFIED Requirements

### Requirement: MMW block assignment 必须平衡 domain 内 beam 分布

`mmw_id_stratified_block_v1` MUST 使用默认 32-base-frame 连续 block 和 manifest schema version 2。assignment MUST 在固定 70/15/15、三 role trajectory 覆盖与零 block/base/weather/window-frame overlap 下，同时优化全局、按 scene/domain 及按 `(scene_id,cav_id)` trajectory 的 train--validation/train--test beam TV，并惩罚条件 held-out beam 在相应 train 中缺失。旧 128-block/global-only assignment manifest MUST 失败关闭并显式 regenerate；其 normalization、GPS 和 sparse-CSI split-specific cache MUST 随 manifest hash 重建。

#### Scenario: 旧全局标签平衡 manifest 被复用

- **WHEN** loader 或 PCPF resolver 收到 manifest schema version 1、旧 assignment algorithm 或 block size 128 的产物
- **THEN** 系统 MUST 在 dataset 创建前拒绝
- **AND** 不得沿用旧 split-specific normalization、GPS、CSI bundle 或 checkpoint

#### Scenario: 生成 seed 0 代表性报告

- **WHEN** builder 完成默认 seed 0 assignment
- **THEN** report MUST 同时给出全局、scene/domain 与 trajectory 的 train--validation/train--test TV、macro/worst 和条件未覆盖 beam 质量
- **AND** MUST 保留简单连续 block baseline 与全部 leakage 检查

### Requirement: PCPF-T 必须绑定唯一 MMW block protocol

PCPF-T resolver、preflight、train、continue-pipeline、gate、matrix 与 sparse-CSI sidecar MUST 只接受 `mmw_id_stratified_block_v1` protocol version 1、manifest schema version 2 和 conditional assignment v2，并绑定相同 split seed、block size、manifest hash、data source hash、window config hash 与 train seed。旧 `mmw_trajectory_disjoint`、clean-inner、group-safe、随机窗口及其 cache/checkpoint MUST 失败关闭。

#### Scenario: PCPF-T 请求旧 split artifact

- **WHEN** resolved config、checkpoint、normalization 或 sparse-CSI bundle 绑定旧 protocol 或不同 manifest hash
- **THEN** runner MUST 在 dataset 或 optimizer 创建前拒绝

### Requirement: sparse CSI 必须遵守 block 与 split-specific cache 边界

channel 默认只可用于泄漏诊断；`use_sparse_csi=true` 时，系统 MAY 使用当前窗口自身五帧历史 channel 生成固定 sparse CSI sidecar。sidecar 与 packed bundle MUST 记录完整 block protocol cache identity，只扫描 train/validation，且不得包含 test feature。原始内容寻址 CSI cache MAY 复用，但 split-specific index、coverage 和 bundle MUST 按新 manifest 重建。

#### Scenario: test 或跨 block channel 进入 packed bundle

- **WHEN** bundle coverage 含 test identity、跨 split frame 或跨 block window
- **THEN** cache validation MUST 失败且不得回退到在线 source channel

### Requirement: PCPF-T 开发默认不得访问 test

开发运行 MUST 固定复用 seed 0 manifest 并默认只构建 train/validation。只有独立显式 test evaluation 才可加载 test；continue-pipeline、Stage 2 gate、R0--R7 model selection 和数据画像 MUST 保持 `test_evaluated=false`。所有可拟合风险状态、prototype、temperature/calibration 与 memory/feature statistics MUST 只来自 train。

#### Scenario: 继续三阶段训练

- **WHEN** continue-pipeline 解析 Stage 2/3
- **THEN** 全部 stage MUST 绑定同一 block manifest 和 train seed
- **AND** test loader MUST 不存在

### Requirement: 数据画像必须适配同 trajectory 的 block overlap 语义

数据画像 MUST 将 trajectory overlap 标记为协议设计目标，并验证 block/base frame/weather copy/window frame overlap 为零。报告 MUST 使用实际 train/validation split，只有显式独立 test report 才可读取 test；不得再把 validation 解释为未知完整 trajectory 泛化。

#### Scenario: 分析 seed 0 开发 split

- **WHEN** 分析器读取 train/validation CSV
- **THEN** 输出 MUST 绑定 block/manifest/window identity 并记录 `test_evaluated=false`
- **AND** 任一旧协议、hash、count 或 role 漂移 MUST 在信号扫描前失败
