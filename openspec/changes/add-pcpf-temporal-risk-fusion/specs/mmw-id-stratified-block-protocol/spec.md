## MODIFIED Requirements

### Requirement: PCPF-T 必须绑定唯一 MMW block protocol

PCPF-T resolver、preflight、train、continue-pipeline、gate、matrix 与 sparse-CSI sidecar MUST 只接受 `mmw_id_stratified_block_v1` version 1 manifest，并绑定相同 split seed、block size、manifest hash、data source hash、window config hash 与 train seed。旧 `mmw_trajectory_disjoint`、clean-inner、group-safe、随机窗口及其 cache/checkpoint MUST 失败关闭。

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
