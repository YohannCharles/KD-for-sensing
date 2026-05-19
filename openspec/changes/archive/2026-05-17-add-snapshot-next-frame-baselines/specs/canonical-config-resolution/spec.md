## ADDED Requirements

### Requirement: Snapshot canonical 配置解析
配置加载流程 MUST 能识别并生成 snapshot next-frame baseline 配置。可生成路径 MUST 包含单模态 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 和 fusion `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`。

#### Scenario: 生成单模态 snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/gps/snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 GPS snapshot 配置
- **AND** 最终配置 MUST 设置 `experiment.task: gps`
- **AND** 最终配置 MUST 设置 `data.dataset.seq_len: 1` 和 `data.dataset.num_pred: 1`
- **AND** 最终配置 MUST 设置 `data.dataset.train_csv_name: train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 最终配置 MUST 设置 `data.dataset.val_csv_name: val_seqs_SNAPSHOT_NEXT_FRAME.csv` 或等价 validation CSV 字段
- **AND** 最终配置 MUST 构建 `snapshot_frame` core

#### Scenario: 生成 fusion snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/fusion/gps_mmwave_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 fusion snapshot 配置
- **AND** 最终配置 MUST 设置启用模态为 `["gps", "mmwave"]`
- **AND** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型

#### Scenario: 拒绝非法 snapshot slug
- **WHEN** 用户加载 `configs/fusion/mmwave_gps_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

### Requirement: Snapshot 配置生成语义
生成的 snapshot 配置 MUST 明确覆盖历史窗口默认值，并保持实体 YAML 优先、命令行覆盖后应用和 schema 校验流程一致。

#### Scenario: 覆盖历史窗口默认值
- **WHEN** 系统生成任一 snapshot 配置
- **THEN** 生成配置 MUST 覆盖默认 `seq_len=8` 和 `num_pred=3`
- **AND** 生成配置 MUST 覆盖默认 GRU representation core
- **AND** 生成配置 MUST 覆盖默认历史窗口 CSV 为 snapshot 专用 train/validation CSV
- **AND** 生成配置 MUST 设置 `output.run_name` 包含 `snapshot_next_frame_no_kd`

#### Scenario: 实体 snapshot 配置优先
- **WHEN** 用户加载磁盘上存在的 snapshot YAML
- **THEN** 配置加载流程 MUST 使用实体 YAML 内容
- **AND** virtual snapshot 规则 MUST 不覆盖同名实体配置

#### Scenario: 命令行覆盖仍生效
- **WHEN** 用户加载 snapshot virtual 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** snapshot 必需字段 MUST 继续满足 `seq_len=1`、`num_pred=1` 和无时序 core 契约，除非用户显式退出 snapshot 变体并通过校验
