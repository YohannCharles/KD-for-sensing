## ADDED Requirements

### Requirement: Resolved config artifact and startup summary
Every debug run MUST save the fully resolved configuration and print a startup summary of the fields needed to compare experiment variants. The summary MUST be generated after defaults, aliases and command-line overrides are applied.

#### Scenario: 保存 resolved config
- **WHEN** a debug run starts
- **THEN** the run output directory MUST contain `resolved_config.yaml` or an equivalent fully resolved config artifact
- **AND** the artifact MUST reflect defaults, generated config values, aliases and command-line overrides

#### Scenario: 打印关键配置摘要
- **WHEN** a debug run starts
- **THEN** startup logs MUST include modalities, dataset path, train/val split paths, `seq_len`, `num_pred`, `num_classes`, batch size, optimizer, learning rate, scheduler and max epochs
- **AND** startup logs MUST include model type, CSI encoder type, `d_model`, `delay_taps`, `view_fusion`, `use_internal_gru`, pilot estimator enabled/mode/SNR, `csi_hardening.enabled` and `csi_degradation.enabled`

### Requirement: Baseline clone config diff artifact
The experiment workflow MUST support comparing a generated baseline clone against a reference baseline resolved config. The diff MUST separate allowed run identity differences from key behavior differences.

#### Scenario: 生成 A0 clone diff
- **WHEN** both `A0_original` and `A0_clone_generated` resolved configs are available
- **THEN** the workflow MUST produce a diff artifact comparing them
- **AND** the diff MUST ignore only allowlisted run identity fields such as run name, output directory, timestamp and seed when configured

#### Scenario: 关键字段差异失败
- **WHEN** the diff finds a difference in optimizer, scheduler, loss, dataset split, normalization, train RMS path, `seq_len`, `num_pred`, `num_classes`, model type, CSI encoder, representation core or beam head
- **THEN** the workflow MUST mark the parity check as failed
- **AND** the failure message MUST list the differing config paths

### Requirement: Module trainability startup report
The training workflow MUST report trainable parameter counts by major module for debug runs. The report MUST distinguish CSI encoder, representation core, beam head and fusion modules when those modules exist.

#### Scenario: 打印模块参数统计
- **WHEN** a debug run builds the model
- **THEN** startup logs MUST include total parameter count and total trainable parameter count
- **AND** startup logs MUST include trainable parameter counts by CSI encoder, representation core, beam head and fusion module where present

#### Scenario: 发现模块无可训练参数
- **WHEN** a required trainable module has zero trainable parameters
- **THEN** startup logs MUST mark the module as suspicious
- **AND** the warning MUST include the module name and resolved model path

### Requirement: Debug metrics logging
The training workflow MUST persist debug diagnostics in machine-readable run logs when debug mode is enabled. The diagnostics MUST be scoped so normal runs are unaffected when debug mode is disabled.

#### Scenario: 持久化首 batch 诊断
- **WHEN** CSI first-batch debug diagnostics are produced
- **THEN** the workflow MUST write them to the run log, metadata artifact or TensorBoard text/scalar stream
- **AND** the stored record MUST distinguish train and validation batch sources

#### Scenario: 持久化 epoch 训练健康指标
- **WHEN** epoch-level grad norm and param delta diagnostics are produced
- **THEN** the workflow MUST append them to the epoch metrics log
- **AND** normal training metrics arrays MUST remain backward compatible for existing analysis scripts
