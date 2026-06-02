## ADDED Requirements

### Requirement: Image-only 合法 crossroad probe 配置
系统 MUST 提供 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`，用于在 MMW Town10 crossroad target scene 上运行 image-only 合法 few-shot probe。该配置 MUST 显式声明启用模态仅为 `image`，禁用 GPS、LiDAR、radar、mmWave、CSI、channel、path 和 beam_power，且 target adaptation MUST 只允许使用 target support image 与 beam label。

#### Scenario: 配置声明 image-only 协议
- **WHEN** 用户加载 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`
- **THEN** resolved modalities MUST 等价于 `["image"]`
- **AND** protocol metadata MUST 记录 `image_only=true`
- **AND** disabled/excluded sensitive fields MUST 包含 `gps`、`lidar`、`radar`、`mmwave`、`csi`、`channel`、`path` 和 `beam_power`
- **AND** `allow_target_unlabeled`、`allow_target_radio_oracle`、`allow_target_path_oracle`、`allow_target_beam_power_oracle` 和 `allow_target_test_labels_for_adaptation` MUST 默认为 false

#### Scenario: few-shot support 标签预算可审计
- **WHEN** image-only probe 使用 `label_budget=10`
- **THEN** sampling metadata MUST 记录 target support sample id、beam label、sampling seed、target scene 和 source scenes
- **AND** target test labels MUST NOT 出现在 target adaptation sampling manifest 中

### Requirement: Image-only probe 运行矩阵
系统 MUST 提供 `scripts/run_image_only_legal_crossroad_probe.sh`，并至少执行 I0 `image_source_only`、I1 `image_target_linear_probe`、I2 `image_v8_target_prior_head`、I3 `image_v9_sector_proto` 四个模式。脚本中所有 Python 训练、验证、评估命令 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: 运行四个 probe mode
- **WHEN** 用户执行 image-only legal crossroad probe 脚本
- **THEN** 系统 MUST 为 I0 写入 `outputs/image_only_legal_seed0/I0_source_only`
- **AND** 系统 MUST 为 I1 写入 `outputs/image_only_legal_seed0/I1_linear_probe`
- **AND** 系统 MUST 为 I2 写入 `outputs/image_only_legal_seed0/I2_v8_target_prior`
- **AND** 系统 MUST 为 I3 写入 `outputs/image_only_legal_seed0/I3_v9_sector_proto`

#### Scenario: 每个 run 执行标准阶段
- **WHEN** 任一 image-only probe run 启动
- **THEN** run MUST 执行 source_train、source_only_target_test_eval、summary 和 eligibility_check
- **AND** 除 I0 外的 run MUST 执行 target_adaptation 和 adapted_target_test_eval
- **AND** I0 MUST NOT 执行 target adaptation

### Requirement: Image feature cache
系统 MUST 支持 image backbone feature cache，以便快速执行 small head probe 和 target adaptation。cache MUST 只保存 image backbone 输出 feature、beam label、scene、sample_id 和 split metadata，不得保存 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power 字段。

#### Scenario: 写出 split feature cache
- **WHEN** `feature_cache.enabled=true` 且 image-only probe 提取 feature
- **THEN** 系统 MUST 在配置的 cache 目录写出 source_train、target_support 和 target_test 对应 cache 文件
- **AND** cache tensor MUST 包含 `features` 和 `labels`
- **AND** metadata MUST 包含 `checkpoint`、`feature_dim`、`modalities`、`image_encoder`、`created_at`、`source_scenes`、`target_scene` 和 `label_budget`

#### Scenario: adaptation 不读取 target_test cache labels
- **WHEN** target adaptation 使用 image feature cache
- **THEN** adaptation MUST 只读取 target_support cache labels 训练 target head、target prior 或 sector prototype
- **AND** target_test cache labels MUST 只允许 evaluation scope 用于最终指标计算

#### Scenario: cache 与 checkpoint 不匹配
- **WHEN** cache metadata 中的 checkpoint、feature_dim、modalities、image_encoder、source scenes、target scene 或 label_budget 与当前 run 不匹配
- **THEN** 系统 MUST 拒绝复用 cache 或要求显式 overwrite
- **AND** 错误信息 MUST 指出不匹配字段

### Requirement: Image-only probe 诊断和汇总产物
每个 image-only probe run MUST 写出机器可读诊断产物，用于判断 source prior collapse、target prior collapse、prototype 稳定性和合法性。总输出目录 MUST 写出 `combined_summary.csv`；当实现提供 histogram 跨 run 汇总时，系统 MUST 写出 `combined_prediction_hist_summary.json`。

#### Scenario: 写出 prediction histogram
- **WHEN** 任一 image-only probe run 完成 target_test evaluation
- **THEN** run directory MUST 包含 `prediction_hist.json`
- **AND** `prediction_hist.json` MUST 至少包含 `true_hist`、`pred_hist`、`true_top_beams`、`pred_top_beams`、`unique_pred_beams`、`top1_pred_beam_ratio`、`top2_pred_beam_ratio`、`top5_pred_beam_ratio`、`mean_abs_beam_error`、`within_1_acc`、`within_2_acc` 和 `within_3_acc`

#### Scenario: 写出 confusion by true beam
- **WHEN** 任一 image-only probe run 完成 target_test evaluation
- **THEN** run directory MUST 包含 `confusion_by_true_beam.json`
- **AND** 文件 MUST 按 true beam 到 predicted beam 计数字典组织

#### Scenario: 写出 combined summary
- **WHEN** image-only probe 脚本完成或部分失败退出
- **THEN** 总输出目录 MUST 写出 `combined_summary.csv`
- **AND** CSV MUST 至少包含 `mode`、`top1`、`top3`、`top5`、`within1`、`within2`、`within3`、`mae`、`bpl_db`、`nrp`、`unique_pred_beams`、`top1_pred_beam_ratio`、`top5_pred_beam_ratio`、`eligible`、`eligibility_reasons` 和 `trainable_ratio`

### Requirement: Image-only probe 成功标准
系统 MUST 将 image-only legal probe 的 P0 成功标准定义为所有四个 run 正常结束、至少一个 eligible run 可进入主结论，且 summary 不含 target oracle、target radio/path supervision 或 split eligibility unknown 排除原因。

#### Scenario: P0 成功判定
- **WHEN** image-only probe summary 生成结论
- **THEN** conclusion MUST 要求所有 image-only run 的 exit code 为 0
- **AND** `eligible_run_count` MUST 大于 0
- **AND** summary MUST 记录 `target_oracle_fields_used=false`
- **AND** summary MUST 记录 `target_radio_label_supervision=false`
- **AND** summary MUST 记录 `target_path_label_supervision=false`
- **AND** summary MUST 记录 `split_eligibility_unknown=false`

#### Scenario: smoke test 覆盖最小链路
- **WHEN** 开发者实现或修改 image-only probe
- **THEN** smoke test MUST 覆盖 image-only dataloader one batch、source forward、target adaptation forward、loss backward、eval metrics 和 eligibility check
- **AND** smoke test 命令 MUST 使用 `conda run -n kd_mm_beam`
