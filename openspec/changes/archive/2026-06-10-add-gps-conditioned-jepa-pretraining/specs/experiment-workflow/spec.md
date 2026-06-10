## ADDED Requirements

### Requirement: 配置驱动 JEPA 预训练
项目 MUST 支持通过现有训练 CLI 运行 GPS-conditioned JEPA 预训练配置。该配置 MUST 使用 `model.primary` 构建 JEPA 主模型，MUST 通过 image/GPS 模态契约准备输入，MUST 使用 `gps_conditioned_jepa` objective 计算自监督 latent loss，并 MUST 不构建 distiller、旧 teacher/student KD runtime 或外部 frozen teacher checkpoint。

#### Scenario: 使用配置启动 JEPA 预训练
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-train --config <jepa_config>`
- **THEN** 系统 MUST 构建包含 image 和 GPS 输入的 dataset/dataloader
- **AND** 系统 MUST 构建 `model.primary` 指定的 GPS-conditioned JEPA 模型
- **AND** 系统 MUST 以 JEPA latent loss 完成 forward、backward、optimizer step 和 EMA target encoder update

#### Scenario: JEPA 配置不启用 KD
- **WHEN** 用户加载 JEPA 预训练配置
- **THEN** 配置 MUST 不包含 `distillation.*`、`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd` 或旧 KD alias
- **AND** 配置加载和训练构建流程 MUST 不构建 distiller 或 frozen teacher checkpoint

### Requirement: JEPA 验证与运行产物
JEPA 预训练 workflow MUST 在验证阶段计算 `val_jepa_loss`，并 MUST 将运行产物写入当前统一输出目录。运行目录 MUST 保存解析后的配置、训练日志、checkpoint、TensorBoard 标量、运行状态和 objective metadata。固定 `output.run_name` 的唯一目录、resume 和 overwrite 语义 MUST 与现有训练 workflow 保持一致。

#### Scenario: JEPA validation 输出
- **WHEN** JEPA 训练完成一个 epoch 并进入验证
- **THEN** validation MUST 计算并返回 `val_jepa_loss`
- **AND** validation MUST 不要求 `target_beam`
- **AND** validation MUST 不写出 beam Top-K、DBA 或 beam prediction report

#### Scenario: JEPA final config metadata
- **WHEN** JEPA 训练写出最终运行配置
- **THEN** `final_config.yaml` MUST 记录解析后的 `experiment.objective: gps_conditioned_jepa`
- **AND** runtime metadata MUST 记录 JEPA loss、主 metric、metric mode、mask sampler、EMA decay 和 context encoder artifact 信息

#### Scenario: JEPA resume 复用运行目录
- **WHEN** 用户设置 `training.resume: true` 并提供 JEPA `output.run_name`
- **THEN** 训练流程 MUST 从该运行目录恢复 checkpoint
- **AND** 恢复 MUST 包含 context encoder、target encoder、optimizer、scheduler、epoch 和 best `val_jepa_loss`

### Requirement: JEPA canonical smoke 配置
项目 MUST 提供一个用于快速验证的 GPS-conditioned JEPA 配置。该配置 MUST 使用当前保留的数据集和模态契约，默认启用 image RGB/ImageNet profile 和 GPS relative-polar feature，默认输出到 `outputs/`，并 MUST 能通过小 epoch/小 batch override 运行 smoke test。

#### Scenario: JEPA smoke 配置可加载
- **WHEN** 开发者加载 JEPA canonical smoke 配置
- **THEN** 配置 MUST 解析为 `experiment.objective: gps_conditioned_jepa`
- **AND** `model.primary.type` MUST 为已注册 JEPA 模型
- **AND** image 与 GPS 输入 profile MUST 与模态契约一致

#### Scenario: JEPA smoke test
- **WHEN** 开发者使用 synthetic 或小比例数据运行 1 epoch JEPA smoke
- **THEN** 训练 MUST 完成 forward、loss、backward、optimizer step、EMA update、validation 和 checkpoint 保存
- **AND** 运行产物 MUST 包含 `val_jepa_loss`

### Requirement: JEPA paper-split full 配置
项目 MUST 提供主 JEPA 预训练和 GPS-biased mask ablation 的 low-memory full 配置。该配置 MUST 使用 DeepSense6G scenes 32、33、34 作为训练来源，并 MUST 将 scenes 31、32、33、34 纳入验证/监控集合。多场景训练运行产物 MUST 记录 source scene 列表与每个 scene 的样本数，避免被误判为 scene31-only 实验。

#### Scenario: JEPA full 配置使用多场景训练
- **WHEN** 用户加载 JEPA full low-memory 配置
- **THEN** `data.dataset.train_scenes` MUST 包含 32、33 和 34
- **AND** `data.dataset.test_scenes` MUST 包含 31、32、33 和 34
- **AND** 输出 run name MUST 明确区分该 run 是 scenes32-34 训练口径

### Requirement: 现有 supervised/adaptation workflow 不变
新增 JEPA 预训练 workflow MUST 不改变现有 beam、occlusion、position、multitask、Raymobtime selection、GPS v2、Top8、BGAM、CSI hardening、viewer 或 supervised fusion workflow 的默认配置和指标。

#### Scenario: 默认 beam 配置行为不变
- **WHEN** 用户加载未设置 `experiment.objective` 的现有 supervised beam 配置
- **THEN** 系统 MUST 继续默认使用 `beam` objective
- **AND** 系统 MUST 继续计算 beam loss、Top-K、DBA 和 `val_adba`

#### Scenario: 旧 KD 入口仍被拒绝
- **WHEN** 用户请求旧 `logits_kd`、`rkd`、`teacher_no_kd` 或 retired fusion KD 配置
- **THEN** 系统 MUST 继续拒绝该配置
- **AND** 错误信息 MUST 继续指向当前 supervised/adaptation 或 JEPA 预训练入口，而不是恢复旧 KD workflow

### Requirement: BeamBench-fair supervised 下游验证
项目 MUST 提供 image+GPS supervised fair low-memory 配置族，用于比较 supervised baseline 与 JEPA context encoder 初始化的下游 beam prediction。该配置族 MUST 使用 DeepSense6G scenes 32、33、34 的训练 split 作为训练来源，MUST 使用训练 split 内部划分的 validation 子集做 checkpoint selection，MUST 在训练完成后单独评估 scenes 31、32、33、34 的 test split，并 MUST 将 final test metrics 写入运行 metadata。

#### Scenario: fair 配置使用独立选模 split
- **WHEN** 用户加载 BeamBench-fair supervised 下游配置
- **THEN** `data.validation_from_train.enabled` MUST 为 true
- **AND** 训练循环 MUST 优先使用 `validation` dataloader 计算 early stopping metric
- **AND** `test` dataloader MUST 不用于 early stopping/checkpoint selection

#### Scenario: fair 配置训练后执行 final test
- **WHEN** fair supervised 训练结束且运行目录存在 `checkpoints/best.pth`
- **THEN** 系统 MUST 重新加载该 checkpoint
- **AND** 系统 MUST 在 `test` dataloader 上计算 final test metrics
- **AND** runtime metadata MUST 记录 `final_test_metrics.evaluation_split: test`
- **AND** runtime metadata MUST 记录 `final_test_metrics.model_selection_split`

#### Scenario: fair 配置使用 BeamBench DBA 口径
- **WHEN** fair supervised 配置计算 DBA 或 ADBA
- **THEN** `evaluation.dba_distance_mode` MUST 支持并设置为 `linear`
- **AND** linear 模式 MUST 使用非环形 beam index 距离
- **AND** 未显式设置该字段的现有配置 MUST 继续使用 circular DBA 默认行为

#### Scenario: fair 配置固定论文预测窗口
- **WHEN** fair supervised 配置被加载
- **THEN** `data.dataset.num_pred` 和 `model.num_pred` MUST 为 1
- **AND** scheduler MUST 设置为 `none`
- **AND** 配置 MUST NOT 因 BeamBench 原文未明确历史输入长度而强制修改现有 `seq_len: 8` 工作流

### Requirement: 2604.05668 对齐 supervised 下游验证
项目 MUST 提供 image+GPS supervised 2604 对齐配置族，用于与 arXiv:2604.05668 的 S32/S33/S34 主表口径比较。该配置族 MUST 合并 DeepSense6G scenes 32、33、34 的官方 train/test labeled CSV，MUST 在每个 scene 内按 beam label 做固定 seed 的 80/10/10 stratified train/validation/test split，MUST 使用 `seq_len: 5` 和 `num_pred: 1`，并 MUST 记录 split protocol 与每个 split 的样本数。

#### Scenario: 2604 配置使用合并后 stratified split
- **WHEN** 用户加载 2604 对齐 supervised 配置
- **THEN** `data.dataset.split_protocol` MUST 为 `stratified_80_10_10`
- **AND** `data.dataset.train_scenes`、`validation_scenes` 和 `test_scenes` MUST 包含 32、33 和 34
- **AND** train/validation/test MUST 来源于每个 scene 的 `train_seqs_RA_GPS_LIDAR.csv` 与 `test_seqs_RA_GPS_LIDAR.csv` 合并集合

#### Scenario: 2604 配置匹配历史窗口
- **WHEN** 2604 对齐 supervised 配置被加载
- **THEN** `data.dataset.seq_len` 和 `model.seq_length` MUST 为 5
- **AND** `data.dataset.num_pred` 和 `model.num_pred` MUST 为 1

#### Scenario: 2604 split 使用 train-only normalization
- **WHEN** 2604 split protocol 构建 image+GPS dataloader
- **THEN** GPS scaler MUST 只从 stratified train 子集拟合
- **AND** validation/test MUST 复用该 scaler
- **AND** runtime metadata MUST 记录 scaler 来源与 split protocol
