# radio-semantic-hist-beam-adaptation Specification

## Purpose
定义 radio-semantic HiST-Beam 适配的标签构造、dataset contract、模型输出、prototype artifact、target adaptation 和评估报告契约，使 radio 语义作为可审计的辅助监督与 prototype 对齐信号，而不是隐式改变 sensing 输入模态边界。
## Requirements
### Requirement: Radio-semantic label 构造
系统 MUST 提供可配置的 RadioSemanticLabelBuilder，用于从 beam-power profile 或 beam label 派生 radio-semantic label。builder MUST 支持 `coarse`、`peak_spread` 和 `kmeans_power` 模式；快速验证默认 MUST 使用 `peak_spread`，并 MUST 在 metadata 中记录模式、阈值、类别数、输入来源和 unavailable reason。

#### Scenario: peak_spread 生成传播语义标签
- **WHEN** 输入样本包含 finite 的 `beam_power` 向量、`num_beams=64`、`group_size=8`、`num_spread_bins=3` 和 entropy thresholds `[0.35, 0.65]`
- **THEN** builder MUST 计算 `best_beam=argmax(beam_power)`、`peak_group=best_beam//8` 和归一化 entropy
- **AND** builder MUST 根据 entropy 生成 spread bin，并输出 `radio_semantic_label = peak_group * 3 + spread_bin`
- **AND** 输出 label MUST 位于 `[0, 24)` 范围内

#### Scenario: coarse fallback 可审计
- **WHEN** 配置允许 fallback 且样本没有可用 `beam_power`
- **THEN** builder MUST 使用 `beam//group_size` 或等价 coarse label 作为 fallback radio label
- **AND** diagnostics MUST 记录 `fallback_reason` 和 `radio_semantic_mode=coarse`

#### Scenario: 不可派生时不伪造标签
- **WHEN** 样本没有可用 `beam_power` 且没有合法 beam label，或 `beam_power` 含 NaN/Inf
- **THEN** builder MUST 不输出伪造的 radio label
- **AND** batch 或 metadata MUST 标记 radio label unavailable reason

### Requirement: Radio-semantic dataset contract
启用 radio-semantic 训练或评估时，dataset MUST 能在不改变 sensing modality 输入边界的情况下返回可选 `radio_semantic_label`、`beam_power`、domain metadata 和 sample id。CSI、channel path 和 beam_power MUST 只作为 label/metric/derived target 来源，不得自动作为 sensing 输入模态。

#### Scenario: MMW 样本返回 radio label
- **WHEN** MMW dataset 配置 `radio_semantic.enabled: true` 且样本 beam power 可派生 radio label
- **THEN** `__getitem__` 结果 MUST 包含 `radio_semantic_label`
- **AND** 结果 MUST 保留 `sample_id`、scenario/town/condition 或等价 domain metadata
- **AND** sensing 输入 MUST 只包含配置启用的 image/radar/GPS/LiDAR/IMU/mmWave feature 模态

#### Scenario: 无 beam power 时保持可训练
- **WHEN** 训练配置允许 radio label unavailable 且 batch 中部分样本缺少 radio label
- **THEN** dataloader MUST 保留这些样本用于 beam loss
- **AND** radio semantic loss MUST 只在有合法 radio label 的样本上计算

### Requirement: Radio-semantic HiST-Beam 模型输出与融合推理
HiST-Beam MUST 支持 radio-semantic prototype baseline。启用后 shared branch MUST 输出 `radio_logits`，beam prediction MUST 由 beam head 读取 shared representation、source 或 target-adapted private representation，以及可选 radio assignment embedding 后产生 64-beam logits。Radio prototype MUST NOT 直接输出 beam prediction。新增 V8 path-level physical prototype 后，radio-semantic 方法 MUST 保留为 V6 baseline 或 fallback，不得与 path-level prototype 混淆。

#### Scenario: source forward 输出 radio logits
- **WHEN** `radio_semantic.enabled: true` 且模型启用 `use_radio_head`
- **THEN** forward 输出 MUST 包含 `radio_logits` 和 shared/private/adapter representations
- **AND** `radio_logits` 的最后一维 MUST 等于 `num_radio_classes`

#### Scenario: radio-conditioned beam head
- **WHEN** `use_radio_condition_in_beam_head: true`
- **THEN** source 阶段 MUST 从 `softmax(radio_logits/tau)` 计算 `e_alpha`
- **AND** target 阶段在可用 source radio prototypes 时 MUST 从 `cosine(c, mu_radio_c)` assignment 计算 `e_alpha`
- **AND** beam head MUST 输出 beam-level logits，而不是 prototype-to-beam 映射

#### Scenario: 关闭 radio condition 时保持旧输入语义
- **WHEN** `use_radio_condition_in_beam_head: false`
- **THEN** beam head MUST 只读取 `concat(c, s_star)` 或当前等价的 shared/private 输入
- **AND** radio head MAY 继续作为 auxiliary supervision 输出

#### Scenario: radio baseline 与 path prototype 可区分
- **WHEN** 用户同时运行 V6 radio-semantic 和 V8 path-level prototype 实验
- **THEN** run metadata 和 summary MUST 区分 `proto_type=radio_semantic` 与 `proto_type=path`
- **AND** 系统 MUST NOT 使用 beam_power radio label 冒充 path_semantic_label，除非配置明确选择 `path_semantic.mode=radio_power` fallback/baseline

### Requirement: Source radio prototype artifact
完成 source training 后，系统 MUST 能基于 source train split 生成 shared radio prototype artifact。artifact MUST 至少包含 `mu_radio_c`、`count_radio`、可选 `mu_coarse_c/count_coarse`、label builder 配置、source domain、target domain、seed 和 class count diagnostics。

#### Scenario: 保存 shared radio prototypes
- **WHEN** source dataloader 中存在合法 `radio_semantic_label`
- **THEN** prototype generator MUST 按 radio label 聚合 shared representation
- **AND** artifact MUST 保存 `mu_radio_c` 与 `count_radio`
- **AND** artifact metadata MUST 记录 `prototype_space=shared_radio_semantic`

#### Scenario: 空 radio class 可诊断
- **WHEN** 某个 radio class 在 source train split 中没有样本
- **THEN** artifact MUST 将该 class 的 count 记录为 0
- **AND** target assignment MUST 不把该 class 当作高置信可用 prototype

#### Scenario: 保留 coarse prototype 作为消融
- **WHEN** 配置同时启用 coarse prototype artifact
- **THEN** 系统 MUST 保存 coarse prototype 和 radio prototype 的独立 counts
- **AND** summary MUST 能区分 V5 coarse prototype 与 V6 radio prototype 使用的 artifact

### Requirement: Radio-semantic target adaptation
系统 MUST 支持 `proto_type: radio_semantic` 的 target adaptation。该模式 MUST 冻结配置指定的 source encoders、fusion、shared/private encoder 和 radio head，只训练 private adapter、允许的 beam head 参数、可选 radio embedding/LayerNorm affine，以及 target-private prototype bank。

#### Scenario: 通过 source radio prototype 分配 target 样本
- **WHEN** target adaptation batch forward 得到 shared representation `c` 且存在 `mu_radio_c`
- **THEN** 系统 MUST 计算 `alpha = softmax(cosine(c, mu_radio_c)/tau)`
- **AND** 系统 MUST 记录 `r_hat`、confidence、prototype coverage 和 used sample count

#### Scenario: Target-private prototype bank 使用 EMA 更新
- **WHEN** target 样本 radio assignment confidence 高于阈值
- **THEN** 系统 MUST 用 adapted private representation `s_adapt` 更新对应 `nu_radio_s`
- **AND** 更新 MUST 使用配置中的 momentum 或等价 EMA 规则

#### Scenario: target-private prototype loss 不对齐 source private
- **WHEN** warmup 完成且 `nu_radio_s[r_hat]` 已初始化
- **THEN** 系统 MUST 对 `s_adapt` 与 stop-gradient 的 `nu_radio_s[r_hat]` 计算 private clustering loss
- **AND** 默认实现 MUST NOT 使用 source private prototype 作为 radio-semantic target private 对齐对象

### Requirement: Radio-semantic loss 与防泄漏
系统 MUST 支持 source radio semantic CE loss，并 MUST 在 target adaptation 中区分 labeled、unlabeled 和 target_test 使用边界。`label_budget=0` 或 unlabeled batch 训练时，系统 MUST NOT 使用 target beam、beam_power、q_power 或 radio_semantic_label 作为 supervised loss。

#### Scenario: source radio loss 只读取合法 radio labels
- **WHEN** source batch 包含合法 `radio_semantic_label` 且 `lambda_radio > 0`
- **THEN** 系统 MUST 对 `radio_logits` 计算 radio semantic CE
- **AND** diagnostics MUST 记录 radio loss 和有效样本 coverage

#### Scenario: 0-label target adaptation 禁止真实 target radio 监督
- **WHEN** target adaptation 的 `label_budget=0`
- **THEN** 系统 MUST 禁止 supervised beam CE、radio CE 和 power-profile KL 读取 target labels
- **AND** adapt log MUST 记录 `used_target_labels=false`、`used_target_beam_power_for_training=false` 和 `used_target_radio_label_for_training=false`

#### Scenario: target_test radio label 只用于评估
- **WHEN** target_test evaluation 可派生 `radio_semantic_label`
- **THEN** 系统 MAY 计算 radio semantic accuracy
- **AND** evaluation MUST 不把 target_test radio label 回传给 adaptation、threshold selection 或 prototype update

### Requirement: Radio-semantic 评估指标
Radio-semantic HiST-Beam evaluation MUST 输出传统 beam 指标与 radio/power/adaptation 指标。缺失必要数据时，系统 MUST 输出 unavailable reason，不得用 0 伪造。

#### Scenario: 输出 radio 与 power 指标
- **WHEN** target_test 样本包含 beam label、beam power 和 radio label
- **THEN** metrics MUST 包含 Top-1/3/5、coarse accuracy、radio semantic accuracy、normalized received power 和 beam power loss dB
- **AND** predictions MUST 记录 sample id、true/pred beam、radio true/pred 或 radio unavailable reason

#### Scenario: 输出 adaptation radio diagnostics
- **WHEN** radio prototype adaptation 完成
- **THEN** metrics MUST 包含 trainable parameter ratio、adaptation time、radio prototype coverage、confidence mean、target-private initialized count 和 prototype loss mean
- **AND** LOSO summary MUST 汇总这些字段

### Requirement: Radio-semantic variant matrix
系统 MUST 提供可配置的 V5 coarse prototype、V6 radio-semantic prototype、V6 radio condition off/on、V8 path-level physical prototype 和 full fine-tuning baseline 对比。工程配置 MUST 明确区分 radio method、path method 与当前 full fine-tuning baseline，不得静默改变已有 variant 语义。

#### Scenario: V5 与 V6 prototype 类型不同
- **WHEN** 用户运行 V5 coarse prototype baseline
- **THEN** 系统 MUST 使用 coarse/private prototype 配置
- **AND** summary MUST 记录 `proto_type=coarse` 或等价 baseline metadata

#### Scenario: V6 radio prototype baseline
- **WHEN** 用户运行 V6 radio-semantic method
- **THEN** 系统 MUST 使用 `proto_type=radio_semantic`
- **AND** summary MUST 记录 radio label mode、radio condition 是否启用和 source radio prototype path
- **AND** summary MUST 将其标记为 radio-semantic baseline，而不是 V8 path-level physical prototype

#### Scenario: V8 path prototype full method
- **WHEN** 用户运行 V8 path-level physical propagation prototype method
- **THEN** 系统 MUST 使用 `proto_type=path`
- **AND** summary MUST 记录 path semantic mode、path condition 是否启用、source path prototype path 和 path descriptor availability

#### Scenario: Full fine-tuning baseline 命名可追溯
- **WHEN** 用户运行 full fine-tuning baseline
- **THEN** summary MUST 将其标记为 full fine-tuning baseline
- **AND** 若工程中保留 `v6_full_finetune` 名称，summary MUST 不把它误标为 radio-semantic prototype method 或 path-level prototype method

