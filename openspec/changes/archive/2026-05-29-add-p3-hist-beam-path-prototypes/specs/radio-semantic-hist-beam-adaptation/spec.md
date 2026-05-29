## MODIFIED Requirements

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
