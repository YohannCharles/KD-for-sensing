# modality-difficulty-pipeline Specification

## Purpose
定义统一的模态输入难度 profile、operator 注册、batch transform、metadata 和测试契约，用于在训练、验证、评估和 JEPA benchmark 中复用 GPS/image 等输入扰动，同时保证 target、sample id 和 split 语义不被改写。
## Requirements
### Requirement: 难度 profile schema 与解析
系统 MUST 提供统一的 modality difficulty profile schema，用于声明输入难度 profile、operator 序列、作用 stage/split、目标模态、condition、severity、seed、fallback 和 metadata 策略。schema MUST 支持 GPS、image 和未来可注册模态 operator，MUST 拒绝未知 operator、未知模态、空 operator 列表、非法 severity 和会移动 target 的配置。

#### Scenario: 解析 GPS 与 image 难度 profile
- **WHEN** 配置声明一个包含 GPS delay/dropout 和 image occlusion 的 difficulty profile
- **THEN** 系统 MUST 标准化 profile id、operator type、canonical modality key、severity、seed、stage/split 和 fallback
- **AND** resolved config 或 runtime metadata MUST 记录该 profile 的 digest

#### Scenario: 拒绝未知难度 operator
- **WHEN** 配置引用未注册的 difficulty operator `gps_magic_noise`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 包含未知 operator、registry 名称和可用 operator 列表

#### Scenario: 拒绝 target-shift 难度配置
- **WHEN** 配置尝试让 GPS delay 同步移动 `target_beam`、`beam_power`、sample id 或 split metadata
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指出 difficulty pipeline 只能扰动输入模态和输入相关 mask/metadata

### Requirement: 统一 batch difficulty transform
系统 MUST 提供统一 batch difficulty transform，用于在训练、验证、评估和 benchmark 中对 flat batch 输入应用难度 operator。transform MUST 默认 clone 或等价保护输入 batch，MUST 保持 batch shape、dtype 语义、target label、beam power、sample id 和 split metadata 不变，MUST 输出 operator metadata、warnings 和必要的 valid/stale/dropout/source 字段。

#### Scenario: synthetic batch 扰动保持 shape 与 target
- **WHEN** synthetic batch 包含 `gps`、`image`、`target_beam`、`beam_power` 和 `metadata.sample_id`
- **THEN** 应用任一 GPS 或 image difficulty profile 后，输入模态 shape 与 dtype MUST 仍可被当前 batch 准备流程消费
- **AND** `target_beam`、`beam_power`、sample id 和 split metadata MUST 与输入一致

#### Scenario: 同 seed 同样本扰动确定
- **WHEN** 使用同一 profile、condition、severity、seed、split 和 sample id 两次应用 difficulty transform
- **THEN** 两次输出的被扰动模态、mask、source index 和 metadata MUST 一致
- **AND** replay metadata MUST 足以解释 seed 派生依据

### Requirement: GPS 难度 operator
系统 MUST 提供可注册 GPS difficulty operators，至少覆盖 clean、Gaussian jitter、cumulative drift、missing/dropout、distractor、fixed/random delay、low-rate stride、forward-fill/zero-fill fallback 和 timestamp-based delay。GPS temporal/async operator MUST 保证每个输出时间步的 GPS 来源不晚于当前 image/input 时间步，MUST 输出 `gps_valid_mask`、`gps_stale_mask`、`gps_delay_steps`、`gps_source_index`、`gps_dropout_mask` 或等价字段。

#### Scenario: 固定 delay 不使用未来 GPS
- **WHEN** GPS sequence `[B,T,D]` 应用固定 `max_delay_steps=2`
- **THEN** 每个有效输出时间步的 `gps_source_index` MUST 小于等于当前 time index
- **AND** target label、power target 和未启用 image operator 的 image batch MUST 保持不变

#### Scenario: 低采样率 GPS 标记 stale
- **WHEN** GPS operator 使用 `gps_stride=2` 且启用 forward-fill
- **THEN** 中间时间步 MUST 使用最近可用且非未来的 GPS
- **AND** 被复用的历史 GPS MUST 通过 `gps_stale_mask` 或 `gps_delay_steps` 标记为 stale

#### Scenario: timestamp delay 降级可审计
- **WHEN** profile 请求 timestamp-based delay 但 batch metadata 缺少 timestamp
- **THEN** 系统 MUST 降级到 frame-index delay 或按配置拒绝运行
- **AND** warnings MUST 记录降级原因、受影响样本数和 fallback

### Requirement: Image 难度 operator
系统 MUST 提供可注册 image difficulty operators，至少覆盖 clean、fog/rain、night、occlusion 和 motion blur。image operator MUST 只扰动 image modality 对应 canonical batch key，MUST 保持 image batch shape、dtype 语义、normalization 口径和 sample metadata 可追踪，MUST 不改变 GPS、target label 或 split metadata。

#### Scenario: image occlusion 不影响 GPS
- **WHEN** batch 同时包含 image 与 GPS，profile 只启用 image occlusion
- **THEN** image batch MUST 被确定性遮挡
- **AND** GPS batch、GPS mask、target label、sample id 和 split metadata MUST 保持不变

#### Scenario: image physical degradation 记录参数
- **WHEN** profile 启用 fog/rain、night 或 motion blur severity sweep
- **THEN** 输出 metadata MUST 记录 operator type、severity、seed、resolved 参数和作用帧范围

### Requirement: 训练评估 benchmark 共享难度管线
训练、验证、评估和 benchmark MUST 共享同一 difficulty profile 解析和 batch transform。系统 MUST 支持按 stage/split 启用 profile，例如 clean training、mild async training、GPS/image dropout training、evaluation severity sweep 和 benchmark Scenario C。未配置 difficulty profile 时，现有训练和评估行为 MUST 保持 clean 输入语义。

#### Scenario: 只在 train split 应用 mild async profile
- **WHEN** 配置将 mild async GPS profile 限定到 `stage=train`
- **THEN** train batch MUST 应用该 profile
- **AND** validation/test batch MUST 保持 clean，除非它们显式声明 profile

#### Scenario: evaluation sweep 复用同一 operator
- **WHEN** evaluation 或 benchmark 声明同一 GPS delay operator 的多个 severity
- **THEN** 系统 MUST 复用同一 operator 实现生成各 severity 输入
- **AND** 输出指标 MUST 按 profile、condition、severity、seed 和 split 分组

### Requirement: 难度 metadata 与产物边界
系统 MUST 将 resolved difficulty profile、operator digest、seed、condition、severity、split/stage、warnings、replay 字段和输出文件清单写入 run metadata、benchmark manifest 或等价本地产物。难度管线生成的表格、图、cache 或 debug 输出 MUST 位于 ignored `outputs/`、`logs/` 或 manifest 指定目录。

#### Scenario: 训练 run 写出 difficulty metadata
- **WHEN** 训练配置启用任一 difficulty profile
- **THEN** `final_config.yaml`、runtime metadata 或等价 artifact MUST 记录 resolved profile、operator 列表、stage/split、seed 和 digest
- **AND** 真实数据、cache、checkpoint 和训练输出 MUST 不进入源码变更

#### Scenario: benchmark 写出 replay metadata
- **WHEN** benchmark 完成一个 difficulty suite
- **THEN** 输出 manifest MUST 记录 profile id、operator type、condition、severity、seed、sample id/source index metadata 和 warnings
- **AND** 这些 metadata MUST 足以重放对应扰动条件

### Requirement: 难度管线可测试性
系统 MUST 为 difficulty profile schema、operator registry、determinism、shape/dtype preservation、target preservation、GPS no-future-leak、metadata 写出和降级 warnings 提供自动化测试。测试 MUST 使用 synthetic/mock batch 或小 fixture，不得读取真实 `dataset/`。

#### Scenario: synthetic no-future-leak 测试
- **WHEN** 单元测试对 toy GPS sequence 应用 random async 或 low-rate profile
- **THEN** 断言 MUST 验证所有有效 `gps_source_index` 不大于当前 time index
- **AND** 同 seed 运行两次 MUST 生成相同 mask 和 source index

#### Scenario: import-lightness 测试
- **WHEN** 测试仅导入 difficulty schema、registry 或 config validation helper
- **THEN** 导入 MUST 不触发 dataset、model、diagnostics renderer、torchvision 权重或训练循环导入

### Requirement: Image observability transform
系统 MUST 提供 image observability transform，用于在统一 difficulty pipeline 中确定性调制 image modality。Transform MUST 支持 frame dropout、burst dropout、weather、blur、occlusion、low-light、seed、valid mask、observability score 和 replay metadata。

#### Scenario: ImageObservabilityTransform 参数
- **WHEN** 开发者构建 image observability transform
- **THEN** transform MUST 支持 `image_dropout_prob`、`image_burst_dropout_prob`、`max_burst_len`、`image_weather_severity`、`image_blur_prob`、`image_occlusion_prob`、`image_occlusion_ratio`、`image_lowlight_prob` 和 `seed`
- **AND** 参数 MUST 可由 difficulty profile/operator config 或 Scenario D preset 标准化生成

#### Scenario: corruption 与 missing 区分
- **WHEN** transform 只应用 weather、low-light、blur 或 occlusion
- **THEN** `image_valid_mask` MUST 保持有效，除非配置另行声明整帧不可用
- **AND** transform MUST 写出 corruption type、severity、frame range 和 operator parameters
- **AND** GPS、target label、sample id 和 split metadata MUST 保持不变

#### Scenario: dropout 生成 invalid mask
- **WHEN** transform 应用 frame dropout 或 burst missing
- **THEN** 系统 MUST zero-fill、mask-fill 或使用配置声明的 missing token 表达缺失 image
- **AND** 系统 MUST 写出 `image_valid_mask`、`image_dropout_mask` 或 `image_burst_dropout_mask`
- **AND** 缺失表达方式 MUST 写入 warnings 或 replay metadata

### Requirement: Image observability score
系统 MUST 计算 `image_observability_score`，用于表达当前 image 输入的可用性。Score MUST 由 dropout、blur、occlusion、low-light 和 weather severity 等输入退化因素确定，MUST 位于可解释范围内，并 MUST 不作为 target supervision。

#### Scenario: score 随退化降低
- **WHEN** image observability transform 应用更高 dropout、blur、occlusion 或 low-light severity
- **THEN** `image_observability_score` MUST 不高于 clean condition 的 score
- **AND** score metadata MUST 记录参与计算的 corruption factors

#### Scenario: score 可被 batch 和模型消费
- **WHEN** difficulty pipeline 输出 batch
- **THEN** batch MUST 包含 `image_observability_score` 或等价 metadata 字段
- **AND** 训练、评估或 benchmark runtime MUST 能将该字段传递给支持 observability-aware fusion 的模型

### Requirement: Scenario D preset 复用 difficulty pipeline
Scenario D 的 `D0` 到 `D7` preset MUST 通过 shared difficulty profile/operator registry 解析和执行。Benchmark、evaluation 和 train-time augmentation 使用相同 profile id、condition、severity、seed、split 和 sample id 时，MUST 产生一致的 image corruption、mask 和 replay metadata。

#### Scenario: 同 seed 可复现
- **WHEN** synthetic image batch 使用同一 Scenario D condition、seed 和 sample id 两次应用 transform
- **THEN** 两次输出的 image tensor、valid/dropout mask、observability score 和 metadata MUST 一致
- **AND** target label 和 sample id MUST 与输入一致

#### Scenario: unknown D-level 被拒绝
- **WHEN** profile 或 manifest 引用未知 image observability level `D9_magic`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 包含未知 condition 和可用 D-level 列表

### Requirement: Predictive robustness difficulty preset
系统 MUST 在 shared difficulty pipeline 中提供 Predictive Robustness preset，用于生成 history-aware image/GPS 输入扰动。Preset MUST 支持 canonical `P0-P5` condition，并 MUST 记录足以复现扰动的 metadata。

#### Scenario: 标准化 P-level condition
- **WHEN** profile 或 benchmark suite 引用 `predictive_jepa_robustness` preset
- **THEN** 系统 MUST 标准化 `P0_clean_current`、`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P3_plausible_wrong_gps_current_image`、`P4_joint_predictive_recovery` 和 `P5_novel_weather_history_available`
- **AND** 每个 condition MUST 映射到现有或新增 image/GPS difficulty operator 参数
- **AND** unknown P-level condition MUST 被拒绝，并在错误中列出可用 condition

#### Scenario: Predictive preset 保持 target 不变
- **WHEN** predictive robustness difficulty profile 应用于 batch
- **THEN** image/GPS tensor、mask、source index、observability score 或 replay metadata MAY 改变
- **AND** `target_beam`、`beam_power`、soft target、sample id 和 split metadata MUST 与输入保持一致

### Requirement: History-aware image missing 和 semantic occlusion
Predictive Robustness image operator MUST 支持当前帧缺失且历史可用、beam-relevant semantic occlusion 和 novel weather/history available condition。Operator MUST 输出 valid mask、observability score、history availability metadata 和 corruption parameters。

#### Scenario: 当前帧缺失但历史可用
- **WHEN** condition 为 `P1_current_frame_missing_history_available`
- **THEN** transform MUST 将当前预测时间步 image 表达为 missing/zero/mask token 或配置声明的 missing expression
- **AND** 历史帧 MUST 保持可用，metadata MUST 记录 history window 和 current frame missing mask

#### Scenario: 语义遮挡可复现
- **WHEN** condition 为 `P2_semantic_occlusion_history_available` 或 `P4_joint_predictive_recovery`
- **THEN** transform MUST 对当前帧应用 deterministic beam-relevant 或 proxy semantic occlusion
- **AND** replay metadata MUST 记录 occlusion ratio、region selection seed、frame range 和是否使用 proxy heuristic

### Requirement: Plausible wrong GPS 扰动
Predictive Robustness GPS operator MUST 支持 plausible wrong GPS，即用同 split 或同场景约束下的邻近但错误 GPS 替换当前 GPS，使其数值看起来可信但指向错误 beam 区域。该扰动 MUST 被标记为 counterfactual input intervention。

#### Scenario: 构造 plausible wrong GPS
- **WHEN** condition 为 `P3_plausible_wrong_gps_current_image` 或 `P4_joint_predictive_recovery`
- **THEN** 系统 MUST 替换或错配 GPS 输入，并记录 source sample、scene constraint、distance/beam offset criteria、seed 和 fallback
- **AND** 替换后的 GPS MUST 不改变当前样本 target label 或 sample id

#### Scenario: 无可用错配样本时降级
- **WHEN** plausible wrong GPS sample pool 不足
- **THEN** 系统 MUST 按配置 skip、fallback 到 deterministic jitter 或拒绝运行
- **AND** warnings MUST 记录 fallback reason 和 affected sample count

### Requirement: Predictive robustness determinism 和 no-future-leak
Predictive Robustness operators MUST 在同 profile、condition、seed、split 和 sample id 下确定性生成扰动，并 MUST 保证 temporal prediction 可用历史不包含未来信息。

#### Scenario: 同 seed 重放一致
- **WHEN** 单元测试对同一 synthetic batch 应用同一 predictive robustness profile 两次
- **THEN** 两次输出的 image/GPS tensors、masks、source indices、observability score 和 metadata MUST 一致

#### Scenario: 历史窗口不使用未来帧
- **WHEN** condition 声明 history window 用于 temporal prediction 或 history availability
- **THEN** 所有 source history index MUST 小于当前预测时间步
- **AND** 若历史不足，metadata MUST 记录不足并按配置 fallback

### Requirement: Visual-ambiguous hard negative condition
The modality difficulty pipeline MUST support a visual-ambiguous hard negative condition for GPS-query evaluation. This condition MUST preserve target labels and sample identity while selecting or marking peer examples whose visual context is similar but beam target differs by a configured margin.

#### Scenario: 构造视觉歧义 peer
- **WHEN** a difficulty profile enables visual-ambiguous hard negative selection
- **THEN** system MUST select peer samples using same split/scene constraints and configured visual similarity proxy or embedding source
- **AND** selected peers MUST satisfy configured target beam offset threshold unless fallback is recorded
- **AND** metadata MUST record source sample id、scene、similarity score、beam offset、seed and fallback reason

#### Scenario: 不改变监督 target
- **WHEN** visual-ambiguous hard negative condition is applied
- **THEN** `target_beam`、`beam_power`、sample id and split metadata MUST remain unchanged
- **AND** any peer feature substitution or metadata marking MUST be recorded as counterfactual input intervention

### Requirement: Beam-offset-constrained wrong GPS
The pipeline MUST support wrong-GPS replacement constrained by beam offset, so plausible wrong GPS interventions are strong enough to test GPS-query reliance.

#### Scenario: wrong GPS 满足 beam offset 下限
- **WHEN** profile enables beam-offset-constrained wrong GPS
- **THEN** replacement GPS MUST be selected from a peer sample whose target beam differs by at least the configured threshold
- **AND** metadata MUST record peer sample id、beam offset、GPS distance、selection pool size and scene/split constraint

#### Scenario: peer 不足时 fallback 可审计
- **WHEN** no peer satisfies the beam offset and scene/split constraints
- **THEN** system MUST use configured fallback, skip, or fail behavior
- **AND** warnings MUST record affected sample count、fallback mode、threshold and available pool size

### Requirement: Combined GPS-query advantage perturbations
The pipeline MUST support combined perturbations that pair GPS reliability degradation with image observability degradation for GPS-query advantage evaluation.

#### Scenario: CxD advantage condition 应用
- **WHEN** a profile requests `C3_random_async` or `C4_severe_async` combined with `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing` or `D7_joint_worst_case`
- **THEN** system MUST apply both GPS and image operators in a deterministic order
- **AND** metadata MUST include both GPS condition and image condition parameters, but model gate inputs MUST receive only continuous reliability fields and masks

#### Scenario: combined perturbation 保持 no-future-leak
- **WHEN** combined perturbation is applied to a temporal sequence
- **THEN** any history source used for temporal prediction or fallback MUST be strictly earlier than the prediction step
- **AND** replay metadata MUST expose history source ranges for audit

### Requirement: Advantage difficulty determinism
GPS-query advantage conditions MUST be deterministic under the same seed, split, sample id, condition id and operator parameters.

#### Scenario: 同 seed 重放一致
- **WHEN** tests apply the same advantage difficulty profile twice to the same synthetic batch
- **THEN** image/GPS tensors, masks, source indices, selected peer ids and replay metadata MUST match exactly

#### Scenario: 不同 seed 可改变 peer 但保留约束
- **WHEN** tests apply the same profile with a different seed
- **THEN** peer selection MAY differ
- **AND** all configured beam offset, scene/split and no-future-leak constraints MUST still hold

### Requirement: Difficulty perturbation cache provenance
The difficulty pipeline MAY be materialized into local cache artifacts for benchmark reuse. Cached artifacts MUST record enough provenance to verify that a loaded perturbed batch corresponds to the requested profile, condition, severity, split, seed and sample ids.

#### Scenario: cache payload 包含 replay metadata
- **WHEN** a perturbed batch is written to cache
- **THEN** payload metadata MUST include profile digest or suite identity, condition, severity, split, seed, sample ids, cache schema version and warnings
- **AND** payload MUST not contain modified target labels or source dataset files

#### Scenario: cache mismatch 被拒绝
- **WHEN** a cached perturbation payload is loaded for a different condition, severity, split, seed or sample id set
- **THEN** loader MUST reject the payload
- **AND** error message MUST include the mismatched cache path or key

### Requirement: Benchmark 复用统一 difficulty pipeline
JEPA GPS shortcut benchmark MUST 使用统一 modality difficulty pipeline 解析和应用 perturbation suites。现有 manifest 中的 GPS jitter、drift、missing/dropout、distractor、image degradation、temporal delay、sampling-rate mismatch 和 Scenario C suite type MUST 继续可解析，但 runner 内部 MUST 委托 shared difficulty operator，而不是维护独立实现分支。

#### Scenario: 旧 perturbation suite 映射到 difficulty operator
- **WHEN** benchmark manifest 使用现有 `gps_gaussian_jitter`、`image_occlusion` 或 `temporal_delay` suite type
- **THEN** runner MUST 将 suite 标准化为对应 difficulty profile/operator
- **AND** 输出 `metrics_by_condition.csv`、`robustness_summary.csv` 和 benchmark manifest 的核心列 MUST 保持兼容

#### Scenario: Scenario C preset 使用 shared GPS async operator
- **WHEN** benchmark manifest 引用 canonical Scenario C preset
- **THEN** runner MUST 通过 shared GPS async operator 构造 `C0_sync` 到 `C4_severe_async`
- **AND** metadata MUST 继续记录 max delay、GPS stride、dropout probability、fallback、source index 或等价 replay 字段

#### Scenario: benchmark 和 evaluation 使用相同扰动
- **WHEN** benchmark 与 evaluation 配置使用相同 profile id、operator、condition、severity、seed、split 和 sample id
- **THEN** 二者应用到同一 synthetic batch 时 MUST 产生一致的扰动输入、mask 和 warnings

### Requirement: Benchmark 输出 difficulty provenance
Benchmark 输出 MUST 记录 shared difficulty pipeline provenance，包括 profile id、operator registry name、resolved operator parameters、profile digest、seed 派生字段、stage、split 和 replay metadata。该 provenance MUST 与模型 comparability metadata 分开记录，避免把输入难度误当成模型结构差异。

#### Scenario: manifest 记录 difficulty provenance
- **WHEN** benchmark 完成一个 difficulty suite
- **THEN** `benchmark_manifest.json` 或等价输出 MUST 包含 difficulty profile digest、operator 列表、condition/severity、seed 和 warnings
- **AND** 模型 config、checkpoint provenance、split metadata 与 difficulty provenance MUST 分字段记录

#### Scenario: strict comparability 允许同一 difficulty profile
- **WHEN** 多个模型在同一 split、label space 和同一 difficulty profile digest 下评估
- **THEN** comparability 校验 MUST 不因共享 difficulty metadata 而失败
- **AND** 若模型使用不同 difficulty profile digest，系统 MUST 拒绝写入同一严格可比较汇总或标记为不可比较

### Requirement: Difficulty operator 注册表
项目 MUST 提供 difficulty operator 注册边界，用于按字符串名称注册、查询和构建 GPS、image 和未来模态输入难度 operator。该注册边界 MAY 复用现有 `Registry` 实现或新增窄 registry，但 MUST 保持轻量导入，不得在导入 registry 时 eager import dataset、model、diagnostics renderer、training loop 或大型视觉依赖。

#### Scenario: 按名称构建 GPS delay operator
- **WHEN** 配置指定 difficulty operator `gps_temporal_delay` 及其参数
- **THEN** 系统 MUST 通过 difficulty operator registry 构建该 operator
- **AND** 训练、评估和 benchmark MUST 能复用同一注册名

#### Scenario: 轻量导入 difficulty registry
- **WHEN** 开发者执行 `import kd_sensing.registries` 或导入 difficulty registry 窄模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入默认 dataset、model、diagnostics renderer、torchvision 权重接口或训练循环

#### Scenario: 未知 difficulty operator 错误可诊断
- **WHEN** 配置引用未注册 difficulty operator
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含 registry 名称、请求 operator 和可用 operator 列表

### Requirement: 默认 difficulty operator 显式注册
内置 difficulty operators MUST 通过显式默认组件导入或 difficulty 专用默认注册函数完成注册。构建流程在解析或应用 difficulty profile 前 MUST 触发该注册动作；仅导入 registry 对象 MUST 不自动注册所有重依赖 operator。

#### Scenario: 构建前导入默认 difficulty operators
- **WHEN** 配置加载或 benchmark runner 需要解析内置 GPS/image difficulty profile
- **THEN** 构建流程 MUST 先触发默认 difficulty operator 注册
- **AND** registry MUST 包含 GPS noise、GPS async、image degradation 等内置注册名

#### Scenario: 自定义 difficulty operator 可插拔
- **WHEN** 开发者在自定义模块中注册新的 image difficulty operator 并在配置中引用
- **THEN** 系统 MUST 能在该模块被显式导入后解析并构建该 operator
- **AND** 训练和 benchmark 主循环 MUST 不需要为该 operator 增加专用分支

### Requirement: Difficulty 配置解析与校验
配置加载流程 MUST 支持解析 top-level 或等价位置的 difficulty profiles，并在实体 YAML、virtual canonical 配置和命令行覆盖之后执行标准化与校验。解析结果 MUST 包含 profile id、operator list、stage/split selector、condition、severity、seed、affected modalities、fallback 和 digest。未知 operator、未知模态、非法 stage/split、非法 severity 或 target-shift 配置 MUST 被拒绝。

#### Scenario: 实体配置解析 difficulty profiles
- **WHEN** 用户加载包含 difficulty profiles 的实体 YAML
- **THEN** 配置加载流程 MUST 在 defaults、overlay 和命令行覆盖后标准化 difficulty 配置
- **AND** `final_config.yaml` 或 resolved config MUST 记录标准化后的 profile 和 digest

#### Scenario: 命令行覆盖 difficulty severity
- **WHEN** 用户通过命令行覆盖某个 difficulty profile 的 severity
- **THEN** 覆盖 MUST 在 difficulty validation 前生效
- **AND** resolved profile digest MUST 反映覆盖后的参数

#### Scenario: 非法 stage 被拒绝
- **WHEN** 配置声明 difficulty stage `preprocess_dataset_files`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出允许的 stage，例如 train、validation、test、evaluation 和 benchmark

### Requirement: Difficulty overlay recipe
canonical/virtual config recipe MAY 生成当前支持的 difficulty overlay，例如 clean、GPS mild async、GPS severe async、GPS/image dropout training、image hard degradation 和 benchmark sweep。recipe 生成的 difficulty 配置 MUST 与实体 YAML 使用同一标准化、validation 和 digest 流程。已退役的旧 KD、G2D、CRAF、MARF 或 image motion profile MUST 不得通过 difficulty overlay 恢复。

#### Scenario: 生成 mild async training overlay
- **WHEN** 用户加载声明支持的 mild async training overlay 配置
- **THEN** 系统 MUST 生成 train-stage GPS async difficulty profile
- **AND** 其它 supervised/fusion 配置语义 MUST 继续由当前 canonical recipe 决定

#### Scenario: difficulty overlay 不恢复 image motion profile
- **WHEN** 用户加载 image degradation difficulty overlay
- **THEN** image modality input profile MUST 仍解析为当前 RGB/ImageNet profile
- **AND** 系统 MUST 不生成或接受已删除的 `motion_mask` image profile、image motion cache 或 motion encoder

### Requirement: Difficulty 解析产物可比较
系统 MUST 为 difficulty profiles 生成稳定 digest，用于 run metadata、benchmark comparability 和论文图表分组。digest MUST 基于标准化后的 operator、condition、severity、stage/split、seed 和 fallback，而不是用户 YAML 字段顺序。

#### Scenario: 字段顺序不同 digest 相同
- **WHEN** 两个配置声明语义相同但 YAML 字段顺序不同的 difficulty profile
- **THEN** 标准化后的 digest MUST 相同
- **AND** benchmark comparability MAY 将它们视为同一 difficulty condition

#### Scenario: severity 改变 digest 改变
- **WHEN** 两个 profile 仅 severity 不同
- **THEN** digest MUST 不同
- **AND** 输出指标 MUST 能按不同 severity 分组

### Requirement: Missing-modality stress presets
Difficulty pipeline MUST 支持 missing-modality stress suite 所需的 canonical presets。Presets MUST 标准化 image/GPS/radar/LiDAR/mmWave 缺失、不可用、噪声和异步条件，并记录 replay metadata。

#### Scenario: 标准化 missing stress preset
- **WHEN** profile 引用 `missing_modality_stress` preset
- **THEN** 系统 MUST 标准化 clean、single missing、multi missing、only modality、random missing 和 unavailable modality 条件
- **AND** unknown condition MUST 被拒绝并列出可用 condition

#### Scenario: target 不变
- **WHEN** missing-modality stress preset 应用于 batch
- **THEN** 输入模态 tensor、valid mask、missing mask 或 reliability metadata MAY 改变
- **AND** `target_beam`、`beam_power`、soft target、sample id 和 split metadata MUST 保持不变

#### Scenario: replay metadata
- **WHEN** stress preset 完成一次 batch transform
- **THEN** metadata MUST 记录 profile id、condition、severity、operator params、seed、split、sample ids、fallback count 和 warnings

### Requirement: Sensing modality unavailable operators
Difficulty pipeline MUST 为 radar、LiDAR 和 mmWave 提供 unavailable/missing 表达，供缺失模态 stress suite 使用。

#### Scenario: radar unavailable
- **WHEN** profile 将 radar 标记为 unavailable
- **THEN** 系统 MUST 通过 zero-fill、mask token 或 valid mask false 表达 radar 不可用
- **AND** metadata MUST 记录 fallback 表达方式和 affected sample count

#### Scenario: LiDAR or mmWave unavailable
- **WHEN** profile 将 LiDAR 或 mmWave 标记为 unavailable
- **THEN** 系统 MUST 保持对应 tensor shape 可被模型消费
- **AND** 目标标签、GPS、image 和 split metadata MUST 不变

### Requirement: Difficulty pipeline 内部依赖必须走 owner module
模态 difficulty pipeline 的内部调用方 MUST 直接依赖 schema、pipeline、operator registry 或 concrete operator owner module。`kd_sensing.data.difficulty` 子包根只可作为轻量 package marker 或明确外部 shim，不得承载内部 re-export barrel。

#### Scenario: schema 和 pipeline 显式导入
- **WHEN** engine、diagnostics、config、tests 或其它内部模块需要 difficulty profile、operator plan、pipeline builder 或 schema 类型
- **THEN** 它们 MUST 从 `kd_sensing.data.difficulty.schema` 或 `kd_sensing.data.difficulty.pipeline` 导入
- **AND** 它们 MUST 不从 `kd_sensing.data.difficulty` package root 获取这些符号

#### Scenario: operator 注册包例外保留
- **WHEN** default component import 流程需要注册 built-in difficulty operators
- **THEN** 它 MAY 导入 `kd_sensing.data.difficulty.operators` 以触发注册 side effect
- **AND** 该例外 MUST 不允许 `kd_sensing.data.difficulty` package root eager import operators 或重依赖

#### Scenario: difficulty package marker 保持轻量
- **WHEN** 后续 change 修改 `src/kd_sensing/data/difficulty/__init__.py`
- **THEN** 该文件 MUST 保持轻量 marker 或明确登记的 public shim
- **AND** 它 MUST 不为了减少内部 import 行数而重新 re-export schema、pipeline 或 operator 符号

