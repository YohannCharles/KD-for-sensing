## ADDED Requirements

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
