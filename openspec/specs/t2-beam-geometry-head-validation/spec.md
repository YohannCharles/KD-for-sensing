# t2-beam-geometry-head-validation Specification

## Purpose
定义 local/manual S1/T2 beam geometry、head 候选、固定 mask 评估与取消实验的证据边界，防止不完整多 seed 结果被误报为当前 claim。
## Requirements
### Requirement: T2 beam geometry 与 head 候选可隔离
系统 MUST 为 local/manual S1/T2 workflow 提供 circular-Gaussian prototype、linear-Gaussian prototype 和 classifier/prototype-loss-off 的显式可区分方法。linear-Gaussian MUST 保持 Gaussian soft target 且只取消首尾 wrap-around；classifier MUST 关闭 prototype alignment、modality prototype 和 prototype-margin router feature，但 MAY 为 checkpoint 兼容保留不参与 forward/loss/router 的 unused prototype bank。所有方法 MUST 复用相同的 S1 temporal pooling、router 主体与监督协议、split、sampler 和优化协议；classifier 只关闭不适用的 prototype-margin 输入。T2 与对应 S1 的唯一额外训练机制 MUST 为 confidence-gated temporal superset KL。

#### Scenario: linear-Gaussian 不退化为 one-hot
- **WHEN** launcher 生成 `S1-LG` 或 `T2-LG`
- **THEN** resolved config MUST 使用 Gaussian prototype target
- **AND** `beam_label_circular` 与 `circular_beam_distance` MUST 为 false
- **AND** prototype target、两条 supervised-router oracle loss 路径与 router eval diagnostics MUST 都使用 linear distance
- **AND** evaluation distance mode MUST 为 linear

#### Scenario: classifier 候选关闭 prototype 依赖
- **WHEN** launcher 生成 `S1-CLS` 或 `T2-CLS`
- **THEN** primary head MUST 为 classifier
- **AND** prototype alignment、modality prototype 与 prototype-margin router feature MUST 关闭
- **AND** 保留的 supervised-router oracle target MUST 使用 linear distance
- **AND** `T2-CLS` MUST 保留 confidence-gated temporal superset KL

### Requirement: 阶段式 GPU0-7 实验门禁
系统 MUST 支持将 current S1/T2 seeds2/3 与四个候选 seed1 组成八个独立作业，并在 GPU0-7 每卡最多运行一个训练进程。候选只有在匹配 S1 对照下通过五档 mean Top1、Drop80、Drop0 guardrail，且相对 current T2 的五档 mean Top1 下降不超过 0.005 后才 MUST 补齐 seeds2/3；未通过候选 MUST 记录 skipped reason。

#### Scenario: 第一轮八卡矩阵
- **WHEN** 用户按本 change 的第一轮命令生成 current multiseed 与 candidate screening 作业
- **THEN** GPU0-3 MUST 对应 S1/T2 current seeds2/3
- **AND** GPU4-7 MUST 对应 S1-LG、T2-LG、S1-CLS 和 T2-CLS seed1
- **AND** 每个作业 MUST 有独立 output、log、config 与 run status

#### Scenario: 失败候选不补多 seed
- **WHEN** 候选相对匹配 S1 的五档 mean Top1 或 Drop80 不为正，Drop0 下降超过 0.005，或候选 T2 相对 current T2 的五档 mean Top1 下降超过 0.005
- **THEN** workflow MUST 不为该候选生成 advancement training
- **AND** 本地报告 MUST 记录具体不合资格原因

### Requirement: 几何可审计的固定 mask 评估
所有 current 与候选 checkpoint MUST 使用同一 group-safe Scene31-34 final test split 和固定 temporal mask cache；validation 只允许作为独立 model-selection split 使用，不得替代 final test evidence。评估前 MUST 通过 sequence group、sample、历史帧和 target 帧 identity 的跨 split 重叠审计。评估产物 MUST 记录 split role、split strategy、group identity policy、identity audit summary、training geometry、prototype target geometry、router oracle geometry、head、prototype enabled、metric profile、DBA distance mode，以及 `mask_index`、`mask_type`、`mask_digest`、cache checksum/seed；同一 checkpoint MUST 支持在不重训的情况下显式复算 circular 或 linear distance metrics。不同 distance mode 的 ADBA、Within@3 和 MAE MUST 不混合聚合。

#### Scenario: 重复 cache entry 不作为独立 mask
- **WHEN** 两个 cache entry 的生成 type/index 不同但实际模态顺序与 `[5,4]` mask 相同
- **THEN** evaluator MUST 生成相同 `mask_digest`
- **AND** summary MUST 在严格配对后按 digest 折叠为一个 unique mask
- **AND** MUST 保留 source indices、source types 与 duplicate count
- **AND** candidate/final 主门禁 MUST 继续使用冻结的每 cell 4-entry protocol matrix 均值，去重结果只作为 paired-mask 证据

#### Scenario: 同 checkpoint 复算 linear metrics
- **WHEN** eval script 对 circular 训练 checkpoint 显式指定 linear distance mode
- **THEN** logits、Top1 和 Top3 MUST 与 config mode 评估一致
- **AND** ADBA、Within@3 和 MAE MUST 使用 linear class distance
- **AND** eval provenance MUST 记录 linear mode

### Requirement: T2 三随机种子主线判定
T2 只有在三个 seed 的五档 mean Top1、Drop0-60 mean 和 Drop80 平均均优于匹配 S1，至少 2/3 seed 的五档 mean delta 为正，且平均 Drop0 下降不超过 0.005 时，才 MUST 保留为实验主线。汇总 MUST 输出 seed summary、mean/std、逐 seed delta、按 `mask_index`、`mask_type` 与 `mask_digest` 去重且可追溯的同 mask paired delta、候选/final gate decision、Top3/ADBA/MAE 和实际可用的 gate diagnostics；重复 cache entry MUST 不解释为独立 mask 证据；缺 seed、缺 rate、身份冲突或不兼容 distance provenance MUST 产生 `unavailable` 而不是 pass；结果 MUST 保持 local experimental。

#### Scenario: 三 seed 通过主线门禁
- **WHEN** T2 与匹配 S1 的三个 seed 固定 mask 评估全部完成
- **THEN** summary MUST 计算三 seed mean/std 与逐 seed差值
- **AND** 必须按主线门禁给出 pass 或 fail 及证据
- **AND** 未经 claim review MUST 不更新正式 claim registry

### Requirement: 进度汇报反馈可追溯
项目 MUST 提供一份中文 Markdown，覆盖 PDF 每处红字的页码、原文、建议替换文案、当前实现或实验依据、叙事与版式调整、下一步矩阵、停止条件和实际执行结果。文档 MUST 区分已完成事实、机制推断、screening 结果和后续工作。

#### Scenario: PPT 修改建议落盘
- **WHEN** 本 change 完成
- **THEN** Markdown MUST 覆盖第 4、5、7、9、14、15 页全部红字
- **AND** MUST 说明 T2 与 S1 的真实增量、linear/circular 核验结果和 prototype 证据边界
- **AND** MUST 不把 MMW weather 或未完成可视化写成已完成结果
