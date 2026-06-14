# jepa-gps-shortcut-benchmark Specification

## Purpose
定义 JEPA GPS shortcut benchmark 的 manifest、模型可比性、扰动套件、输出表格和可视化消费契约，用于在固定 split、label space、seed 和输入扰动条件下比较 GPS-only、image+GPS baseline 与 JEPA query-pool 表征对 GPS shortcut collapse、image degradation 和 temporal async 的鲁棒性。
## Requirements
### Requirement: Benchmark manifest 契约
系统 MUST 提供 JEPA GPS shortcut benchmark manifest 契约，用于声明模型组、数据 split、评估协议、扰动套件、随机种子、指标、图表和输出目录。manifest MUST 可由包内 runner 或 JEPA visual analysis 入口读取，并 MUST 不要求新增仓库根旧脚本。

#### Scenario: 读取 benchmark manifest
- **WHEN** 用户运行 benchmark runner 并传入 manifest 路径
- **THEN** 系统 MUST 解析模型组、扰动套件、severity sweep、评估 split、指标列表和输出目录
- **AND** 系统 MUST 在输出目录写出解析后的 benchmark manifest 或 digest
- **AND** 系统 MUST 不修改输入 manifest、训练配置、checkpoint 或 split CSV

#### Scenario: manifest 拒绝未知模型引用
- **WHEN** manifest 引用不存在的模型配置、权重路径或未知模型组 key
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 包含不可解析的模型名和路径

### Requirement: 模型矩阵可比性
Benchmark MUST 支持 GPS-only neural、Camera AE + GPS、ResNet/Transformer image+GPS、JEPA mean pooling 和 JEPA GPS-query pooling 等模型组。系统 MUST 校验各模型使用可比较的 split、label space、metric profile、enabled modalities、normalization artifact 和 checkpoint provenance。

#### Scenario: 可比较模型组通过校验
- **WHEN** manifest 中多个模型声明同一 test split、同一 label space 和兼容 metric profile
- **THEN** benchmark MUST 对这些模型执行同一扰动矩阵
- **AND** 输出 metadata MUST 记录每个模型的 config、weights、modalities、checkpoint provenance 和样本数

#### Scenario: 不一致 split 被拒绝或隔离
- **WHEN** 两个模型的 test split、样本数、label space 或 metric profile 不一致
- **THEN** 系统 MUST 拒绝将它们写入同一可比较汇总表，或将其标记为不可比较分组
- **AND** 报告 MUST 明确记录不一致字段

### Requirement: GPS reliability collapse 扰动套件
Benchmark MUST 提供 deterministic GPS reliability collapse 扰动套件，至少覆盖 clean GPS、Gaussian jitter、cumulative drift、missing/dropout 和 GPS as distractor intervention。扰动 MUST 只作用于评估或分析 batch 的 GPS 输入，不得改写真实 dataset、split CSV 或训练统计。

#### Scenario: GPS noise sweep 可复现
- **WHEN** 用户配置 Gaussian GPS jitter 的多个强度和固定 seed
- **THEN** 系统 MUST 对每个模型使用相同样本、相同强度和相同 seed 生成 GPS 扰动
- **AND** 重复运行 MUST 产生相同的扰动参数、样本顺序和指标表

#### Scenario: GPS missing 不改变 batch 契约
- **WHEN** 用户配置 GPS dropout 或 missing GPS 条件
- **THEN** 系统 MUST 以模型可消费的 mask、zero-fill、learned missing token 或配置声明的方式表达缺失 GPS
- **AND** 输出 metadata MUST 记录缺失表达方式
- **AND** image 输入和 target label MUST 保持不变

#### Scenario: GPS distractor intervention 可审计
- **WHEN** 用户启用 GPS as distractor intervention
- **THEN** 系统 MUST 将 GPS 替换、错配或延迟为 manifest 声明的 misleading condition
- **AND** 系统 MUST 记录错配策略、seed、可用 sample pool 和是否保持 scene 内约束
- **AND** 报告 MUST 将该结果标记为 counterfactual intervention 而不是自然采样环境

### Requirement: Image physical degradation 扰动套件
Benchmark MUST 提供 deterministic image physical degradation 扰动套件，至少覆盖 fog/rain、night、occlusion 和 motion blur。每个扰动 MUST 保持 image batch shape、dtype 语义、normalization 口径和 sample metadata 可追踪。

#### Scenario: 图像退化 sweep 可复现
- **WHEN** 用户配置 fog/rain、night、occlusion 或 motion blur 的 severity sweep
- **THEN** 系统 MUST 对每个 severity 生成 deterministic 图像扰动
- **AND** 输出 metadata MUST 记录 degradation type、severity、seed、参数和作用帧范围

#### Scenario: 图像遮挡不影响 GPS 输入
- **WHEN** 用户启用 image occlusion sweep
- **THEN** 系统 MUST 只扰动 image batch
- **AND** GPS batch、beam target、sample id 和 split metadata MUST 保持不变

### Requirement: Asynchronous multimodal drift 扰动套件
Benchmark MUST 支持 Scenario C / Asynchronous Position Feedback，用于评估当前视觉 sensing 下 delayed、stale、low-rate 或 missing GPS 对 beam prediction 的影响。预测 target MUST 始终保持当前 beam label `y[t]` 和可选 power vector `P[t]`，不得随 GPS delay、stride、dropout 或 timestamp shift 一起移动。默认语义 MUST 只扰动 GPS 输入；image sequence MUST 保持当前对齐，除非 manifest 另行启用 image degradation suite。系统 MUST 记录 delay 单位、最大 delay、GPS stride、dropout probability、fallback/forward-fill 策略、source index 或 timestamp metadata，以及无法构造 delay 时的处理方式。

#### Scenario: GPS delay sweep
- **WHEN** 用户配置 GPS delay 范围为 0 到 5 秒或等价帧偏移
- **THEN** 系统 MUST 为每个 severity 构造 `G[k] -> G[max(0, k-delta)]` 或 timestamp 等价条件
- **AND** beam label、power target、sample id 和未启用退化的 image sequence MUST 保持不变
- **AND** 输出指标 MUST 按 delay severity 分组
- **AND** metadata MUST 记录时间到帧偏移的换算依据

#### Scenario: delay 不使用未来 GPS
- **WHEN** 用户启用固定 delay、随机 delay、低采样率 GPS 或 timestamp-based delay
- **THEN** 系统 MUST 保证任意输出时间步的 GPS 来源不晚于该 image time 减去声明 delay
- **AND** 若记录 source index 或 source timestamp，所有 source MUST 小于等于当前 index 或 timestamp
- **AND** 单元测试 MUST 能对 toy sequence 验证不存在未来 GPS 泄漏

#### Scenario: validity mask 和 delay metadata
- **WHEN** delay、低采样率、dropout 或历史不足导致 GPS stale 或 missing
- **THEN** 系统 MUST 输出 `gps_valid_mask`、`gps_delay_steps` 或 manifest 声明的等价字段
- **AND** forward-fill 或 clamp 得到的 stale GPS MUST 保留 invalid/stale 标记，不得被误标为 fresh GPS
- **AND** zero-fill、skip、clamp 或 forward-fill fallback MUST 写入 warnings 或 perturbation metadata

#### Scenario: 固定 Scenario C 评估设置
- **WHEN** manifest 引用 canonical Scenario C preset
- **THEN** 系统 MUST 支持 `C0_sync`、`C1_mild_stale`、`C2_low_rate`、`C3_random_async` 和 `C4_severe_async`
- **AND** `C0_sync` MUST 使用 `max_delay_steps=0`、`gps_stride=1`、`gps_dropout_prob=0.0`
- **AND** `C1_mild_stale` MUST 支持最多 1 step GPS delay、`gps_stride=1`、`gps_dropout_prob=0.0`
- **AND** `C2_low_rate` MUST 支持最多 2 step GPS delay、`gps_stride=2`、`gps_dropout_prob=0.1`
- **AND** `C3_random_async` MUST 支持最多 4 step GPS delay、`gps_stride` 从 `{1,2,3}` deterministic sampling、`gps_dropout_prob=0.3`
- **AND** `C4_severe_async` MUST 支持最多 4 step GPS delay、`gps_stride` 从 `{2,3,4}` deterministic sampling、`gps_dropout_prob=0.5`

#### Scenario: 低采样率 GPS forward-fill
- **WHEN** GPS stride 大于 1 且 `use_forward_fill=true`
- **THEN** 系统 MUST 用最近可用且非未来的 GPS 填充中间时间步
- **AND** 被 forward-fill 的时间步 MUST 通过 mask 或 delay metadata 标记为 stale 或 invalid
- **AND** 若 `use_forward_fill=false`，系统 MUST 使用 zero-fill 或 manifest 声明的 fallback，并保留 invalid mask

#### Scenario: timestamp-based delay
- **WHEN** batch metadata 提供 image timestamp 和 GPS timestamp
- **THEN** 系统 MUST 支持按 `gps_time <= image_time - delta_t` 选择最近 GPS measurement
- **AND** 若 timestamp 不可用，系统 MUST 降级为 frame-index delay 并记录 fallback

#### Scenario: delay 不足时记录降级
- **WHEN** 某个样本历史长度不足以构造指定 delay
- **THEN** 系统 MUST 使用 manifest 声明的 padding、clamp、skip 或 fallback 策略
- **AND** 输出 warnings MUST 记录受影响样本数

#### Scenario: 同 seed 下所有模型看到相同异步输入
- **WHEN** 多个模型在同一 split、suite、condition、severity 和 seed 下运行 Scenario C
- **THEN** benchmark MUST 对每个 sample 复用相同 delay、stride、dropout mask 和 fallback 结果
- **AND** 输出 metadata MUST 保存足以 replay corruption 的 seed、suite id、condition、sample id 和 corruption 参数

### Requirement: 训练与评估协议
Benchmark MUST 支持 evaluation-only、train-then-evaluate 和 reuse-existing-runs 三种协议。所有训练、评估和分析 Python 命令 MUST 使用 `conda run -n kd_mm_beam ...`，并 MUST 复用现有配置加载、模型 registry、dataset runtime、checkpoint loading 和 evaluation metrics。

#### Scenario: evaluation-only 协议
- **WHEN** manifest 为每个模型提供 config 和 weights
- **THEN** runner MUST 只执行只读评估和分析
- **AND** runner MUST 不启动训练、不修改 checkpoint、不修改训练 run 目录

#### Scenario: train-then-evaluate 协议
- **WHEN** manifest 声明某个模型需要先训练
- **THEN** runner MUST 通过现有训练入口或等价包内 API 执行训练
- **AND** 训练产物 MUST 写入 ignored 输出目录
- **AND** benchmark metadata MUST 记录训练命令、resolved config、run dir 和 selected checkpoint

#### Scenario: Scenario C 训练测试协议
- **WHEN** manifest 声明 Scenario C protocol A、B 或 C
- **THEN** Protocol A MUST 表示 clean training 后在 `C0_sync`、`C1_mild_stale`、`C2_low_rate`、`C3_random_async` 和 `C4_severe_async` 上测试
- **AND** Protocol B MUST 表示使用 `C0_sync`、`C1_mild_stale` 和 `C2_low_rate` 的 mild async mixture 训练，并在 C0 到 C4 全部设置上测试
- **AND** Protocol C MUST 表示使用 GPS dropout 和 image dropout 训练，并在 C0 到 C4 全部设置上测试
- **AND** 所有协议 MUST 保持相同 train/val/test split、label space 和 corruption seed 可追踪

### Requirement: Benchmark 指标和论文图产物
Benchmark MUST 输出结构化指标和论文图产物。指标 MUST 至少包含 clean 指标、每个扰动条件下的 Top-K、DBA 或当前 objective 正式指标、相对下降、collapse slope、area-under-robustness-curve 和可比较性 metadata。

#### Scenario: 写出鲁棒性汇总表
- **WHEN** benchmark 完成至少一个模型和一个扰动 suite
- **THEN** 输出目录 MUST 包含 `metrics_by_condition.csv` 或等价表格
- **AND** 输出目录 MUST 包含 `robustness_summary.csv` 或等价汇总
- **AND** 每行 MUST 记录 model、suite、condition、severity、seed、split、sample_count、primary metric 和 clean delta

#### Scenario: 导出论文曲线
- **WHEN** benchmark 启用 figure export
- **THEN** 系统 MUST 导出 GPS noise/dropout 曲线、image degradation 曲线或 temporal delay 曲线中已配置的图表
- **AND** 图表 MUST 标注模型名、split、样本数、metric、severity 单位和 seed 或 digest

### Requirement: Modality reliance 与反事实诊断
Benchmark MUST 支持 drop GPS、drop image、misleading GPS、GPS-only collapse slope 和可选 attention/gradient/ablation summary，用于分析模型是否依赖 GPS shortcut。系统 MUST 把这些诊断与任务性能指标分开记录。

#### Scenario: Drop GPS 反事实表
- **WHEN** benchmark 启用 drop GPS condition
- **THEN** 系统 MUST 为支持 GPS 的模型计算 clean 与 drop GPS 的指标差异
- **AND** 汇总表 MUST 记录 drop magnitude、missing expression 和模型是否仍可 forward

#### Scenario: attention 不可用时降级
- **WHEN** 某个模型不提供 attention 或 gradient diagnostics
- **THEN** 系统 MUST 跳过该模型的对应 reliance 图
- **AND** 系统 MUST 在 manifest、warnings 和 report 中记录 unavailable reason
- **AND** 其它 benchmark 指标 MUST 继续生成

### Requirement: Benchmark 复现和产物边界
Benchmark MUST 将所有新增输出写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定的本地产物目录。输出 MUST 包含命令、环境、manifest digest、git status 摘要、模型配置路径、checkpoint 路径、split metadata、扰动参数、随机种子、warnings 和文件清单。

#### Scenario: 写出 benchmark manifest
- **WHEN** benchmark 运行结束
- **THEN** 输出目录 MUST 包含 `benchmark_manifest.json` 或等价机器可读 manifest
- **AND** manifest MUST 记录输入 manifest digest、模型配置/权重、扰动 suite、seeds、split metadata、metric profile、输出文件清单和 warnings

#### Scenario: 本地产物不进入源码
- **WHEN** benchmark 生成图表、表格、cache、checkpoint 或报告
- **THEN** 这些文件 MUST 位于 ignored 本地产物目录
- **AND** OpenSpec、源码和文档 MUST 不要求提交真实数据、训练输出、checkpoint 或 cache

### Requirement: Benchmark 可测试性
系统 MUST 为 benchmark manifest schema、扰动 determinism、shape 保持、指标聚合和降级行为提供自动化测试。测试 MUST 使用 synthetic/mock batch 或小型 fixture，不得读取真实 `dataset/`。

#### Scenario: synthetic batch 扰动测试
- **WHEN** 单元测试使用同一 synthetic image/GPS batch、suite config 和 seed 调用 perturbation transform 两次
- **THEN** 两次输出 MUST 完全一致
- **AND** 输出 batch shape、target label 和 sample id MUST 与输入兼容

#### Scenario: manifest schema 测试
- **WHEN** 测试加载最小 benchmark manifest
- **THEN** schema validation MUST 成功
- **AND** 缺少必需字段或包含未知 suite type 时 MUST 报出清晰错误

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

### Requirement: Scenario D benchmark suite
JEPA GPS shortcut benchmark MUST 支持 Scenario D image observability suite。Suite MUST 复用 shared difficulty pipeline，且 MUST 能与 existing Scenario C async GPS suite 组合为 Cx-Dy matrix。

#### Scenario: manifest 引用 Scenario D suite
- **WHEN** benchmark manifest 声明 suite type `scenario_d_image_observability`
- **THEN** runner MUST 标准化 D-level condition、image operator 参数、seed 和 output artifact plan
- **AND** runner MUST 将 image corruption 委托给 shared difficulty operator
- **AND** runner MUST 不维护独立平行的 image corruption 实现

#### Scenario: Scenario C 与 D 联合执行
- **WHEN** manifest 声明 joint suite `scenario_c_x_d_image_observability`
- **THEN** runner MUST 对每个模型执行 Scenario C condition 与 Scenario D condition 的笛卡尔组合
- **AND** 每个 row MUST 记录 `gps_condition`、`image_condition`、C severity、D severity、seed 和 difficulty digest

### Requirement: Scenario D required model groups
Benchmark MUST 支持 Scenario D 指定的模型组：GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only 和 Image-JEPA+GPS。Runner MUST 将这些模型组映射到现有 config/weights/registry 语义，并 MUST 记录模型是否消费 image/GPS reliability metadata。

#### Scenario: required model group 校验
- **WHEN** manifest 声明 strict Scenario D evaluation
- **THEN** runner MUST 校验 required model groups 是否齐全，或在显式允许 partial run 时记录缺失模型组
- **AND** report MUST 区分 standard fusion、CNN/AE visual encoder、JEPA visual encoder 和 observability-aware fusion

#### Scenario: Image-JEPA only 不消费 GPS 输入
- **WHEN** model group 为 Image-JEPA only
- **THEN** runner MUST 仍按 Cx-Dy 条件记录 GPS condition metadata 以保持矩阵对齐
- **AND** 模型 forward MUST 不要求 GPS input tensor

### Requirement: Scenario D aggregation 和图表
Benchmark MUST 聚合 Scenario D matrix，并导出 Cx-Dy heatmap、robustness surface、phase transition、CNN vs JEPA crossing point 和 modality dominance 图表或表格。图表生成失败时，metrics CSV 和 manifest MUST 仍然写出，并记录 warning。

#### Scenario: 输出 Cx-Dy aggregation
- **WHEN** Scenario D matrix 完成至少一个模型
- **THEN** runner MUST 写出包含 model、gps_condition、image_condition、metric、sample_count、seed 和 clean delta 的 long-form CSV
- **AND** runner MUST 写出按模型排序的 heatmap NPY 或等价矩阵 artifact

#### Scenario: attention 不可用时 dominance 降级
- **WHEN** 某个模型不提供 attention 或 fusion weights
- **THEN** modality dominance ratio MUST 使用配置声明的 fallback 或跳过该模型
- **AND** warnings MUST 记录 unavailable reason

