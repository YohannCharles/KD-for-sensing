# hist-beam-cross-scene-adaptation Specification

## Purpose
定义 HiST-Beam 跨场景自适应方法的模型变体、层次化 beam label、shared/private 表征、adapter/prototype 适配、训练诊断和评估输出契约，确保快速验证中的 source-only、adapter、adapter+prototype 与 full fine-tuning baseline 可配置、可复现并能被 LOSO workflow 汇总比较。
## Requirements
### Requirement: HiST-Beam 模型变体配置
系统 MUST 提供可通过配置和模型注册表构建的 HiST-Beam fusion 模型能力，用于 DeepSense6G 和 MMW 跨场景快速验证。配置 MUST 能选择 flat source-only、hierarchical source-only、shared-private、decoupled shared-private、adapter-only、adapter+coarse prototype、adapter+radio-semantic prototype、adapter+path-level physical prototype、shared physical private residual 和 full fine-tuning baseline 变体，并 MUST 默认保持既有 DeepSense6G `image`、`radar`、`gps` 三模态快速验证兼容。

#### Scenario: 构建 flat source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v0_flat` 或等价 flat 模式
- **THEN** 系统 MUST 构建普通 64 类 beam classifier
- **AND** 模型输出 MUST 继续兼容现有 beam Top-K 评估流程

#### Scenario: 构建 hierarchical source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v1_hierarchical`
- **THEN** 系统 MUST 构建 coarse head 和 fine head
- **AND** 系统 MUST 不启用 shared/private 解耦 loss 或 target adapter

#### Scenario: 构建 shared-private 解耦变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v3_decoupled`
- **THEN** 模型 MUST 产生 shared representation、private representation、coarse logits、fine logits 和 beam-level prediction
- **AND** 训练 MUST 能启用 orthogonality、shared scene confusion 和 private scene preservation loss

#### Scenario: 构建 adapter 和 full fine-tuning 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v4_adapter`、`v5_adapter_proto`、`v6_radio_proto`、`v8_path_proto` 或 `v6_full_finetune`
- **THEN** 系统 MUST 从 source checkpoint 初始化 target adaptation run
- **AND** 系统 MUST 按变体选择 adapter 训练、coarse/radio/path prototype adaptation 或全量 fine-tuning 策略
- **AND** 若工程继续保留 `v6_full_finetune` 配置名，summary MUST 将其标记为 V7 full fine-tuning baseline 或等价 full fine-tuning baseline metadata

#### Scenario: 构建 V6 radio-semantic prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v6_radio_proto`
- **THEN** 系统 MUST 使用 beam_power 派生的 radio-semantic label/prototype 作为 V6 baseline
- **AND** 系统 MUST 不把 V6 radio prototype 静默标记为 V8 path-level physical prototype

#### Scenario: 构建 V8 path-level physical prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v8_path_proto` 或等价 P3-HiST-Beam 模式
- **THEN** 模型 MUST 产生 beam_logits、path_logits、shared representation `c` 和 private representation `s`
- **AND** target adaptation MUST 支持 `proto_type=path`
- **AND** path prototype MUST 作为 semantic anchor 或 condition，而不是直接预测 beam

#### Scenario: 构建 V7 shared physical private residual 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v7_shared_physical_private_residual`
- **THEN** 系统 MUST 构建 shared beam head、physical beamspace head、private adapter、private residual head 和 residual gate
- **AND** 模型 MUST 输出 `logits_shared`、`logits_final`、`delta_logits_private`、`alpha`、`pred_beamspace_power`、shared representation 和 private representation
- **AND** `logits` 与 `beam_logits` MUST 指向 `logits_final` 以保持现有评估入口兼容
- **AND** v7 默认 MUST NOT 启用 history-anchor 或读取历史 beam label

### Requirement: 层次化 beam label 与输出契约
HiST-Beam MUST 支持将 64 类 beam label 拆分为 coarse group 和 fine offset。`group_size` MUST 可配置，快速验证默认值 MUST 为 8；当 `num_classes=64` 且 `group_size=8` 时，coarse group 数 MUST 为 8。

#### Scenario: 生成 coarse 和 fine label
- **WHEN** 输入 beam label 为合法的 64 类整数，且 `group_size=8`
- **THEN** 系统 MUST 将 coarse label 计算为 `beam // 8`
- **AND** 系统 MUST 将 fine offset 计算为 `beam % 8`

#### Scenario: 拒绝不可整除 group size
- **WHEN** 用户配置的 `num_classes` 不能被 `group_size` 整除
- **THEN** 系统 MUST 拒绝构建 HiST-Beam 配置
- **AND** 错误信息 MUST 包含 `num_classes`、`group_size` 和可执行的修复提示

#### Scenario: 输出 beam-level prediction
- **WHEN** hierarchical head 完成 forward
- **THEN** 模型 MUST 输出 coarse logits、fine logits 和 beam-level logits 或 log probabilities
- **AND** beam-level 输出 MUST 能用于 Top-1、Top-3 和 Top-5 指标

#### Scenario: 保持 horizon 维兼容
- **WHEN** 配置 `model.num_pred` 大于 1
- **THEN** HiST-Beam 输出 MUST 保持 `[B, H, C]` 的 beam-level 形状
- **AND** coarse/fine diagnostics MUST 与同一 horizon 对齐

### Requirement: Shared-private 表示解耦
HiST-Beam shared branch MUST 主要服务 coarse semantics，private branch MUST 主要服务 scene-specific refinement。启用 shared-private 时，coarse head MUST 只读取 shared representation，fine head MUST 读取 shared 与 private 的组合。

#### Scenario: coarse head 不读取 private representation
- **WHEN** 模型启用 shared-private hierarchical 模式
- **THEN** coarse head 的输入 MUST 只来自 shared representation
- **AND** private representation 的变化 MUST 不直接作为 coarse head 输入

#### Scenario: private branch 保留 scene 信息
- **WHEN** 配置启用 private scene preservation loss
- **THEN** 模型 MUST 输出 private scene classifier logits
- **AND** private branch 参数 MUST 接收该 loss 的梯度，除非配置显式关闭该 loss

#### Scenario: shared branch 使用 scene confusion
- **WHEN** 配置启用 shared scene confusion loss
- **THEN** 模型 MUST 输出 shared scene classifier logits
- **AND** shared branch MUST 通过 GRL 或等价反向机制降低 scene 可辨识性

### Requirement: HiST-Beam 训练 loss
系统 MUST 在显式启用 HiST-Beam 时计算层次化 loss、flat auxiliary loss、orthogonality loss、shared scene confusion loss 和 private scene preservation loss。普通非 HiST 配置 MUST 不受这些 loss 影响。

#### Scenario: 计算 hierarchical loss
- **WHEN** 训练 hierarchical 变体
- **THEN** 系统 MUST 对 coarse logits 计算 coarse CE
- **AND** 系统 MUST 只在真实 coarse group 对应的 fine logits 上计算 fine CE
- **AND** 系统 MUST 按配置权重合成 hierarchical loss

#### Scenario: 计算 flat auxiliary loss
- **WHEN** 配置 `lambda_flat` 大于 0
- **THEN** 系统 MUST 从 beam-level 输出计算 beam class 辅助 loss
- **AND** 该 loss MUST 参与总 loss 以约束最终 beam prediction

#### Scenario: HiST loss 不影响普通模型
- **WHEN** 用户运行非 HiST-Beam 模型或未启用 HiST loss 的配置
- **THEN** 训练流程 MUST 使用既有 beam loss 语义
- **AND** 系统 MUST 不要求模型输出 coarse/fine/shared/private diagnostics

### Requirement: Source prototype artifact
完成 source training 后，系统 MUST 能基于 source train split 生成 coarse prototype artifact。artifact MUST 至少包含 shared prototype、private prototype、每个 coarse group 的样本计数和生成配置 metadata。

#### Scenario: 保存 source prototypes
- **WHEN** HiST-Beam source training 完成且配置启用 prototype 保存
- **THEN** 系统 MUST 在运行目录保存 prototype artifact
- **AND** artifact MUST 记录 `group_size`、`num_groups`、source scenes、target scene、seed 和样本计数

#### Scenario: 空 group prototype 可诊断
- **WHEN** 某个 coarse group 在 source train split 中没有样本
- **THEN** prototype artifact MUST 记录该 group 的 count 为 0
- **AND** target adaptation MUST 不把该 group 用作高置信 prototype 目标

### Requirement: Target adapter adaptation
系统 MUST 支持从 source checkpoint 启动 target adaptation。Adapter-only 和 Adapter+Prototype 变体 MUST 冻结 source backbone 和 shared/coarse 主干，只训练配置允许的少量参数；full fine-tuning baseline MUST 能在相同 target labeled samples 上更新全部参数。

#### Scenario: Adapter 训练参数受限
- **WHEN** 用户运行 adapter-only 或 adapter+prototype adaptation
- **THEN** 系统 MUST 冻结 image/radar/gps encoder、fusion module、shared branch、coarse head 和原始 private branch
- **AND** 系统 MUST 只训练 private adapter、允许的 fine head 参数、可选 LayerNorm affine 参数和配置允许的 prototype 参数

#### Scenario: Adapter 初始等价 source model
- **WHEN** private adapter 初始化完成且尚未训练
- **THEN** adapter 输出 MUST 与未启用 adapter 的 private representation 等价或数值上等价
- **AND** source-only evaluation MUST 不因创建 adapter 模块而改变预测

#### Scenario: Full fine-tuning 更新全部参数
- **WHEN** 用户运行 full fine-tuning baseline
- **THEN** 系统 MUST 允许全部模型参数参与训练
- **AND** 训练日志 MUST 记录 trainable parameter ratio 为 100% 或等价全量比例

### Requirement: 无标签与半监督 target adaptation
系统 MUST 支持 label budget 为 0 的无标签 target adaptation，以及 label budget 大于 0 时 labeled target loss 与 unlabeled target loss 的组合。无标签 loss MUST 只使用 `target_adapt` split，不得读取 `target_test` 标签或样本。

#### Scenario: 0-label adaptation 使用无标签目标
- **WHEN** `label_budget=0` 且配置启用 prototype alignment
- **THEN** 系统 MUST 使用 entropy minimization 和高置信 prototype consistency 进行 adaptation
- **AND** 系统 MUST 不读取 target label 作为 supervised loss

#### Scenario: Few-shot adaptation 合成监督与无监督 loss
- **WHEN** `label_budget` 大于 0 且 target_adapt 中存在未标注样本
- **THEN** 系统 MUST 对 labeled subset 计算 supervised hierarchical/flat loss
- **AND** 系统 MUST 按配置权重对 unlabeled subset 计算无监督 loss

#### Scenario: 低置信 prototype 样本被忽略
- **WHEN** target 样本到 source prototype 的最大 soft assignment 低于 confidence threshold
- **THEN** 系统 MUST 不对该样本计算 prototype consistency loss
- **AND** adaptation metrics MUST 记录 prototype coverage 或等价可诊断统计

### Requirement: HiST-Beam 指标与预测产物
HiST-Beam evaluation 和 adaptation MUST 输出 Top-1、Top-3、Top-5、coarse group accuracy、fine offset accuracy、trainable parameter ratio 和 adaptation time。若样本提供 beam power vector，系统 MUST 输出 normalized received power 和 beam power loss dB；若没有 power vector，系统 MUST 明确跳过 power 指标。启用 path-level prototype 或 path head 时，系统 MUST 额外输出 path semantic accuracy、path descriptor regression MSE、prototype assignment confidence、prototype coverage per class 和 source-target path class histogram，或记录不可用原因。

#### Scenario: 输出 coarse 和 fine 指标
- **WHEN** 评估 HiST-Beam hierarchical 变体
- **THEN** metrics MUST 包含 coarse group accuracy
- **AND** metrics MUST 包含 fine offset accuracy 或在 flat 变体中明确标记该指标不可用

#### Scenario: 输出 adaptation 效率指标
- **WHEN** target adaptation 完成
- **THEN** metrics MUST 包含 trainable parameter count、total parameter count、trainable parameter ratio 和 adaptation time
- **AND** adapter 变体的 trainable parameter ratio MUST 可与 full fine-tuning baseline 横向比较

#### Scenario: 保存 test predictions
- **WHEN** source-only evaluation 或 target adaptation evaluation 完成
- **THEN** 系统 MUST 保存 target_test predictions 文件
- **AND** predictions MUST 至少包含 sample id、scene、true beam、predicted beam、top-k predictions、coarse true/pred 和当前变体 metadata
- **AND** 若 path diagnostics 可用，predictions MUST 包含 path true/pred、assignment confidence 或 path unavailable reason

#### Scenario: 缺失 beam power 时不伪造指标
- **WHEN** target_test 样本不包含 beam power vector
- **THEN** 系统 MUST 不输出虚假的 power gain 或 power loss 指标
- **AND** metrics MUST 记录 power metrics unavailable 的原因

#### Scenario: 输出 path prototype 诊断
- **WHEN** 评估 V8 path-level physical prototype 变体
- **THEN** metrics MUST 包含 path semantic accuracy、prototype assignment confidence 和 prototype coverage per class，或记录这些字段不可用的原因
- **AND** summary MUST 能与 V5 coarse prototype、V6 radio-semantic prototype 和 full fine-tuning baseline 横向比较

### Requirement: HiST-Beam execute run 产物
HiST-Beam quick validation 的每个 source-only evaluation 和 adapted evaluation run MUST 输出可追踪产物。产物 MUST 至少包含 `metrics.json`、target_test predictions、配置快照、fold/split/sampling metadata 和当前 variant metadata。

#### Scenario: source-only evaluation 写出标准产物
- **WHEN** execute runner 完成 `v0_flat` 或 `v3_decoupled` 的 source-only target_test evaluation
- **THEN** run directory MUST 包含 `metrics.json`
- **AND** run directory MUST 包含 target_test predictions
- **AND** artifact metadata MUST 记录 target scene、source scenes、variant、budget、seed 和 source checkpoint path

#### Scenario: adapted evaluation 写出标准产物
- **WHEN** execute runner 完成 `v4_adapter`、`v5_adapter_proto` 或 `v6_full_finetune` 的 adapted target_test evaluation
- **THEN** run directory MUST 包含 `metrics.json`
- **AND** run directory MUST 包含 target_test predictions
- **AND** artifact metadata MUST 记录 adaptation checkpoint path、source checkpoint path 和 adaptation strategy

#### Scenario: predictions 包含对比所需字段
- **WHEN** 系统写出 HiST-Beam target_test predictions
- **THEN** predictions MUST 至少包含 sample id、scene、true beam、predicted beam、top-k predictions、coarse true/pred、fine true/pred 和 variant metadata
- **AND** predictions MUST 标明样本来自 `target_test`

### Requirement: Adaptation 效率指标
HiST-Beam adaptation run MUST 记录 trainable parameter count、total parameter count、trainable ratio、adaptation time 和 prototype coverage 或不可用原因。这些指标 MUST 写入 run-level `metrics.json` 或 run metadata，并 MUST 被 LOSO summary 汇总。

#### Scenario: adapter run 记录 trainable ratio
- **WHEN** 系统执行 `v4_adapter` 或 `v5_adapter_proto` adaptation
- **THEN** metrics 或 metadata MUST 记录 trainable parameter count、total parameter count 和 trainable ratio
- **AND** trainable ratio MUST 反映实际参与优化的参数集合

#### Scenario: full fine-tuning run 记录全量参数比例
- **WHEN** 系统执行 `v6_full_finetune` adaptation
- **THEN** metrics 或 metadata MUST 记录 trainable parameter count、total parameter count 和 trainable ratio
- **AND** trainable ratio MUST 表示全部或等价全量参数参与训练

#### Scenario: adaptation time 可横向比较
- **WHEN** adaptation stage 完成
- **THEN** metrics 或 metadata MUST 记录 total adaptation time
- **AND** 若可获得 epoch 信息，系统 MUST 记录 adapt time per epoch

#### Scenario: prototype coverage 不可用时说明原因
- **WHEN** variant 未启用 prototype alignment 或 prototype artifact 缺失
- **THEN** 系统 MUST 将 prototype coverage 标记为不可用
- **AND** 系统 MUST 记录不可用原因，不得用 `0` 伪造 coverage

#### Scenario: few-shot 标签从 DeepSense6G beam power 路径解析
- **WHEN** target adapt CSV 的 `future_beamN` 或 `beamN` 字段是 beam-power 文件路径而非整数标签
- **THEN** few-shot sampler MUST 读取该 power vector 并使用 `argmax` 作为 beam label
- **AND** sampler MUST 优先使用 `future_beam_labelN` 或 `beam_labelN` 显式标签列
- **AND** sampling manifest MUST 记录 labeled sample 的 beam、coarse group 和 label source

#### Scenario: quick validation adaptation 超参显式可见
- **WHEN** 用户使用完整 `quick_validation` 配置执行 target adaptation
- **THEN** 配置 MUST 显式声明 adaptation epochs、entropy weight 和 prototype weight
- **AND** 0-label adaptation MUST NOT 因缺少默认权重而静默跳过所有有效更新

### Requirement: Quick validation 对比结论
系统 MUST 基于 quick validation summary 输出机器可读的快速验证结论。结论 MUST 比较 adapter/prototype variants 相对 source-only 和 full fine-tuning baseline 的效果与效率，并 MUST 明确标记缺失或不可比的 run。

#### Scenario: adapter 与 source-only 对比
- **WHEN** 同一 target scene、budget 和 seed 下存在 `v3_decoupled` source-only metrics 以及 `v4_adapter` 或 `v5_adapter_proto` adapted metrics
- **THEN** 结论 MUST 比较 Top-1、Top-3、Top-5、coarse accuracy 和 fine accuracy
- **AND** 结论 MUST 标明 adapter variant 是否优于 source-only

#### Scenario: adapter prototype 与 full fine-tuning 对比
- **WHEN** 同一 target scene、budget 和 seed 下存在 `v5_adapter_proto` 和 `v6_full_finetune` metrics
- **THEN** 结论 MUST 比较 accuracy 指标、trainable ratio 和 adaptation time
- **AND** 结论 MUST 标明 adapter+prototype 是否在效果或效率上优于 full fine-tuning

#### Scenario: 缺失 run 时结论不可判定
- **WHEN** 生成结论所需的 source-only、adapter、prototype 或 full fine-tuning run 缺失
- **THEN** 结论 MUST 将对应比较标记为 `inconclusive`
- **AND** 结论 MUST 记录缺失的 variant、target scene、budget、seed 和原因

#### Scenario: 结论文件写入执行输出目录
- **WHEN** quick validation execute 完成或 partial failure 结束
- **THEN** 系统 MUST 在输出目录写出 `quick_validation_conclusion.json` 或等价机器可读文件
- **AND** 结论文件 MUST 引用产生依据的 LOSO summary 路径

#### Scenario: smoke 资源探针不伪装方法结论
- **WHEN** 用户运行轻量 `quick_smoke` 配置
- **THEN** 结论文件 MAY 标记关键 adapter/full-finetune 对比为 `inconclusive`
- **AND** 系统 MUST 通过 missing/inconclusive 原因说明该配置只覆盖了资源探针矩阵

### Requirement: Geometry-aware transferable knowledge
HiST-Beam MUST 显式建模可迁移知识，包括 coarse angular/beam semantics、angular neighborhood continuity、RSU-CAV relative geometry 和 cross-modal geometric consistency。每项知识 MUST 在配置、模型输出或 loss diagnostics 中有可追踪字段。

#### Scenario: 输出 transferable diagnostics
- **WHEN** geometry-aware HiST-Beam forward 完成
- **THEN** 模型输出 MUST 包含 coarse logits、beam-level logits、shared geometry representation 和 geometry diagnostics
- **AND** diagnostics MUST 至少记录启用的 geometry fields、可用性 mask 和 direct/proxy 标记

#### Scenario: coarse head 绑定 shared geometry
- **WHEN** geometry-aware shared/private 模式启用
- **THEN** coarse head MUST 只读取 shared geometry representation 或其投影
- **AND** private scene representation MUST 不直接作为 coarse head 输入

### Requirement: Scene-private knowledge as explicit refinement
HiST-Beam MUST 将 town/scene layout、RSU pose/local coordinate frame、local scatterer/occluder proxy 和 coarse sector 内 fine beam mapping 作为 scene-private refinement 处理。scene-private 分支 MUST 服务 fine mapping adapter，而不是替代 coarse shared semantics。

#### Scenario: fine head 读取 private refinement
- **WHEN** 模型启用 scene-private branch
- **THEN** fine head MUST 读取 shared geometry representation 与 adapted private representation 的组合
- **AND** adapted private representation MUST 可由 coarse sector embedding 或 coarse context 条件化

#### Scenario: proxy 不伪装成真实标签
- **WHEN** private branch 使用 occluder 或 scatterer 相关输入
- **THEN** 模型 diagnostics MUST 将这些输入标记为 proxy
- **AND** summary MUST 不将 proxy 字段报告为真实 scene semantics 标签

### Requirement: Angular smoothing loss
HiST-Beam MUST 支持 angular smoothing loss，用 beam/codebook 邻接关系构造 soft target，以约束相邻角度 beam 的连续性。该 loss MUST 支持按配置选择 linear ULA 邻接或 circular 邻接。

#### Scenario: 线性 codebook smoothing
- **WHEN** 配置 `angular_smoothing.enabled: true` 且 codebook topology 为 `linear`
- **THEN** 系统 MUST 按 beam index 或 steering angle 的非循环距离构造 soft target
- **AND** beam 0 与最后一个 beam 不得被视为相邻，除非配置显式启用 circular topology

#### Scenario: angular loss diagnostics
- **WHEN** angular smoothing loss 参与训练
- **THEN** loss diagnostics MUST 包含 angular loss 数值、sigma 或温度参数、topology 和有效样本数

### Requirement: Multimodal geometry consistency loss
HiST-Beam MUST 支持 multimodal geometry consistency loss，用于约束 GPS/IMU、CAV/RSU pose、LiDAR、depth、bbox、radar point cloud 和 channel-derived geometry 之间的一致性。该 loss MUST 对缺失模态使用 mask，并 MUST 记录 coverage。

#### Scenario: 可用模态计算 geometry consistency
- **WHEN** batch 同时包含 relative pose 与至少一个可几何对齐的视觉、LiDAR、depth、bbox、radar 或 channel 字段
- **THEN** 系统 MUST 计算配置启用的 geometry consistency 子 loss
- **AND** diagnostics MUST 记录每个子 loss 的 coverage

#### Scenario: 缺失模态跳过子 loss
- **WHEN** 某个 geometry consistency 子 loss 所需模态缺失
- **THEN** 系统 MUST 跳过该子 loss
- **AND** diagnostics MUST 将对应 coverage 记录为 0 或 unavailable reason

### Requirement: Private prototype alignment must be effective
Adapter+prototype 变体 MUST 对齐 coarse sector 条件下的 private/adapter representation，而不是只对 shared representation 做无差别对齐。prototype loss MUST 具有可诊断的 confidence、coverage、used sample count 和非零权重路径；否则该 run MUST 被标记为 prototype no-op。

#### Scenario: private prototype loss 使用 adapter representation
- **WHEN** `v5_adapter_proto` 或等价 prototype 变体执行 target adaptation
- **THEN** prototype consistency MUST 使用 adapted private representation 或配置指定的 private projection
- **AND** prototype target MUST 按 coarse sector 和 confidence threshold 选择
- **AND** shared-only prototype alignment MUST NOT 作为默认实现

#### Scenario: prototype no-op 可诊断
- **WHEN** prototype loss 权重为 0、prototype artifact 缺失、coverage 为 0 或没有样本超过 confidence threshold
- **THEN** adaptation metrics MUST 标记 prototype status 为 `no_op` 或 `unavailable`
- **AND** quick validation conclusion MUST 不把该 run 描述为有效 prototype variant

#### Scenario: v4 与 v5 对比记录 prototype 差异
- **WHEN** 同一 fold、budget、seed 下存在 adapter-only 和 adapter+prototype run
- **THEN** summary MUST 比较两者 accuracy 与 prototype diagnostics
- **AND** 若两者 prediction 完全一致，summary MUST 记录 `prototype_prediction_delta: 0` 或等价诊断

### Requirement: Geometry-aware HiST-Beam 指标
Geometry-aware HiST-Beam evaluation MUST 输出传统 beam 指标以及角度、几何和 prototype 指标。若某项指标缺少必要数据，系统 MUST 标记 unavailable reason，不得伪造数值。

#### Scenario: 输出角度和几何指标
- **WHEN** geometry-aware HiST-Beam evaluation 完成
- **THEN** metrics MUST 包含 Top-1、Top-3、Top-5、coarse accuracy、fine accuracy 和 mean angular error
- **AND** 若启用 geometry loss，metrics MUST 包含 geometry loss coverage 或 unavailable reason

#### Scenario: 输出 prototype 指标
- **WHEN** prototype alignment 启用
- **THEN** metrics MUST 包含 prototype coverage、confidence mean、used sample count 和 prototype loss mean
- **AND** 这些字段 MUST 被 LOSO summary 汇总

### Requirement: HiST-Beam radio-semantic prototype variant
HiST-Beam MUST 在现有 flat、hierarchical、shared-private、adapter-only、adapter+coarse-prototype 和 full fine-tuning baseline 之外，支持 radio-semantic prototype variant。该 variant MUST 显式使用 radio-semantic label、shared radio prototype 和可选 radio-conditioned beam inference，并 MUST 与现有 `v5_adapter_proto` coarse/private prototype baseline 可区分。

#### Scenario: 构建 radio prototype variant
- **WHEN** 用户配置 HiST-Beam variant 为 `v6_radio_proto`、`adapter_radio_proto` 或等价 radio-semantic prototype 模式
- **THEN** 系统 MUST 构建 shared/private/adapted private 表征、radio head、beam head 和 radio prototype diagnostics
- **AND** variant metadata MUST 记录 `proto_type=radio_semantic`

#### Scenario: existing full fine-tuning baseline 不被重解释
- **WHEN** 用户配置现有 `v6_full_finetune` 或 `full_finetune`
- **THEN** 系统 MUST 继续按 full fine-tuning baseline 更新参数
- **AND** summary MUST 不把该 run 标记为 radio-semantic prototype method

### Requirement: HiST-Beam radio branch diagnostics
启用 radio-semantic HiST-Beam 时，模型输出、loss diagnostics 和 evaluation artifact MUST 包含足以证明 radio branch 生效的字段。普通非 radio 配置 MUST 不要求这些字段。

#### Scenario: radio branch 输出被记录
- **WHEN** radio-semantic 配置启用且 forward 完成
- **THEN** 模型 diagnostics MUST 包含 `radio_logits` 或等价 radio prediction 输出
- **AND** diagnostics MUST 包含 `num_radio_classes`、radio label mode 和 radio condition 是否启用

#### Scenario: radio loss no-op 可诊断
- **WHEN** 配置启用 `lambda_radio` 但 batch 没有合法 radio labels
- **THEN** loss diagnostics MUST 将 radio loss 标记为 unavailable 或 coverage 0
- **AND** 系统 MUST 不用 0 coverage 的 radio loss 证明 radio branch 已生效

### Requirement: HiST-Beam radio-conditioned beam head
HiST-Beam MUST 支持 radio-conditioned beam head 作为 opt-in 行为。启用时，beam head 输入 MUST 包含 shared representation、adapted private representation 和 radio assignment embedding；关闭时，系统 MUST 保持现有 shared/private beam head 行为。

#### Scenario: source 阶段使用 predicted radio assignment
- **WHEN** source training 启用 `use_radio_condition_in_beam_head`
- **THEN** 系统 MUST 从 `radio_logits` 的 soft assignment 计算 radio embedding
- **AND** beam logits MUST 来自包含该 embedding 的 beam head 输入

#### Scenario: target 阶段优先使用 source radio prototype assignment
- **WHEN** target adaptation 或 target_test evaluation 启用 radio condition 且 source radio prototypes 可用
- **THEN** 系统 MUST 使用 shared representation 到 `mu_radio_c` 的 assignment 计算 radio embedding
- **AND** 若 prototype artifact 不可用，系统 MUST 记录 fallback 或 unavailable reason

### Requirement: HiST-Beam source prototype 按需生成
HiST-Beam LOSO executor MUST 根据 variant 和配置决定是否生成 source prototype。只有后续 stage 需要 prototype 的 variant 或用户显式要求保存 prototype 时，source training 才应生成 prototype artifact。

#### Scenario: source-only baseline 跳过 prototype
- **WHEN** LOSO run 的 source variant 为 `v0_flat`、`v1_hierarchical`、`v2_shared_private` 或其它不需要 target prototype alignment 的 source-only baseline
- **THEN** source training 默认 MUST 跳过 source prototype 生成
- **AND** run metadata MUST 记录 prototype status 为 `skipped` 及跳过原因

#### Scenario: prototype variant 按需生成或复用
- **WHEN** 后续 `v5_adapter_proto`、`v6_radio_proto` 或 `adapter_radio_proto` stage 需要 source prototype
- **THEN** executor MUST 生成或复用与 fold、source scenes、variant、seed 和 prototype type 匹配的 source prototype artifact
- **AND** 若 artifact 不可用，target adaptation MUST 给出清晰失败或 no-op 诊断

### Requirement: Source prototype 进度与耗时诊断
Source prototype 生成 MUST 提供 stage progress 和耗时诊断，避免 image-heavy source split 二次扫描时表现为无进度卡死。

#### Scenario: prototype pass 写出 progress
- **WHEN** executor 正在生成 source prototype
- **THEN** stage progress MUST 周期性记录 processed batches、total batches 或可用近似进度
- **AND** progress MUST 标明当前 phase 为 `source_prototype`

#### Scenario: prototype metrics 记录额外扫数成本
- **WHEN** source prototype 生成完成
- **THEN** metrics MUST 记录 prototype generation duration、processed sample count、processed batch count 和 prototype coverage
- **AND** LOSO summary MUST 能区分 source training time 和 prototype generation time

### Requirement: MMW HiST-Beam LOSO stage 内存边界
HiST-Beam MMW LOSO 执行器 MUST 在每个 stage 结束后关闭不再需要的 DataLoader worker，并释放 stage-local dataset/loader 引用，使后续 stage 或 run 不继承 image-heavy worker 内存。

#### Scenario: source stage 结束释放 loader
- **WHEN** `source_train` stage 完成、失败或被中断
- **THEN** executor MUST 关闭 source DataLoader worker
- **AND** stage metadata MUST 不保留不可序列化的大 dataset 或 loader 对象

#### Scenario: run summary 记录吞吐配置
- **WHEN** MMW HiST-Beam run 完成或 partial failure
- **THEN** run metadata 或 summary MUST 记录 batch size、num_workers、persistent_workers、prefetch_factor、enabled modalities、seq_len、image cache policy 和 prototype strategy
- **AND** 这些字段 MUST 足以解释 GPU 低利用率和 CPU 内存压力

### Requirement: MMW sensor-assisted HiST-Beam profile
HiST-Beam LOSO workflow MUST support an MMW sensor-assisted profile that uses `image`、`gps`、`lidar` 和 `radar` as model inputs. This profile MUST remain separate from existing MMW `image+gps+mmwave` experiments and MUST expose modality profile metadata in plan、run 和 summary artifacts.

#### Scenario: 构建 sensor-assisted 模型配置
- **WHEN** 用户加载 MMW sensor-assisted HiST-Beam 配置
- **THEN** model modalities MUST resolve to `image`、`gps`、`lidar` 和 `radar`
- **AND** student model field defaults MUST include compatible image、gps、lidar 和 radar encoder settings
- **AND** model construction MUST fail with an actionable error if any enabled modality has no compatible sample key

#### Scenario: 变体矩阵沿用 HiST-Beam baseline
- **WHEN** sensor-assisted LOSO plan 生成
- **THEN** plan MUST support source-only、adapter-only、coarse prototype、radio prototype、path prototype、path condition off 和 full fine-tuning baseline variants where available
- **AND** run metadata MUST distinguish sensor-assisted modality profile from `image+gps+mmwave` profile

#### Scenario: summary 输出负迁移诊断
- **WHEN** sensor-assisted quick validation 写出 `loso_summary`
- **THEN** summary MUST include adapted-source Top-K deltas for adaptation variants
- **AND** summary MUST include negative-transfer flags when adapted Top-1 is lower than corresponding source-only Top-1
- **AND** summary MUST preserve trainable ratio and adaptation time fields for parameter-efficiency comparison

#### Scenario: last-beam baseline 不改变输入语义
- **WHEN** evaluation computes last-beam diagnostic baseline
- **THEN** HiST-Beam summary MAY report last-beam Top-K
- **AND** model input construction MUST NOT add previous beam labels or beam power to sensor-assisted sensing modalities because of that diagnostic

### Requirement: Quick validation conclusion 排除不可用于主结论的 run
HiST-Beam quick validation conclusion MUST 消费 run-level eligibility metadata。`main_conclusion_eligible=false`、target leakage、未授权 target sensitive supervision、prototype no-op 或关键对比 run 缺失的结果 MUST 不被描述为主结论改进。

#### Scenario: ineligible run 不参与胜负判断
- **WHEN** 同一 fold、budget 和 seed 下某个 adapter 或 prototype run 记录 `main_conclusion_eligible=false`
- **THEN** quick validation conclusion MUST 不把该 run 用于证明方法优于 source-only 或 full fine-tuning
- **AND** conclusion MUST 记录该 run 被排除的 variant、target scene、budget、seed 和 eligibility reasons

#### Scenario: excluded baseline 导致比较不可判定
- **WHEN** 生成 adapter/prototype 与 source-only 或 full fine-tuning 对比所需的 baseline run 缺失或被标记为不可用于主结论
- **THEN** 对应比较 MUST 标记为 `inconclusive`
- **AND** conclusion MUST 记录缺失或被排除的 run key 和原因

#### Scenario: prototype no-op 不作为有效 prototype 证据
- **WHEN** prototype run 的 metrics 标记 prototype status 为 `no_op`、`unavailable`、coverage 为 0 或 prototype loss 未实际生效
- **THEN** conclusion MUST 不把该 run 描述为有效 prototype variant
- **AND** 若 accuracy 仍有变化，conclusion MUST 将变化归为补充诊断而不是 prototype 主结论

#### Scenario: conclusion 汇总 eligibility
- **WHEN** quick validation conclusion 文件写出
- **THEN** 文件 MUST 包含 eligible run 数、excluded run 数、inconclusive comparison 数和 exclusion reason histogram
- **AND** 文件 MUST 引用产生 eligibility metadata 的 summary 或 run artifact 路径

### Requirement: History-anchored HiST-Beam 变体
HiST-Beam MUST 支持显式配置的 history-anchored 变体。该变体 MUST 在保留现有 sensing modality fusion、shared/private representation 和 adapter/prototype 框架的基础上，接收历史 beam anchor，并输出 residual/delta logits 与可重建的绝对 beam logits。

#### Scenario: 构建 history-anchored residual 变体
- **WHEN** 用户配置 `hist_beam.history_anchor.enabled=true` 且 `hist_beam.history_anchor.mode=residual_delta`
- **THEN** 系统 MUST 构建包含 beam-history embedding 或等价 conditioning 的 HiST-Beam 模型
- **AND** 模型 MUST 输出 residual logits、reconstructed beam logits、shared representation 和 private representation
- **AND** 输出 MUST 继续兼容现有 Top-K beam evaluation 流程

#### Scenario: 关闭 history anchor 保持旧变体语义
- **WHEN** 用户运行 `v0_flat`、`v3_decoupled`、`v6_radio_proto`、`v8_path_proto` 或 full fine-tuning baseline 且未显式启用 history anchor
- **THEN** 模型 forward MUST 不要求 `input_beam_batch`
- **AND** source-only evaluation MUST 与当前绝对 beam prediction 语义保持兼容

#### Scenario: history absolute classifier 作为消融
- **WHEN** 用户配置 `hist_beam.history_anchor.mode=absolute_with_history`
- **THEN** 模型 MAY 使用历史 beam embedding 预测绝对 beam logits
- **AND** summary MUST 将其标记为 history-input absolute classifier ablation，而不是 residual 主方法

### Requirement: Residual beam loss
启用 history-anchored residual 模式时，训练流程 MUST 使用 residual/delta label 计算主 beam loss，并 MAY 保留可配置的绝对 beam auxiliary loss。残差 loss MUST 支持多 horizon 输出，并 MUST 与现有 hierarchical、radio、path 和 orthogonality loss 组合。

#### Scenario: source training 计算 residual CE
- **WHEN** source training batch 包含合法 `input_beam` 和 future beam label
- **THEN** training loop MUST 计算 circular residual label
- **AND** training loop MUST 对 residual logits 计算 CE loss
- **AND** metrics MUST 记录 residual loss 和 reconstructed absolute Top-K

#### Scenario: 绝对 auxiliary loss 可配置
- **WHEN** 配置设置 `hist_beam.history_anchor.lambda_absolute_aux > 0`
- **THEN** training loop MAY 对 reconstructed absolute beam logits 计算 auxiliary CE
- **AND** 该 auxiliary loss MUST 使用真实 future beam label，而不是 residual label

#### Scenario: history anchor 缺失时失败而非静默降级
- **WHEN** history-anchored residual 模式的训练 batch 缺少合法历史 beam anchor
- **THEN** training MUST 失败并输出包含 `input_beam` 或 `last_beam` 的错误信息
- **AND** 系统 MUST NOT 静默改用绝对 beam CE 继续训练

### Requirement: Residual shared-private 解耦
history-anchored residual 模式下，HiST-Beam shared/private 解耦 MUST 将 shared branch 定义为相对传播 residual 表征，将 private branch 定义为场景私有校准表征。模型 MUST 在 metadata 中区分 residual shared prediction 与 private calibration。

#### Scenario: shared branch 预测 residual distribution
- **WHEN** 模型启用 shared/private 和 history-anchored residual 模式
- **THEN** shared branch MUST 产生用于 residual/delta prediction 的 representation 或 logits
- **AND** shared branch MAY 继续输出 path/radio/geometry auxiliary head
- **AND** shared branch MUST NOT 被解释为直接学习 source 场景绝对 beam prior 的主分支

#### Scenario: private branch 产生校准项
- **WHEN** 模型启用 private calibration
- **THEN** private branch 或 adapter MUST 能输出 logit bias、temperature、offset、prototype-conditioned correction 或等价场景私有校准项
- **AND** calibration metadata MUST 记录实际启用的校准类型和 trainable parameter count

#### Scenario: prototype 不直接替代 residual prediction
- **WHEN** radio 或 path prototype 与 history-anchored residual 模式同时启用
- **THEN** prototype MAY 作为 shared assignment、private calibration 或 auxiliary diagnostic
- **AND** prototype MUST NOT 绕过 residual head 直接输出最终 beam prediction，除非该 run 明确标记为非 residual 消融

### Requirement: History-anchored few-shot private calibration
target adaptation 在 history-anchored residual 模式下 MUST 支持低参数 private calibration。默认策略 MUST 冻结 source encoders、fusion backbone 和 shared residual branch，只训练配置允许的 private adapter、calibration head、logit bias、temperature、LayerNorm affine 或等价低参数模块。

#### Scenario: few-shot adaptation 冻结 shared residual backbone
- **WHEN** 用户运行 history-anchored residual target adaptation 且未显式选择 full fine-tuning
- **THEN** 系统 MUST 冻结 sensing encoders、fusion module 和 shared residual branch
- **AND** 系统 MUST 只训练配置允许的 private calibration 参数
- **AND** metrics MUST 记录 trainable parameter count、total parameter count 和 trainable ratio

#### Scenario: labeled target 使用 residual supervised loss
- **WHEN** `label_budget>0` 且 labeled target_adapt 样本存在合法 future beam label
- **THEN** adaptation MUST 基于 labeled target_adapt 样本计算 residual supervised loss
- **AND** unlabeled target_adapt 样本 MUST NOT 读取 future beam label 作为 supervised loss

#### Scenario: target sensitive supervision 保持可审计
- **WHEN** history-anchored residual adaptation 使用 target path、radio、beam_power 或 channel-derived supervision
- **THEN** run metadata MUST 记录对应 sensitive usage flag
- **AND** summary MUST 根据 profile 规则标记该 run 是否可用于主结论

### Requirement: History-anchored HiST-Beam 预测产物
history-anchored residual evaluation MUST 在现有 HiST-Beam predictions 和 metrics 基础上新增 residual 诊断字段。产物 MUST 同时保留 residual-space 信息和 reconstructed absolute beam 信息。

#### Scenario: predictions 保存 residual 字段
- **WHEN** source-only evaluation 或 adapted evaluation 完成 history-anchored residual run
- **THEN** predictions MUST 包含 sample id、scene、last_beam、true beam、true residual、predicted residual、top-k residual、predicted beam 和 top-k reconstructed beam
- **AND** predictions MUST 标明样本来自 `target_test`

#### Scenario: metrics 输出 residual 与绝对指标
- **WHEN** evaluation 完成 history-anchored residual run
- **THEN** metrics MUST 包含 residual accuracy 或 residual error diagnostic
- **AND** metrics MUST 包含 reconstructed absolute Top-1、Top-3、Top-5
- **AND** 若 beam_power 可用，metrics MUST 包含 reconstructed absolute prediction 的 normalized received power 和 beam power loss dB

#### Scenario: summary 可比较 residual 和 absolute baseline
- **WHEN** summary 汇总同一 source、target、budget 和 seed 下的 absolute baseline 与 residual run
- **THEN** summary MUST 输出 residual run 相对 absolute source-only 的 delta
- **AND** summary MUST 输出 residual run 相对 last-beam 和 Markov delta baseline 的 delta 或不可比原因

### Requirement: HiST-Beam 主线默认 no-KD
HiST-Beam 跨场景适配主线 MUST 默认使用 no-KD supervised/adaptation 训练。source-only、shared/private、adapter-only、adapter+prototype、path/radio prototype、history-anchored residual、private calibration 和 full fine-tuning baseline MUST 不要求 teacher-student distillation、teacher checkpoint 或 KD loss。

#### Scenario: HiST-Beam source training 不加载 teacher
- **WHEN** 用户运行当前推荐的 HiST-Beam source training 或 sensor-assisted LOSO source stage
- **THEN** 系统 MUST 只构建当前配置指定的主模型
- **AND** 系统 MUST 不构建 frozen teacher model
- **AND** run metadata MUST 记录 `distillation_enabled=false`

#### Scenario: target adaptation 不计算 KD loss
- **WHEN** 用户运行 adapter、prototype、path/radio prototype、residual calibration 或 full fine-tuning target adaptation
- **THEN** adaptation loss MUST 来自 supervised target loss、无标签一致性/prototype/entropy/calibration loss 或对应方法定义
- **AND** adaptation MUST 不要求 teacher/student logits 对齐

### Requirement: KD 仅作为显式 HiST-Beam baseline 或增强
HiST-Beam 工作流 MAY 保留 KD baseline 或后续 KD 增强，但必须显式 opt-in，并且 MUST 与默认 HiST-Beam 主线矩阵分离。KD baseline 不得静默加入 sensor-assisted quick validation、history-anchored residual quick validation 或主结论 conclusion。

#### Scenario: 显式运行 HiST-Beam KD baseline
- **WHEN** 用户选择明确标注为 HiST-Beam KD baseline 的配置
- **THEN** plan metadata MUST 记录 `method_family=legacy_kd` 或等价字段
- **AND** run metadata MUST 记录 teacher source、teacher checkpoint、distillation type 和 student variant
- **AND** summary MUST 将该 run 归入 KD baseline 分组

#### Scenario: 默认矩阵排除 KD baseline
- **WHEN** 用户生成默认 HiST-Beam quick validation、MMW sensor-assisted quick validation 或 history-anchored residual quick validation plan
- **THEN** plan MUST 不包含 KD baseline variant
- **AND** 只有用户显式选择 legacy KD baseline profile 时才生成 KD run

### Requirement: HiST-Beam shared/private 语义不依赖 KD
HiST-Beam shared/private 解耦、prototype alignment、history residual 和 scene-private calibration MUST 以跨场景可迁移/场景私有表征为核心定义，不得把 teacher-student distillation 作为这些分支生效的必要条件。

#### Scenario: shared/private diagnostics 无 teacher 仍完整
- **WHEN** HiST-Beam shared/private 或 residual calibration 模型在 no-KD 配置下 forward
- **THEN** 模型 MUST 输出该变体要求的 shared/private/residual/prototype/calibration diagnostics
- **AND** diagnostics MUST 不依赖 teacher features 或 teacher logits

#### Scenario: prototype alignment 不使用 teacher soft target
- **WHEN** adapter+prototype 或 path/radio prototype adaptation 计算 prototype loss
- **THEN** prototype target MUST 来自 source prototype、target representation、path/radio semantic assignment 或配置定义的物理/几何 proxy
- **AND** 系统 MUST 不把 teacher prediction distribution 作为默认 prototype target

### Requirement: V7 shared physical private residual forward contract
V7 模型 MUST 让 shared 分支独立预测 beam，并让 private 分支只产生 gated residual correction。private residual MUST NOT 作为完整 beam prediction 单独训练或评估为主输出。

#### Scenario: final logits 由 shared 加 residual 得到
- **WHEN** V7 forward 接收有效 multimodal batch
- **THEN** `logits_final` MUST 等于 `logits_shared + alpha * delta_logits_private`
- **AND** `alpha` 第一版 MUST 支持 shape `[B, H, 1]` 或可 broadcast 到 beam class 维

#### Scenario: shared 分支独立可评估
- **WHEN** evaluation 读取 V7 输出
- **THEN** 系统 MUST 能仅使用 `logits_shared` 计算 Top-K、beam power loss 和 NRP
- **AND** shared-only 指标 MUST 与 final 指标分开记录

#### Scenario: private residual 不作为完整预测
- **WHEN** V7 训练或评估运行
- **THEN** 系统 MUST NOT 把 `delta_logits_private` 直接作为 beam classifier 主 logits
- **AND** 训练 MUST 对 residual magnitude 或 gate 增加约束，防止 private 分支偷走完整预测任务

### Requirement: V7 source training losses
系统 MUST 在 V7 source training 中计算 shared hard CE、final hard CE、beamspace soft KL、physical head KL、residual L2、gate L1 和 shared/private difference loss，并按配置权重合成 total loss。

#### Scenario: 使用 beamspace_power_label 计算 shared physical loss
- **WHEN** V7 source batch 包含有效 `beamspace_power_label`
- **THEN** 系统 MUST 使用 `log_softmax(logits_shared / T)` 与 BSP target 计算 KL loss
- **AND** 系统 MUST 使用 `pred_beamspace_power` 与 BSP target 计算 physical head KL loss

#### Scenario: warmup 阶段禁用 private residual
- **WHEN** 当前 epoch 小于 `training.shared_warmup_epochs`
- **THEN** V7 training MUST 令 final prediction 等价于 shared prediction
- **AND** total loss MUST 不包含 final residual、residual L2、gate L1 或 private residual 相关项

#### Scenario: BSP 缺失时不静默训练物理 loss
- **WHEN** V7 source batch 缺少有效 `beamspace_power_label`
- **THEN** 系统 MUST 按配置拒绝训练或将 physical loss 标记为 unavailable
- **AND** diagnostics MUST 记录不可用原因

### Requirement: V7 target private residual adaptation
系统 MUST 支持从 V7 source checkpoint 启动 target adaptation，并在默认策略中冻结 shared backbone、shared heads 和 physical head，只训练 target private adapter、private residual head、residual gate 和配置允许的 norm affine 参数。

#### Scenario: V7 adaptation 冻结 shared 参数
- **WHEN** 用户应用 adaptation strategy `v7_private_residual`
- **THEN** modality encoders、fusion transformer、shared branch、shared beam head 和 physical head 参数 MUST `requires_grad=false`
- **AND** trainable parameter summary MUST 反映实际白名单参数比例

#### Scenario: V7 adaptation loss 不使用 target physical oracle
- **WHEN** target labeled adaptation batch 包含 hard beam label 和 target-side `beamspace_power_label`
- **THEN** 默认 adaptation loss MUST 只使用 final hard CE、residual L2 和 gate L1
- **AND** 系统 MUST NOT 使用 target-side BSP 对 shared 分支进行训练反传

#### Scenario: V7 不使用历史 label
- **WHEN** V7 source training 或 target adaptation 运行
- **THEN** 模型输入 kwargs MUST NOT 要求 `input_beam_batch` 或 `last_beam_batch`
- **AND** leakage diagnostics MUST 标记 `uses_input_beam_as_model_input=false`

### Requirement: V7 evaluation metrics and artifacts
系统 MUST 在 V7 evaluation、adapted target_test evaluation 和 LOSO summary 中输出 shared-only 与 final prediction 的对比指标，以及 gate、residual 和 physical alignment 诊断。

#### Scenario: metrics 包含 shared 和 final 指标
- **WHEN** V7 evaluation 完成
- **THEN** metrics MUST 包含 `shared_top1`、`shared_top3`、`final_top1`、`final_top3`
- **AND** 若 beam power vector 可用，metrics MUST 包含 `shared_beam_loss_db`、`final_beam_loss_db`、`shared_nrp` 和 `final_nrp`

#### Scenario: metrics 包含 residual 诊断
- **WHEN** V7 evaluation 完成
- **THEN** metrics MUST 包含 `alpha_mean`、`alpha_std` 和 `delta_norm`
- **AND** 若 BSP target 可用，metrics MUST 包含 `phys_kl`

#### Scenario: predictions 标明 final 和 shared 输出
- **WHEN** 系统写出 V7 target_test predictions
- **THEN** predictions MUST 至少包含 sample id、scene、true beam、final predicted beam、shared predicted beam、final top-k、shared top-k 和 variant metadata
- **AND** predictions MUST 标明样本来自 `target_test`

#### Scenario: LOSO summary 汇总 V7 字段
- **WHEN** V7 source-only 或 adapted run 写入 LOSO summary
- **THEN** summary MUST 包含 variant、target_scene、budget、seed、shared/final accuracy、alpha/residual 诊断和 physical KL
- **AND** summary MUST 能与 v3/v4/v6/v8/full-finetune baseline 横向比较

