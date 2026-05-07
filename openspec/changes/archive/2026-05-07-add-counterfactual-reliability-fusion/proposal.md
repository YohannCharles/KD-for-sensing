## Why

当前 CRAF 已经接入项目训练体系并具备 all-modal fusion 能力，但最新实验显示它更像一个高容量 all-modal 模型：训练集准确率较高，验证集准确率在 warmup 后下降，且 reliability 排序没有反映 GPS/mmWave 强、image/LiDAR/radar 弱的单模态事实。这说明当前反事实 gate 监督过早、目标噪声较大，弱模态 auxiliary loss 与 beam soft loss 也可能在主任务尚未稳定时干扰 Top-1。

本次改进的目标是把 CRAF 从“可运行”推进到“可靠性监督可诊断、训练策略更稳、可解释地抑制弱模态”的下一版方案。

## What Changes

- 调整 CRAF 训练阶段划分：延长 warmup，warmup 阶段固定 gate 为全 1 或等价无门控，并在 `counterfactual.start_epoch` 后再启用反事实 gate supervision。
- 改进 counterfactual target：贡献差值只使用主任务 CE，不混入 beam soft、unimodal auxiliary、KD 或 gate loss；对小幅 `delta` 增加 ignore band，只监督明确有益或有害的样本-模态对。
- 新增 `context_marginal` 反事实模式：随机采样不含目标模态的上下文子集 `A`，比较 `A` 与 `A ∪ {m}` 的 CE 差异，估计条件边际贡献。
- 支持 competitive softmax modality gate：可用模态之间通过 softmax 竞争，并支持温度退火、`min_gate` 与不可用模态 mask，避免所有 gate 长期挤在 0.5 附近。
- 调整附加 loss 调度：单模态 auxiliary loss 支持 warmup-only 或 warmup/after 两段权重；beam soft loss 默认降权，优先稳定 Top-1 后再优化 DBA。
- 增加 gate loss ramp：反事实监督启用后按 epoch 逐步增加权重，避免 warmup 后立即扰动主模型。
- 扩展诊断日志：记录每模态 `cf/delta_mean_*`、`cf/target_mean_*`、`cf/target_valid_rate_*`、gate 温度、gate loss 有效权重和各附加 loss 的实际权重。
- 新增稳定化实验配置与消融矩阵：token transformer 无 gate、CRAF 无反事实、CRAF 稳定化 gate、固定强模态 prior sanity check。
- 保持 legacy fusion、单模态和 KD 默认训练路径不变；上述策略仅在 CRAF 显式配置时启用。

## Capabilities

### New Capabilities

- `counterfactual-reliability-fusion`: 描述 CRAF 模型、可靠性估计、competitive gate、反事实贡献监督、loss schedule、诊断日志和 baseline 对比的行为契约。

### Modified Capabilities

- `configurable-multimodal-fusion`: CRAF 配置需要支持稳定化训练默认值、softmax gate、context-marginal 反事实和消融实验入口。
- `experiment-workflow`: 训练和评估流程需要支持 warmup gate 固定、CE-only counterfactual target、ignore band、gate/loss ramp、分模态 counterfactual 诊断日志和稳定化 smoke test。
- `component-registry`: 新增或扩展的 gate、loss、counterfactual target 与 mask helper 必须保持可测试的窄模块边界，并通过现有注册/默认导入方式接入。

## Impact

- 影响代码：`src/kd_sensing/models/fusion/` 的 reliability gate、`src/kd_sensing/engine/trainer.py` 的 CRAF 附加 loss 和 counterfactual 训练路径、CRAF loss/mask helper、日志聚合与测试。
- 影响配置：新增或调整 `training.warmup_epochs`、`training.counterfactual.mode: context_marginal`、`training.counterfactual.ignore_delta_eps`、`training.counterfactual.use_ce_only`、`loss.gate_ramp_epochs`、`loss.uni_weight_warmup`、`loss.uni_weight_after_warmup`、`model.student.reliability.gate_type` 与 gate 温度字段。
- 影响输出：训练日志、TensorBoard 和 metrics 增加每模态反事实贡献、target 均值、target 有效率、gate 温度和附加 loss 有效权重。
- 兼容性：非 CRAF 模型、CRAF 附加 loss 权重为 0 的配置、legacy `fusion_teacher`/`fusion_student` 和既有单模态配置必须保持现有行为。
- 依赖：不新增外部运行时依赖；继续使用 PyTorch 和现有项目组件。
