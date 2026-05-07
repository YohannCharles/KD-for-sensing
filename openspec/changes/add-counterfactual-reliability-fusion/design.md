## Context

当前仓库已经形成配置驱动实验体系：模型、loss、metric、distiller 通过 `kd_sensing.registries` 构建；训练和评估入口集中在 `src/kd_sensing/engine/trainer.py` 与 `validator.py`；fusion 数据输入通过 `engine.batch.prepare_fusion_inputs` 根据 `modalities` 准备；canonical fusion 配置可以按固定模态顺序生成多模态组合。

CRAF 第一阶段已经完成模型构建、输出适配、反事实训练、beam soft loss、单模态辅助 head、示例配置和 smoke tests。最新 all-modal CRAF 实验暴露出新的问题：模型训练准确率高，但验证准确率在 warmup 后下滑；reliability 分数集中在 0.46-0.58，且 LiDAR/image 等弱模态没有被明显压低，GPS/mmWave 也没有稳定排在前面。这说明当前 gate 监督没有学到干净的模态可靠性，更可能在主模型未稳定时从噪声反事实差值中学习。

因此本设计不再扩大 CRAF 的功能面，而是收敛训练策略：延长 warmup、净化 counterfactual target、加入 ignore band、改用 competitive gate、降低 auxiliary/beam soft 干扰，并补齐能定位 gate 是否学对的日志。

## Goals / Non-Goals

**Goals:**

- 让 warmup 阶段的 CRAF 先学习稳定的 all-modal token fusion，再启用 reliability gate 监督。
- 让 counterfactual contribution 只反映 beam classification 主任务 CE 的变化。
- 对不明确的 `delta` 不产生 0.5 附近的模糊监督，避免 reliability 全部塌在中间值。
- 支持 softmax-normalized modality gate，使可用模态之间形成竞争，并通过温度退火逐步增强选择性。
- 将 unimodal auxiliary loss 改为 warmup-only 或两段式 schedule，减少弱单模态分支在后期污染融合模型。
- 降低 beam soft loss 默认权重，先保证 Top-1 不低于强单模态 teacher，再用 soft loss 优化 DBA。
- 增加 counterfactual delta、target、valid rate 和有效 loss 权重日志，使训练失败能被快速定位。
- 提供最小稳定化实验矩阵，区分 backbone、gate、counterfactual 和固定 prior 的贡献。

**Non-Goals:**

- 不重写 DeepSense6G dataset、collate、预处理和 CSV 生成流程。
- 不引入新训练脚本或绕开 `kd_sensing.cli.train`、`kd_sensing.cli.evaluate`。
- 不要求 CRAF 与 KD 在本阶段联合调优；稳定化版本仍优先 no-KD。
- 不把所有 canonical fusion 默认切换到 CRAF。
- 不在本阶段强制实现层级 temporal-then-cross-modal Transformer；该结构作为后续风险缓解方案保留。

## Decisions

1. warmup 阶段固定 gate 为全 1，而不是继续使用 reliability estimator 输出。

   现有曲线显示 CRAF 早期验证性能较好，warmup 后出现下滑。反事实贡献依赖 full/drop forward 的 loss 差异，主模型未稳定时该差异噪声很大。实现上在 `epoch < warmup_epochs` 或 `epoch < counterfactual.start_epoch` 时，对可用模态使用等价 `r_m = 1` 的 gate，并跳过 gate loss。替代方案是只降低 gate loss 权重，但模型仍会被不稳定 gate 改变 token 幅值。

2. counterfactual contribution 只使用主任务 CE。

   贡献定义用于判断某模态是否提升 beam classification，而 beam soft、unimodal auxiliary、KD 和 gate loss 都不是这个判断本身。helper 应提供 CE-only per-sample loss，并明确不把附加 loss 混入 `delta`。替代方案是复用训练总 loss，但这会让 `delta` 同时反映 auxiliary head、soft label 和 gate 自身误差，目标不干净。

3. 对小幅 `delta` 使用 ignore band，并生成二值 target。

   当 `|delta| <= ignore_delta_eps` 时，删除或加入模态的影响不明确，不应强制映射为 0.5 target。helper 返回 `target` 和 `target_valid_mask`：`delta > eps` 监督为 1，`delta < -eps` 监督为 0，其余忽略。替代方案是继续使用 sigmoid(delta / tau)，但大量接近 0 的 delta 会让 gate 学到集中在 0.5 附近的无区分分数。

4. 增加 `context_marginal` 反事实模式，保留 `sample_one` 和 `leave_one_out`。

   leave-one-out 比较 `S` 与 `S \ {m}`，但 full set 已被弱模态污染时，边际贡献可能不稳定。`context_marginal` 随机采样不含目标模态的上下文 `A`，比较 `CE(A)` 与 `CE(A ∪ {m})`，更贴近“模态在不同组合下的条件贡献”。替代方案是完全替换 leave-one-out，但保留旧模式有利于回归和消融对比。

5. 默认新增 softmax-normalized modality gate。

   independent sigmoid gate 容易让所有可用模态停在相近的 0.5 附近，弱模态仍大量进入 Transformer。softmax gate 在可用模态集合上计算 `alpha_m = softmax(score_m / T_g)`，再用 `K_available * alpha_m` 保持总体幅值；不可用模态仍由 mask 排除，`min_gate` 只作用于可用模态。替代方案是对 sigmoid 增加强正则，但它不能直接制造模态间竞争。

6. gate 温度和 gate loss 权重都采用 schedule。

   gate 温度从较平滑的 `gate_temperature_start` 退火到更有选择性的 `gate_temperature_end`；gate loss 在 `counterfactual.start_epoch` 后用 `gate_ramp_epochs` 线性增大到目标权重。这样降低 warmup 边界处的突然扰动。替代方案是固定低温和固定 gate 权重，但容易在 gate 刚启用时破坏已经学到的融合表示。

7. 单模态 auxiliary loss 改成 warmup-only 或两段式权重。

   image、LiDAR、radar 在当前场景的单模态验证表现明显较弱，后期持续优化这些 auxiliary head 会鼓励弱分支拟合训练集，并污染 confidence/reliability 特征。配置上支持 `uni_weight_warmup` 与 `uni_weight_after_warmup`，推荐后期为 0 或很小值。替代方案是完全删除 auxiliary head，但 reliability estimator 仍需要 confidence 特征和可选诊断。

8. beam soft loss 默认降权。

   Beam-aware soft label 对 DBA 有价值，但当前优先目标是恢复 Top-1，并使 CRAF 不低于 GPS/mmWave teacher。默认把 `beam_soft_weight` 降到保守值，等 Top-1 稳定后再提高。替代方案是保持原权重，但可能让 soft target 优化压过主任务 CE。

9. 诊断日志必须覆盖 delta、target 和 valid rate，而不仅是 reliability。

   只看 reliability 无法判断 gate 学错是因为 contribution 计算错、target 全在 0.5 附近，还是 valid target 太少。训练聚合需要按模态记录 `cf/delta_mean_*`、`cf/target_mean_*`、`cf/target_valid_rate_*`，以及 gate 温度和有效 loss 权重。替代方案是只记录总 gate loss，但它不足以定位模态级失败。

10. 固定强模态 prior 作为 sanity check 配置，不作为默认算法。

   如果固定 `mmwave/gps` 高、`image/lidar/radar` 低的 prior 能提升验证准确率，说明“抑制弱模态”方向成立，问题主要在 learned reliability；如果不能提升，则需要检查融合 backbone 或数据处理。该配置只用于实验诊断，不替代 learned gate。

## Risks / Trade-offs

- [Risk] 延长 warmup 会增加一次实验的有效调参周期。Mitigation：保留短 smoke test 配置，并只在正式 all-modal 实验中使用 20-30 epoch warmup。
- [Risk] ignore band 过大导致有效 target 太少。Mitigation：记录 `target_valid_rate`，并把 `ignore_delta_eps` 暴露为配置，默认从 0.03 起步。
- [Risk] softmax gate 可能过早压制弱模态，损失潜在互补信息。Mitigation：使用温度退火、`min_gate` 和 gate loss ramp，必要时回退到 sigmoid gate 做消融。
- [Risk] `context_marginal` 需要额外 forward，训练吞吐下降。Mitigation：默认 `num_drop_per_batch: 1`，支持 no-grad drop forward，并保留 `sample_one` 作为低成本模式。
- [Risk] warmup-only auxiliary 可能削弱 reliability estimator 的 confidence 特征。Mitigation：forward 仍保留 auxiliary logits/confidence，可用很小的 after-warmup 权重做消融。
- [Risk] 诊断字段增加日志体积。Mitigation：只聚合 epoch 标量，不保存 per-sample 细节。

## Migration Plan

1. 先扩展 CRAF 配置解析、gate schedule、loss schedule 和 counterfactual target helper 的单元测试。
2. 接入训练循环，确保 warmup 阶段固定 gate、CE-only delta、ignore band 和 gate ramp 都只在 CRAF 显式配置时生效。
3. 增加 softmax gate 与 `context_marginal` 模式，并保留旧 sigmoid/sample-one/leave-one-out 配置用于回归。
4. 扩展日志聚合和 TensorBoard 标量，补齐每模态 delta/target/valid rate。
5. 新增稳定化 CRAF 配置和消融配置，完成短训练 smoke test。
6. 使用 all-modal 场景跑正式对比：token transformer 无 gate、CRAF 无 counterfactual、CRAF 稳定化 gate、固定强模态 prior。

回滚策略是将 CRAF 配置切回 `reliability.gate_type: sigmoid`、关闭 `context_marginal`、设置附加 loss 权重为 0 或使用 token transformer baseline；legacy fusion 和单模态路径不需要迁移。
