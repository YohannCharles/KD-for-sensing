## Context

当前 MMW Town10 的 P3/HiST-Beam 跨场景结果呈现出明确的结构性问题：同场景单模态和多模态训练能够正常学习，但 `skybridge -> crossroad` 的 source-only target test 接近 64 类随机水平；同时 `last_beam` 诊断基线在目标场景上很强。这说明数据加载、label 解析和评估主路径大概率不是根因，根因更像是当前模型只预测绝对 beam ID，且没有消费 `input_beam` 这个最强的历史锚定信息。

现有代码结构也支持这个判断：`HistBeamFusionNet.forward()` 主要接收 image/radar/gps/lidar/mmwave/geometry/prototype 等输入，`prepare_fusion_inputs()` 只按配置模态构造 sensing inputs；`evaluation_pass.py` 中的 `input_beam` 主要用于诊断 baseline，而不是模型输入。当前 MMW sensor-assisted 主 profile 只把 `image`、`gps` 和 `lidar` 作为主 sensing inputs，radar 仍保留为显式可选的通用模态，不进入默认主结论。因此本 change 必须新增显式 opt-in 的 history-anchored profile，而不是隐式改变现有主结论语义。

## Goals / Non-Goals

**Goals:**

- 新增一个可审计的 history-anchored residual beam 方法族，显式使用样本历史窗口中的 `input_beam`。
- 将跨场景主预测目标从绝对 beam ID 改为 `last_beam` 条件下的 circular residual/delta，并在评估时重建回 64 类绝对 beam 空间。
- 保留现有 Top-K、NRP、beam power loss dB、predictions artifact 和 LOSO summary 兼容性。
- 将 shared/private 解耦语义调整为：shared 学相对传播规律，private 学场景私有校准。
- 提供最小验证矩阵，用已有 3 个 sunny/Town10 场景、2 个 seed、budget 10 快速判断 residual 是否能解决随机水平和少样本负迁移。

**Non-Goals:**

- 不把历史 beam 输入并入默认 `sensor_assisted_quick_validation` 主结论；默认主 profile 继续保持 image/GPS/LiDAR 输入边界，并排除 radar sensing input。
- 不要求本 change 同时完成 multi-source DG 全矩阵；multi-source 作为 residual 闭环成立后的第二阶段增强。
- 不新增外部深度学习依赖，不重写现有 dataset、LOSO runner 或 P3 prototype 框架。
- 不允许使用 target_test 的 future beam、beam_power、path/radio label 来训练、选阈值或更新 prototype。

## Decisions

### Decision 1: history anchor 作为独立 opt-in profile

新增 `hist_beam.history_anchor.enabled` 或等价配置开关，并在 run metadata 中记录 `profile=history_anchored`、`uses_input_beam_as_model_input=true`。启用后，batch preparation 将 `input_beam`/`last_beam` 作为模型条件输入；关闭时现有 sensor-assisted 和 P3 路径不变。

选择该方式是因为现有 sensor-assisted spec 明确把 last-beam 作为 diagnostic baseline，而不是默认模型输入。独立 profile 能让论文/报告同时保留两类结论：严格 `image`/`gps`/`lidar` 跨场景结论，以及历史锚定 beam prediction 结论。

备选方案是直接把 `input_beam` 加入所有 HiST-Beam forward。该方案实现更快，但会污染已有 sensor-assisted 主结论，并使历史 run 与当前 P3/V8 run 不可比。

### Decision 2: circular residual/delta 是主训练目标

新增 residual label 工具：

- `delta = (future_beam - last_beam) mod num_classes`
- 训练 residual head 输出 `[B, H, num_classes]` delta logits
- 评估时对每个样本执行环形平移，将 delta logits 重建为绝对 beam logits，再复用现有 Top-K/power 评估

这样模型学习的是“相对当前 beam 往哪边变化、变化几格”，而不是 source 场景中 beam 34/33/35 这类绝对 ID 先验。对 crossroad 这种绝对主 beam 分布和 skybridge 明显错位的场景，residual target 更符合可迁移传播知识。

备选方案是只做 `input_beam embedding + absolute classifier`。该方案可以作为 ablation，但仍可能学到“last_beam 附近的 source-specific 绝对类偏置”，不能直接解决绝对标签语义漂移。

### Decision 3: shared/private 解耦改为 residual + calibration

history-anchored 模式下，shared branch 的主 head 预测 residual/delta distribution，并可继续承担 path/radio/geometry auxiliary；private branch 或 adapter 输出场景私有校准项，例如 class logit bias、temperature、beam offset、prototype-conditioned correction 或低秩 adapter correction。最终绝对 beam prediction 来自：

1. shared residual logits；
2. 基于 `last_beam` 的绝对空间重建；
3. private calibration 对绝对 logits 或 residual logits 的轻量修正。

这保留了现有 P3/HiST-Beam 的 shared/private 框架，但把“共享知识”从绝对 beam 分类迁移到更稳定的相对传播规律。

备选方案是继续对绝对 beam logits 做 shared/private 分支。这已经在当前实验中表现为 source prior collapse，因此只适合作为失败 baseline。

### Decision 4: few-shot adaptation 只开放低参数私有校准路径

在 residual 模式的 target adaptation 中，默认冻结 sensing encoders、fusion backbone 和 shared residual branch；只训练 private adapter、residual/calibration head、logit bias、temperature、LayerNorm affine 或等价低参数校准模块。`label_budget>0` 时只用 labeled target_adapt beam label 计算 residual/absolute CE；unlabeled target_adapt 可使用 entropy、consistency 或 prototype confidence loss；target_test 不参与任何训练决策。

当前 budget 10 且只训练极小 adapter 的负效果表明，少样本阶段需要校准“绝对映射/场景偏置”，而不是大幅改写 shared representation。

### Decision 5: 先跑最小诊断矩阵，再扩展 multi-source

最小矩阵应覆盖一个 source 泛化到两个 target、2 个 seed、budget 10，并包含：

- `last_beam` 和 Markov delta baseline；
- 当前 absolute source-only 失败 baseline；
- history input + absolute classifier；
- residual-only；
- residual + private calibration；
- 可选 residual + path/radio prototype。

只有当 residual-only 明显脱离随机水平后，再扩展 multi-source LOSO 或更多 budget sweep。这样能避免在错误 label formulation 上继续堆复杂度。

## Risks / Trade-offs

- [历史 beam shortcut 影响论文口径] → 用独立 `history_anchored` profile、metadata 和 summary filter 隔离，默认 sensor-assisted 主结论不变。
- [residual 重建实现错位导致 Top-K 错误] → 为 delta label、环形重建、horizon 维度和 top-k 排序新增单元测试。
- [budget 10 下 private calibration 仍过拟合] → 默认只开放低参数校准，记录 trainable ratio，支持 early stopping/weight decay/temperature clamp，并和 residual-only 做配对比较。
- [last_beam baseline 太强，方法难以超过] → 将目标拆成两个层次：先证明 residual 修复 absolute transfer collapse，再用 Top-K/NRP/DBA 和长预测 horizon 证明相对 baseline 的增益。
- [旧配置和旧 summary 混用] → 所有 history-anchored run 必须写出 `uses_input_beam_as_model_input`、`history_anchor_mode`、`residual_target_enabled` 和 `main_conclusion_profile`。

## Migration Plan

1. 新增纯工具层 residual label/reconstruction 与测试，不触碰现有模型行为。
2. 在 batch preparation 和 model forward 中加入 opt-in `input_beam_batch`，默认关闭时保持旧 forward kwargs 和指标一致。
3. 增加 residual head/loss/eval 重建路径，并用 smoke test 验证单 batch 可训练和可评估。
4. 增加 target private calibration 参数选择、trainable ratio metadata 和 leakage flags。
5. 新增 history-anchored quick validation YAML/script，先跑最小矩阵；确认有效后再考虑 multi-source 和更大 budget sweep。

回滚策略：关闭 `hist_beam.history_anchor.enabled` 即可回到当前绝对 beam HiST-Beam/P3 路径；新增 run 通过 metadata 与旧 run 分离，不需要迁移旧输出。

## Open Questions

- residual head 的第一个实现应只预测单步 `last_beam -> future_beam[h]`，还是为每个 horizon 使用不同 `last_beam` 定义；建议先使用样本历史窗口最后一个 beam 对所有 future horizon 计算 delta，保持实现简单且符合当前 dataset 结构。
- private calibration 首版选择 absolute logit bias/temperature，还是 residual-domain offset；建议先实现 bias+temperature，再把 offset/prototype correction 作为后续任务。
- Markov delta baseline 是否按 source train、target_adapt unlabeled 还是 target_adapt labeled 估计；建议同时报告 source Markov 和 target_adapt labeled Markov，并在 metadata 中标明使用的数据。
