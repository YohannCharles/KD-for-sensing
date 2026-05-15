## Context

当前项目已经具备 DeepSense6G Scenario 31 的五模态数据加载、mmWave 64 维 power 特征、GPS-Rel-Polar 特征、LiDAR BEV、CLS-token Transformer fusion，以及统一的训练/验证/评估入口。现有监督目标仍是未来 beam 分类：dataset 返回 `input_beam` 和 `target_beam`，模型主输出为 `[B, H, C]` beam logits，训练和评估围绕 CE、Top-K、DBA 展开。

arXiv:2603.25799 把 receive beam、blockage probability 和 2D position 放在一个 Transformer fusion 网络里联合预测。该论文的关键可迁移点是：当前 sweep power vector 只用于生成监督标签和评估，推理输入只使用历史/上一时刻 radio context；遮挡标签用训练集最大接收功率的固定分位数阈值自动生成；位置头从融合 CLS 表示回归二维局部坐标。

本变更应顺着现有工程结构扩展，而不是新增训练脚本或替换 fusion 主干。默认 beam-only 配置必须保持兼容。

## Goals / Non-Goals

**Goals:**

- 为 DeepSense6G dataset 增加可选的 `occlusion_label`、`position_target` 和有效 mask，支持按 horizon 与 `target_beam` 对齐。
- 为 CLS-token Transformer fusion 增加可选遮挡检测头和位置估算头，输出进入 `ModelOutput.diagnostics`，主 logits 契约保持不变。
- 在训练流程中加入可配置的多任务损失：beam CE/KD 主损失 + 遮挡 BCE + 位置 MSE。
- 在验证/评估中输出遮挡 accuracy/F1 和位置 RMSE，并写入 metrics、epoch log 与 TensorBoard 标量。
- 提供五模态 recommended 配置或 overlay，便于复现实验。

**Non-Goals:**

- 不实现论文中的离线 LiDAR map 构建或 SLAM scan-to-map 优化；本变更只做二维位置监督和数值指标。
- 不把当前/未来 mmWave sweep vector 作为推理输入；它只用于标签、阈值和评估，避免标签泄漏。
- 不要求所有单模态模型都立即支持 auxiliary heads；首期支持 CLS-token Transformer fusion，后续模型可复用同一输出/损失契约扩展。
- 不改变现有 beam-only early stopping、checkpoint registry 和 KD 默认行为。

## Decisions

### Decision 1: 遮挡标签从 future beam power 文件派生

实现时复用 `future_beam_paths` 指向的 64-beam sweep 文件，读取 raw power vector，计算 `p_max = max(power)`，按训练 split 的 `threshold_percentile` 生成阈值 `tau`，标签为 `p_max < tau`。默认分位数使用 20，与论文示例一致；阈值必须来自训练 split，并复用于 test/eval。

Rationale: 现有序列 CSV 已包含 future beam 文件路径，beam label 也从这些文件 `argmax` 派生，因此不需要新增 future mmWave 输入列。这样能保持当前 CSV 基本可用，并避免把当前监督 sweep 放进模型输入。

Alternatives considered:

- 使用历史 `mmwave` 输入的最后一帧生成遮挡标签：实现简单，但监督目标会与未来 beam label 不对齐。
- 使用固定 dB 阈值：跨场景和预处理尺度更脆弱，不如训练分位数稳定。

### Decision 2: 位置目标使用 future GPS local XY，并通过预处理显式补列

位置监督默认要求序列 CSV 包含 `future_gps1..future_gpsH` 和 `future_bs_gps1..future_bs_gpsH`，dataset 将 UE/BS GPS 读成经纬度后转换为 UTM-like 坐标，并返回 `ue_xy - bs_xy` 的二维米制目标。配置可允许 fallback 到最后一个历史 GPS 作为 smoke test 目标，但正式多任务实验必须使用 future GPS target。

Rationale: 项目现有 GPS 输入是 `[distance, sin(theta), cos(theta)]`，它适合作为模型输入但不适合作为二维位置回归目标。显式 future GPS target 可以和 `target_beam` 的 horizon 对齐，也避免从输入 GPS 直接复制标签。

Alternatives considered:

- 直接回归 GPS-Rel-Polar 三维特征：复用代码更多，但不符合“二维位置估算头”的目标，也难以用米制 RMSE 解释。
- 从 metadata 解析 frame index 后回查原 CSV：实现复杂、容易引入场景耦合；预处理补列更清晰。

### Decision 3: 辅助输出放入模型 dict，不改变主 logits

CLS-token Transformer fusion 在 `auxiliary_heads.enabled` 时从 CLS/future representation 输出：

- `occlusion_logits`: `[B, H]`
- `position`: `[B, H, 2]`

主 `logits` 仍为 `[B, H, C]`。`adapt_model_output()` 不需要把这些字段提升为主结构，训练/评估 helper 从 diagnostics 读取即可。

Rationale: 现有 distiller、Top-K、DBA、checkpoint 和模型适配器围绕主 logits 工作。把辅助输出留在 diagnostics 可以最小化破坏面，同时保留测试和扩展空间。

Alternatives considered:

- 新增 `ModelOutput` 字段：类型更显式，但会触及更多调用点。
- 返回 tuple 扩展：和当前 dict diagnostics 风格不一致，容易破坏旧模型。

### Decision 4: 多任务 loss 作为训练扩展 helper 接入主循环

新增 `prepare_auxiliary_targets()` 与 `compute_auxiliary_multitask_loss()` 一类窄 helper。训练时先保留现有 distiller 产生的 `base_loss`，再按配置叠加：

- `lambda_occlusion * BCEWithLogitsLoss(occlusion_logits, occlusion_label)`
- `lambda_position * MSE(position, position_target)`，只在 `position_valid` 为真时参与。

验证时用同样 helper 计算 loss 分量，但 metrics 仍分别报告。未启用多任务或 batch/model 缺少辅助字段时，行为必须回到 beam-only。

Rationale: 这样可兼容 no-KD、logits KD、RKD、G2D/CRAF/MARF 等当前训练路线，不需要重写 distiller API。

Alternatives considered:

- 注册一个新的 `multi_task_cross_entropy` loss：实现集中，但现有 distiller 期望 task criterion 是 beam CE，会引入更多接口变化。
- 在模型内部计算 loss：会把训练策略塞进模型，不符合当前工程边界。

### Decision 5: 阈值、scaler 和运行配置进入 artifact/metadata

训练 split 拟合出的 `occlusion_threshold`、`position_target_scaler`（若启用归一化）必须保存到运行目录，并在 test/eval dataset 构建时复用。最终配置和运行日志记录阈值来源、分位数、正类比例和位置坐标模式。

Rationale: 遮挡标签和位置归一化都是数据依赖状态；不记录会导致训练/评估不可复现。

Alternatives considered:

- 每个 split 独立拟合阈值：指标不可比较，会污染 test。
- 不归一化位置目标：实现简单，但 MSE 尺度可能压过 beam/occlusion loss。

## Risks / Trade-offs

- [Risk] 遮挡正类比例过低导致 BCE 学不到 blocked class -> Mitigation: 支持 `pos_weight: auto`，从训练 split 统计正负样本并记录到日志。
- [Risk] 位置 MSE 尺度压过 beam CE -> Mitigation: 默认使用较小 `lambda_position`，可选训练集位置标准化，并在日志中分开记录 loss 分量。
- [Risk] 未来 GPS target 列缺失 -> Mitigation: 启用 position head 时给出明确错误，预处理任务补充 `include_position_targets` 选项；仅 smoke test 可启用 `last_input_gps` fallback。
- [Risk] 读取 future beam power 生成遮挡标签增加 I/O -> Mitigation: 使用与 beam label cache 类似的轻量 power-stat cache，只缓存 `max_power` 和标签，不缓存大模态数组。
- [Risk] 旧配置被辅助字段影响 -> Mitigation: 所有新字段默认关闭；未启用时 dataset 不读取 future GPS，也不拟合遮挡阈值。

## Migration Plan

1. 增加 mmWave power-stat helper 和 dataset 轻量缓存，先用单元测试验证 `argmax` 标签与 `p_max` 标签生成可并存。
2. 扩展序列预处理，增加 `future_gps*` / `future_bs_gps*` 输出开关，并补充 CSV fixture 测试。
3. 扩展 DeepSense6G dataset 的可选辅助目标返回与训练 split artifact 保存/复用。
4. 扩展 CLS-token Transformer fusion 的可选 auxiliary heads 与 output diagnostics。
5. 扩展训练/验证/评估 helper、metrics、日志和 TensorBoard 标量。
6. 增加五模态多任务 canonical/overlay 配置，并运行小数据 smoke test。

Rollback strategy: 关闭 `auxiliary_heads.enabled` 和 dataset 的 multi-task target 配置即可回到 beam-only；新增代码路径默认不生效。

## Open Questions

- 正式实验是否需要对 position target 做 standardization 后训练、反归一化后评估；建议默认启用 target standardization，但实现时以 smoke test 稳定性确认。
- Scenario 31 的 CSV 是否已经有可用 future GPS/BS GPS 路径；若没有，需要先运行新增预处理开关生成统一 split。
- early stopping 默认继续使用 `val_adba`，还是为多任务实验增加 `val_multitask_score`；建议首期保持 `val_adba`，辅助指标用于诊断。
