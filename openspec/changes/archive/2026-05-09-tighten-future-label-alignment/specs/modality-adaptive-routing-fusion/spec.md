## MODIFIED Requirements

### Requirement: MARF fusion model construction
系统 MUST 提供可通过 registry 构建的 MARF fusion student。该模型 MUST 使用项目固定模态顺序，支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意非空有效组合，并 MUST 保持现有 `experiment.task: fusion` 输入契约。配置 `model.num_pred: N` 时，MARF MUST 直接输出 `N` 个 future prediction slot。

#### Scenario: 构建五模态 MARF
- **WHEN** 用户配置 `model.student.type: marf_fusion` 且 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 构建五个模态 encoder 或 projector
- **AND** 模型 MUST 接收现有 fusion 输入键对应的五个张量
- **AND** 模型 MUST 输出 beam logits

#### Scenario: 构建任意子集 MARF
- **WHEN** 用户配置 `marf_fusion` 且 `modalities` 为任意合法非空模态组合
- **THEN** 模型 MUST 只要求该组合对应的输入张量
- **AND** 输出 diagnostics 中的模态顺序 MUST 与标准化后的 `modalities` 一致

#### Scenario: MARF horizon 对齐现有标签
- **WHEN** 配置中的 `model.num_pred` 为 `N`
- **THEN** MARF logits MUST 输出 `N` 个 prediction slot
- **AND** 这些 slot MUST 能直接传入现有 `select_prediction_slots()` 与 `prepare_labels()` 结果对齐
- **AND** MARF MUST 不再输出用于当前或历史最后一个 beam 的额外 prediction slot

### Requirement: MARF forward output contract
MARF forward MUST 返回包含主 logits 与路由诊断的 dict。输出张量 MUST 能支持训练、验证、subset 评估和 TensorBoard 诊断，并且所有 horizon-wise 张量的 prediction slot 数 MUST 等于 `model.num_pred`。

#### Scenario: 输出核心张量
- **WHEN** batch size 为 `B`、启用模态数为 `K`、历史长度为 `T`、hidden 维度为 `D`、配置 `model.num_pred` 为 `N`、beam 类别数为 `C`
- **THEN** `outputs["logits"]` MUST 具有形状 `[B, N, C]`
- **AND** `outputs["token_features"]` MUST 能表示为 `[B, K, T, D]`
- **AND** `outputs["anchor_weights"]` 和 `outputs["residual_weights"]` MUST 具有形状 `[B, N, K]`
- **AND** `outputs["h_anchor"]` 和 `outputs["h_final"]` MUST 具有形状 `[B, N, D]`

#### Scenario: 输出路由诊断
- **WHEN** MARF 完成 forward
- **THEN** 输出 MUST 包含 `anchor_logits`、`anchor_weights`、`residual_logits`、`residual_weights`、`residual_delta`、`effective_modality_mask`、`prior` 和 `modalities`
- **AND** `adapt_model_output()` MUST 能从该 dict 中解析 logits、input features、output features 和 diagnostics

#### Scenario: 旧 horizon 配置不被接受
- **WHEN** MARF 主 logits、anchor weights 或 residual weights 的 prediction slot 数为 `num_pred + 1`
- **THEN** MARF 定向测试或训练 shape 检查 MUST 将其视为 horizon 契约错误
- **AND** 系统 MUST 不把第一个 slot 当作历史 beam 静默丢弃
