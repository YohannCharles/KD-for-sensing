# g2d-multimodal-distillation Specification

## Purpose
定义 G2D 多模态蒸馏 distiller、teacher ensemble、SMP 和诊断输出契约。
## Requirements
### Requirement: G2D distiller construction
系统 MUST 提供可通过 `distillation.type: g2d` 构建的 G2D distiller。G2D distiller MUST 支持 `lite`、`global` 和 `horizon_diagnostic` 三种 mode，并 MUST 使用 `model.num_pred` 作为唯一 prediction horizon 数。

#### Scenario: 构建 G2D-lite
- **WHEN** 用户加载 `distillation.type: g2d` 且 `distillation.g2d.mode: lite`
- **THEN** 系统 MUST 构建 G2D distiller
- **AND** distiller MUST 启用 supervised CE、feature KD 和 logit KD
- **AND** distiller MUST 不启用 SMP 梯度屏蔽

#### Scenario: 构建 G2D-global
- **WHEN** 用户加载 `distillation.type: g2d` 且 `distillation.g2d.mode: global`
- **THEN** 系统 MUST 构建 G2D distiller
- **AND** distiller MUST 启用 supervised CE、feature KD、logit KD 和 global SMP 梯度屏蔽

#### Scenario: 构建 G2D-horizon diagnostics
- **WHEN** 用户加载 `distillation.type: g2d` 且 `distillation.g2d.mode: horizon_diagnostic`
- **THEN** 系统 MUST 构建 G2D distiller
- **AND** distiller MUST 输出 horizon-wise teacher confidence 和 ranking
- **AND** distiller MUST 不执行 per-horizon backward 或 per-horizon 梯度屏蔽

### Requirement: G2D future-only shape contract
G2D 训练、损失、teacher confidence、metrics 和 diagnostics MUST 统一使用 future-only shape。student logits 和 teacher logits MUST 为 `[B, H, C]`，labels MUST 为 `[B, H]`，其中 `H == model.num_pred` 且默认 `H == 3`，horizon 名称 MUST 为 `t+1`、`t+2`、`t+3`。

#### Scenario: 接收三步 future logits
- **WHEN** G2D 收到 student logits `[B,3,64]`、teacher logits `[B,3,64]` 和 labels `[B,3]`
- **THEN** 系统 MUST 将三个 slot 分别解释为 `t+1`、`t+2`、`t+3`
- **AND** supervised CE、feature KD、logit KD 和 confidence 计算 MUST 使用这三个 slot

#### Scenario: 拒绝旧四步 logits
- **WHEN** 任一 G2D teacher 输出 logits `[B,4,64]`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 系统 MUST 不把第一个 slot 当作历史 beam 静默丢弃

#### Scenario: 拒绝 label horizon 不匹配
- **WHEN** labels 的 horizon 数与 `model.num_pred` 不一致
- **THEN** G2D loss MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含期望 horizon 和实际 shape

### Requirement: G2D teacher ensemble
系统 MUST 为 G2D 构建多个单模态 teacher。每个 teacher MUST 从配置或 checkpoint registry 加载权重，MUST 使用对应单模态输入前向，MUST 处于 eval 模式，并且所有 teacher 参数 MUST 设置为 `requires_grad=False`。

#### Scenario: 加载五个单模态 teacher
- **WHEN** G2D 配置列出 image、radar、gps、lidar 和 mmWave teacher
- **THEN** 系统 MUST 构建五个单模态 teacher 模型
- **AND** 系统 MUST 为每个 teacher 加载 checkpoint 或从 registry 解析 checkpoint
- **AND** 每个 teacher forward MUST 返回可由 `adapt_model_output()` 解析的输出

#### Scenario: teacher checkpoint 缺失
- **WHEN** G2D 启用且某个 teacher checkpoint 无法解析或文件不存在
- **THEN** 训练启动 MUST 失败
- **AND** 错误信息 MUST 包含缺失 teacher 的模态名和解析来源

#### Scenario: teacher 不参与训练
- **WHEN** G2D 执行一个训练 batch
- **THEN** 所有 teacher forward MUST 在 `torch.no_grad()` 中执行
- **AND** teacher 参数 MUST 不产生梯度

### Requirement: G2D loss components
G2D loss MUST 计算 supervised CE、feature KD 和 logit KD，并按配置权重组合为总损失。默认 supervised weight MUST 为 1.0，feature KD 和 logit KD MUST 可通过权重单独关闭。

#### Scenario: supervised CE 使用全部 horizon
- **WHEN** student logits 为 `[B,3,64]` 且 labels 为 `[B,3]`
- **THEN** supervised CE MUST 在展平后的 `[B*3,64]` logits 和 `[B*3]` labels 上计算
- **AND** ignore index 语义 MUST 与现有 task criterion 兼容

#### Scenario: logit KD 对所有 teacher 求平均
- **WHEN** G2D 收到多个 teacher logits
- **THEN** logit KD MUST 对每个 teacher 与 student logits 计算 temperature KL
- **AND** 总 logit KD MUST 对启用 teacher 的 KL 取平均
- **AND** teacher logits MUST detach

#### Scenario: feature KD 自动投影
- **WHEN** student modality feature 与 teacher feature 维度不同且配置启用 auto projection
- **THEN** G2D MUST 为该模态创建 trainable projection
- **AND** feature KD MUST 在投影后的 student feature 与 detached teacher feature 上计算

### Requirement: Teacher confidence and ranking
G2D MUST 根据 teacher 对真实 label 的 softmax probability 计算 teacher confidence。系统 MUST 输出每个模态在 `t+1`、`t+2`、`t+3` 和三步平均上的 confidence，并 MUST 生成 weak-to-strong modality ranking。

#### Scenario: 计算 horizon confidence
- **WHEN** teacher logits 为 `[B,3,64]` 且 labels 为 `[B,3]`
- **THEN** `confidence[m,h]` MUST 等于该模态 teacher 在第 h 个 horizon 上对真实 label 的平均 softmax probability
- **AND** `h=0`、`h=1`、`h=2` MUST 分别对应 `t+1`、`t+2`、`t+3`

#### Scenario: 生成弱到强排序
- **WHEN** 每个模态都有三步 teacher confidence
- **THEN** 系统 MUST 按三步平均 confidence 从低到高生成 `avg` ranking
- **AND** `horizon_diagnostic` mode MUST 额外按每个 horizon 生成 `t+1`、`t+2`、`t+3` ranking

### Requirement: Sequential Modality Prioritization
G2D-global MUST 使用 teacher confidence 排序驱动 Sequential Modality Prioritization。SMP MUST 在 backward 后、optimizer step 前屏蔽 inactive modality encoder 的梯度，并 MUST 保留 active modality、fusion module 和 prediction head 的梯度。

#### Scenario: 按弱到强调度 active modality
- **WHEN** confidence 平均值排序为 `image, radar, lidar, gps, mmwave` 且 `per_modality_tau: 2`
- **THEN** epoch 0 和 1 的 active modalities MUST 为 `["image"]`
- **AND** epoch 2 和 3 的 active modalities MUST 为 `["radar"]`
- **AND** epoch 4 和 5 的 active modalities MUST 为 `["lidar"]`
- **AND** epoch 6 和 7 的 active modalities MUST 为 `["gps"]`
- **AND** epoch 8 和 9 的 active modalities MUST 为 `["mmwave"]`
- **AND** epoch 10 及之后的 active modalities MUST 为全部模态

#### Scenario: 屏蔽 inactive encoder 梯度
- **WHEN** active modalities 为 `["image"]`
- **THEN** image encoder 参数梯度 MUST 保留
- **AND** radar、gps、lidar 和 mmWave encoder 参数梯度 MUST 清零
- **AND** fusion module 与 prediction head 参数梯度 MUST 保留

### Requirement: G2D diagnostics artifact
G2D MUST 在每个 epoch 保存 JSON 诊断文件。诊断文件 MUST 包含 `num_pred`、`horizon_names`、teacher confidence、weak-to-strong ranking、active modalities 和 loss breakdown。

#### Scenario: 保存 epoch diagnostics
- **WHEN** G2D 训练完成一个 epoch
- **THEN** 系统 MUST 在运行目录下保存 `diagnostics/g2d_epoch_<epoch>.json`
- **AND** JSON MUST 包含 `horizon_names: ["t+1", "t+2", "t+3"]`
- **AND** JSON MUST 包含 `loss.supervised`、`loss.feature_kd`、`loss.logit_kd` 和 `loss.total`

#### Scenario: student branch confidence 可用时输出 ratio
- **WHEN** student 输出包含 per-modality branch logits 或 unimodal logits
- **THEN** G2D diagnostics MUST 保存 student branch confidence
- **AND** G2D diagnostics MUST 保存 student branch confidence 与 teacher confidence 的 ratio
### Requirement: G2D 支持包含 CSI 的模态集合
G2D teacher ensemble、teacher confidence、ranking、SMP gradient masking 和 diagnostics MUST support any configured modality subset that is valid in the project modality registry, including subsets that contain `csi`. Existing five-modality G2D configs MUST remain valid.

#### Scenario: 构建 GPS+CSI G2D teacher ensemble
- **WHEN** G2D 配置的模态集合为 `gps` 和 `csi`
- **THEN** 系统 MUST 构建 `gps` teacher 和 `csi` teacher
- **AND** 每个 teacher MUST 使用对应单模态输入前向
- **AND** teacher checkpoint 缺失时错误信息 MUST 包含缺失的模态名

#### Scenario: CSI teacher confidence 参与排序
- **WHEN** G2D teacher logits 包含 `gps` 和 `csi` 的 `[B,H,C]` 输出且 labels 为 `[B,H]`
- **THEN** 系统 MUST 计算 `gps` 和 `csi` 对真实 label 的 teacher confidence
- **AND** weak-to-strong ranking MUST 包含 `csi`

#### Scenario: SMP 可以激活 CSI
- **WHEN** SMP 调度器将 active modalities 设置为 `["csi"]`
- **THEN** 系统 MUST 保留 CSI encoder、fusion module 和 prediction head 的梯度
- **AND** 系统 MUST 清零 inactive modality encoder 的梯度

#### Scenario: G2D diagnostics 记录 CSI
- **WHEN** G2D epoch diagnostics 写出 JSON
- **THEN** diagnostics MUST 在 teacher confidence、ranking 和 active modalities 中使用真实配置模态名
- **AND** 当配置包含 `csi` 时 diagnostics MUST 能记录 `csi` 项
