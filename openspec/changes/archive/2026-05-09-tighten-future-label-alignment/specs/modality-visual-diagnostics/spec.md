## MODIFIED Requirements

### Requirement: 复用真实处理后张量
Manifest 数据准备能力 MUST 基于 Dataset 实际返回的处理后张量或由同一预处理/cache policy 产生的处理后文件路径生成 viewer 记录，确保 Gradio 展示内容与训练、验证或评估输入一致。系统 MUST 不用单独的旁路预处理逻辑替代 Dataset 的 image motion mask、radar RA/DA、LiDAR BEV、GPS 和 mmWave 处理。模型预测导出能力写出的 future distribution MUST 与 `prepare_labels()` 的 future-only 语义一致。

#### Scenario: image manifest 使用 motion mask 来源
- **WHEN** manifest 导出启用 image 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 raw image reference 和 processed image motion mask 表示
- **AND** 样本记录 MUST 标明 processed image 与训练输入一致或记录其导出来源

#### Scenario: radar manifest 使用 RA/DA 来源
- **WHEN** manifest 导出启用 radar 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 processed radar RA/DA 表示或其可视化文件
- **AND** 样本记录 MUST 保留足够元数据用于 Gradio viewer 区分 raw radar 与 processed radar

#### Scenario: LiDAR manifest 使用 BEV 来源
- **WHEN** manifest 导出启用 LiDAR 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 LiDAR BEV 表示或其可视化文件
- **AND** 样本记录 MUST 包含总体非零率或通道级非零率等诊断摘要

#### Scenario: GPS 和 mmWave manifest 使用数值序列
- **WHEN** manifest 导出启用 GPS 或 mmWave 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 GPS 或 mmWave 数值序列、JSON 文件或 viewer 可解析的中间文件
- **AND** Viewer MUST 能将这些记录渲染为轨迹、曲线、bar chart 或 heatmap

#### Scenario: 模型预测导出 beam distribution
- **WHEN** 用户使用 viewer 的模型预测导出能力生成单模态预测诊断
- **THEN** 输出预测文件 MUST 为每个样本和模态写入 `beam_distribution[modality].prob`
- **AND** 如果 logits 可用，输出预测文件 MUST 同时写入 `beam_distribution[modality].logit`
- **AND** `prob` MUST 来自 softmax 后分布，`logit` MUST 来自 softmax 前输出，shape MUST 为 `[num_pred, num_beams]`
- **AND** `beam_distribution[modality].prob[0]`、`confidence_curves[modality][0]` 和 `prediction.modalities[modality].future_labels[0]` MUST 共同表示 `t+1`
- **AND** Manifest 合并 MUST 将该字段保留到样本顶层，供 Gradio viewer 读取

#### Scenario: 模型预测导出不执行旧 slot 偏移
- **WHEN** 已经通过统一 helper 得到与 future labels 对齐的模型预测
- **THEN** viewer prediction export MUST 使用完整的 `num_pred` 个预测 slot 写出 payload
- **AND** 导出器 MUST 不执行 `probs[1:]`、`logits[1:]` 或 `labels[1:]` 风格的旧 current-slot 偏移
