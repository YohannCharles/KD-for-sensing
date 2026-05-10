# modality-visual-diagnostics Specification

## Purpose
Define the compatibility and manifest-preparation behavior for modality visualization diagnostics after the primary user workflow has moved to the Gradio viewer.
## Requirements
### Requirement: 诊断入口与配置
系统 MUST 将现有模态可视化诊断入口从静态 PNG 报告主流程迁移为 Gradio viewer 的 manifest 数据准备或迁移提示能力。旧入口 MUST 不再作为主要用户可视化路径生成静态总览图；当保留兼容命令时，它 MUST 要么导出 viewer manifest，要么明确提示用户改用 Gradio viewer。

#### Scenario: 旧命令提示迁移路径
- **WHEN** 用户运行旧的 `kd-sensing-visualize-modalities` 或 `scripts/visualize_modalities.py`
- **THEN** 系统 MUST 提示静态可视化入口已被 Gradio viewer 替代
- **AND** 提示信息 MUST 包含 manifest 导出命令或 Gradio viewer 启动命令

#### Scenario: 兼容命令导出 manifest
- **WHEN** 项目选择短期保留旧命令作为兼容入口
- **THEN** 该命令 MUST 生成 Gradio viewer 可读取的 manifest
- **AND** 该命令 MUST 不再要求生成 PNG 总览图作为成功条件

#### Scenario: 不修改训练配置
- **WHEN** 用户使用训练配置准备 viewer manifest
- **THEN** 系统 MUST 读取现有训练配置中的 dataset、scene、split、序列长度、预测长度、启用模态和 cache policy
- **AND** 系统 MUST 不修改原始训练配置文件

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

### Requirement: 只读诊断行为
Gradio viewer 与 manifest 数据准备入口 MUST 默认保持只读行为，不得修改训练 checkpoint、训练日志、评估报告或已存在的 split CSV。对 image motion cache 和 LiDAR BEV cache 的访问 MUST 遵循现有 cache policy；当 policy 为 `read_only` 或 `off` 时，manifest 导出不得写入新的 cache 文件。

#### Scenario: Viewer 不修改训练产物
- **WHEN** 用户启动 Gradio viewer 浏览已有 manifest
- **THEN** 系统 MUST 不修改 checkpoint、`train_log.json`、`metrics.json`、`final_config.yaml` 或 split CSV
- **AND** Viewer MAY 读取 manifest 引用的图片或 JSON 文件

#### Scenario: Manifest 导出不修改训练产物
- **WHEN** 用户对已有训练配置运行 manifest 导出
- **THEN** 系统 MUST 不修改该训练运行目录中的 checkpoint、`train_log.json`、`metrics.json` 或 `final_config.yaml`
- **AND** 所有新产物 MUST 写入用户指定的 viewer manifest 输出目录

#### Scenario: read_only cache 不写入
- **WHEN** 用户设置 `data.cache.policy: read_only`
- **THEN** manifest 导出 MUST 允许读取已有 image motion mask cache 或 LiDAR BEV cache
- **AND** cache miss 时系统 MUST 在线计算当前样本所需处理结果但不得写入新 cache 文件

#### Scenario: off cache 不访问 cache 文件
- **WHEN** 用户设置 `data.cache.policy: off`
- **THEN** manifest 导出 MUST 禁用 image motion mask cache 和 LiDAR BEV cache 的读取与写入
- **AND** 系统 MUST 仍能通过在线处理生成 viewer 所需记录或明确记录对应模态不可用
