# modality-visual-diagnostics Specification

## Purpose
Define the compatibility and manifest-preparation behavior for modality visualization diagnostics after the primary user workflow has moved to the Gradio viewer.
## Requirements
### Requirement: 诊断入口与配置
系统 MUST 将模态诊断入口收敛到 viewer manifest 数据准备和 JEPA visual analysis。旧静态 PNG 报告入口、仓库级 Gradio viewer support、`tools/visualization/` 启动路径和旧兼容命令 MUST 不再作为可运行工作流或安装入口保留。Manifest 导出 MAY 继续读取 `diagnostics.visualization` 配置字段以保持配置兼容，但实现 MUST 位于当前 `viewer_manifest_*` helper 边界中。

#### Scenario: 旧命令被拒绝
- **WHEN** 用户运行旧的 `the retired modality visualization command`、`the retired script entry` 或仓库级 Gradio viewer 启动路径
- **THEN** 系统 MUST 拒绝该入口或不再提供该入口
- **AND** 错误信息或文档 MUST 指向 `kd-sensing-export-viewer-manifest` 和 `kd-sensing-jepa-visual-analysis`

#### Scenario: manifest 导出使用 canonical 入口
- **WHEN** 用户准备 viewer manifest
- **THEN** 用户 MUST 使用 `kd-sensing-export-viewer-manifest` 或对应包内 CLI
- **AND** 该入口 MUST 不要求生成 PNG 总览图或启动 Gradio Web UI 作为成功条件

#### Scenario: 不修改训练配置
- **WHEN** 用户使用训练配置准备 viewer manifest
- **THEN** 系统 MUST 读取现有训练配置中的 dataset、scene、split、序列长度、预测长度、启用模态和 cache policy
- **AND** 系统 MUST 不修改原始训练配置文件

### Requirement: 复用真实处理后张量
Manifest 数据准备能力 MUST 基于 Dataset 实际返回的处理后张量或由同一预处理/cache policy 产生的处理后文件路径生成样本记录，确保 manifest 内容与训练、验证或评估输入一致。系统 MUST 不用单独的旁路预处理逻辑替代 Dataset 的 RGB/ImageNet image、radar RA/DA、LiDAR BEV、GPS 和 mmWave 处理。模型预测导出能力写出的 future distribution MUST 与 `prepare_labels()` 的 future-only 语义一致。

#### Scenario: image manifest 使用 RGB/ImageNet 来源
- **WHEN** manifest 导出启用 image 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 raw image reference 和 processed RGB/ImageNet image 表示
- **AND** 样本记录 MUST 标明 processed image 与训练输入一致或记录其导出来源

#### Scenario: radar manifest 使用 RA/DA 来源
- **WHEN** manifest 导出启用 radar 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 processed radar RA/DA 表示或其可视化文件
- **AND** 样本记录 MUST 保留足够元数据用于外部查看器区分 raw radar 与 processed radar

#### Scenario: LiDAR manifest 使用 BEV 来源
- **WHEN** manifest 导出启用 LiDAR 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 LiDAR BEV 表示或其可视化文件
- **AND** 样本记录 MUST 包含总体非零率或通道级非零率等诊断摘要

#### Scenario: GPS 和 mmWave manifest 使用数值序列
- **WHEN** manifest 导出启用 GPS 或 mmWave 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 GPS 或 mmWave 数值序列、JSON 文件或外部 viewer 可解析的中间文件
- **AND** 记录 MUST 包含足够元数据让外部工具渲染轨迹、曲线、bar chart 或 heatmap

#### Scenario: 模型预测导出 beam distribution
- **WHEN** 用户使用模型预测导出能力生成单模态预测诊断
- **THEN** 输出预测文件 MUST 为每个样本和模态写入 `beam_distribution[modality].prob`
- **AND** 如果 logits 可用，输出预测文件 MUST 同时写入 `beam_distribution[modality].logit`
- **AND** `prob` MUST 来自 softmax 后分布，`logit` MUST 来自 softmax 前输出，shape MUST 为 `[num_pred, num_beams]`
- **AND** `beam_distribution[modality].prob[0]`、`confidence_curves[modality][0]` 和 `prediction.modalities[modality].future_labels[0]` MUST 共同表示 `t+1`
- **AND** Manifest 合并 MUST 将该字段保留到样本顶层，供外部 viewer 或离线诊断读取

#### Scenario: 模型预测导出不执行旧 slot 偏移
- **WHEN** 已经通过统一 helper 得到与 future labels 对齐的模型预测
- **THEN** viewer prediction export MUST 使用完整的 `num_pred` 个预测 slot 写出 payload
- **AND** 导出器 MUST 不执行 `probs[1:]`、`logits[1:]` 或 `labels[1:]` 风格的旧 current-slot 偏移

### Requirement: 只读诊断行为
Manifest 数据准备入口和 JEPA visual analysis MUST 默认保持只读输入行为，不得修改训练 checkpoint、训练日志、评估报告或已存在的 split CSV。对 LiDAR BEV cache 的访问 MUST 遵循现有 cache policy；当 policy 为 `read_only` 或 `off` 时，manifest 导出不得写入新的 LiDAR BEV cache 文件。系统 MUST 不再读取或写入 image motion cache。

#### Scenario: Manifest 导出不修改训练产物
- **WHEN** 用户对已有训练配置运行 manifest 导出
- **THEN** 系统 MUST 不修改该训练运行目录中的 checkpoint、`train_log.json`、`metrics.json` 或 `final_config.yaml`
- **AND** 所有新产物 MUST 写入用户指定的 viewer manifest 输出目录

#### Scenario: read_only cache 不写入
- **WHEN** 用户设置 `data.cache.policy: read_only`
- **THEN** manifest 导出 MUST 允许读取已有 LiDAR BEV cache
- **AND** LiDAR BEV cache miss 时系统 MUST 在线计算当前样本所需处理结果但不得写入新 cache 文件
- **AND** manifest 导出 MUST 不读取或写入 image motion cache

#### Scenario: off cache 不访问 cache 文件
- **WHEN** 用户设置 `data.cache.policy: off`
- **THEN** manifest 导出 MUST 禁用 LiDAR BEV cache 的读取与写入
- **AND** 系统 MUST 仍能通过在线处理生成 viewer 所需记录或明确记录对应模态不可用
- **AND** manifest 导出 MUST 不访问 image motion cache 文件

### Requirement: 可视化诊断内部轻量模块边界
Manifest 诊断内部 MUST 区分轻量 helper 与重依赖运行模块。配置解析、采样候选选择、metadata 写出和 JSON payload 规范化 MUST 保持轻量导入，并 MUST 位于 `kd_sensing.diagnostics.viewer_manifest_*` 或等价当前命名模块中；旧 `kd_sensing.diagnostics.visualization` 包名 MUST 不再作为当前 helper 边界。数据集构建、processed asset 生成和模型预测导出 MAY 导入 pandas、torch、PIL 或 engine builders，但这些依赖 MUST 限定在对应职责模块或函数内。

#### Scenario: 解析 viewer 诊断配置不导入渲染依赖
- **WHEN** 开发者导入或调用 viewer manifest 配置解析 helper
- **THEN** 系统 MUST 能解析 `diagnostics.visualization` 的输出目录、splits、sample count、seed、filters、modalities 和 scene comparison 配置
- **AND** 该路径 MUST 不导入 matplotlib、旧 PNG render helper 或 `kd_sensing.diagnostics.visualization.core`

#### Scenario: 采样 helper 不读取 dataset
- **WHEN** 开发者导入采样 helper 并传入候选记录
- **THEN** 系统 MUST 能按 seed、seq_index、label 和 per-seq sample count 选择样本
- **AND** 采样 helper MUST 不构建 dataset、不读取 CSV 文件、不加载 checkpoint

#### Scenario: 写出 helper 只负责文件序列化
- **WHEN** 开发者调用 JSON、JSONL 或 CSV metadata 写出 helper
- **THEN** helper MUST 只负责目标路径创建和 payload 序列化
- **AND** helper MUST 不导入 dataset builder、model builder、matplotlib 或 PIL

### Requirement: Manifest 行为保持兼容
收紧诊断内部 import 边界时，manifest 导出和 viewer prediction 导出的公开行为 MUST 保持兼容。输出 manifest、metadata、processed asset 路径、prediction bundle 合并和显式 cache 参数语义 MUST 不因内部模块整理而改变。默认 cache 目录 MAY 从 Gradio 命名收敛为 viewer manifest 命名。`viewer_command` MAY 不再指向仓库级 Gradio viewer，但 MUST 不指向已删除的 `tools/visualization` 路径。

#### Scenario: manifest 导出 payload 兼容
- **WHEN** 用户运行 `kd-sensing-export-viewer-manifest` 或包内 CLI 导出 viewer manifest
- **THEN** 输出 manifest MUST 保持当前字段语义
- **AND** manifest 记录 MUST 继续包含 sample id、scene、split、sequence、raw/processed assets、label、enabled modalities 和 statistics

#### Scenario: 模型预测导出兼容
- **WHEN** 用户使用 viewer 模型预测导出能力
- **THEN** 输出 prediction 文件 MUST 继续包含每个样本和模态的 beam distribution、confidence curves 和 future labels
- **AND** 导出流程 MUST 继续复用统一 runtime forward 和 future-only label 语义

