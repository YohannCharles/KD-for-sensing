## MODIFIED Requirements

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
Manifest 数据准备能力 MUST 基于 Dataset 实际返回的处理后张量或由同一预处理/cache policy 产生的处理后文件路径生成 viewer 记录，确保 Gradio 展示内容与训练、验证或评估输入一致。系统 MUST 不用单独的旁路预处理逻辑替代 Dataset 的 image motion mask、radar RA/DA、LiDAR BEV、GPS 和 mmWave 处理。

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
- **AND** 样本记录 SHOULD 包含总体非零率或通道级非零率等诊断摘要

#### Scenario: GPS 和 mmWave manifest 使用数值序列
- **WHEN** manifest 导出启用 GPS 或 mmWave 模态
- **THEN** 输出样本记录 MUST 引用 Dataset 对应的 GPS 或 mmWave 数值序列、JSON 文件或 viewer 可解析的中间文件
- **AND** Viewer MUST 能将这些记录渲染为轨迹、曲线、bar chart 或 heatmap

#### Scenario: 模型预测导出 beam distribution
- **WHEN** 用户使用 viewer 的模型预测导出能力生成单模态预测诊断
- **THEN** 输出预测文件 SHOULD 为每个样本和模态写入 `beam_distribution[modality].prob`
- **AND** 如果 logits 可用，输出预测文件 SHOULD 同时写入 `beam_distribution[modality].logit`
- **AND** `prob` MUST 来自 softmax 后分布，`logit` MUST 来自 softmax 前输出，shape SHOULD 为 `[H, num_beams]`
- **AND** Manifest 合并 MUST 将该字段保留到样本顶层，供 Gradio viewer 读取

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

## REMOVED Requirements

### Requirement: 可复现样本选择
**Reason**: Gradio viewer 通过 manifest 中的有序样本和页面过滤完成浏览，不再以静态诊断运行时抽样作为主能力。
**Migration**: 需要固定样本集合时，在 manifest 导出阶段生成确定性样本清单，并由 viewer 按 manifest 顺序浏览。

### Requirement: 诊断产物结构
**Reason**: 静态 PNG 总览图、样本 CSV/JSONL、运行级 `summary.json` 和配置快照不再是主验收产物。
**Migration**: 使用 Gradio viewer manifest、README、示例 manifest 和交互页面作为新主产物；必要的导出摘要可作为 manifest 附属元数据。

### Requirement: 跨 split 与跨场景比较
**Reason**: 旧能力要求一次静态运行生成多个 split/scene 的文件树，已被 viewer 的 scene/split 过滤和 manifest 合并方式替代。
**Migration**: 在 manifest 中包含多个 scene/split 的样本记录，用户通过 Gradio 控件切换查看。

### Requirement: 按序列分层采样
**Reason**: 分层采样是旧静态报告的抽样策略，不再属于 Gradio viewer 的核心浏览能力。
**Migration**: 若仍需每个 `seq_index` 固定样本数，应在 manifest 导出脚本中实现并记录导出策略。

### Requirement: 聚合 split 统计产物
**Reason**: `split_stats.json` 不再是主可视化页面的必需产物。
**Migration**: 可将 split 级统计作为 manifest 的附属文件或 `extra` 信息提供给 Gradio Diagnostics 区域。

### Requirement: 跨场景同口径比较
**Reason**: 多 scene 静态输出目录隔离不再是主交互方案的核心要求。
**Migration**: 多 scene 样本应合并或分别导出为 manifest，并通过 viewer 的 scene 下拉选择器查看。

### Requirement: 可读性改进的样本总览图
**Reason**: 单样本 PNG 总览图被 Gradio 页面中的 raw、processed 和 diagnostics 分区替代。
**Migration**: 将 raw image reference、processed motion mask、radar RA/DA、LiDAR BEV 和统计摘要展示在 Gradio 页面中。

### Requirement: 元数据产物不覆盖历史运行
**Reason**: 旧静态运行的多文件元数据命名策略不再适用于主 viewer。
**Migration**: Manifest 导出脚本 MUST 避免无提示覆盖用户指定的 manifest；具体命名策略由导出脚本文档定义。
