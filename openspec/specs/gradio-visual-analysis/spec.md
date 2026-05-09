# gradio-visual-analysis Specification

## Purpose
Define the Gradio Blocks multimodal viewer for manifest-driven sample browsing, cache reuse, modality visualization, diagnostics, and optional viewer dependencies.
## Requirements
### Requirement: Gradio 交互入口
系统 MUST 提供一个 Gradio Blocks 交互式多模态数据分析入口，用于选择数据集配置、自动处理全部样本、复用处理 cache，并浏览通信多模态融合样本。入口 MUST 支持命令行参数 `--config`、`--manifest`、`--cache-dir`、`--force-rebuild`、`--host`、`--port`、`--share` 和 `--debug`，并 MUST 能在不启动训练、不加载 checkpoint、不执行模型推理的情况下运行。

#### Scenario: 使用配置启动 viewer
- **WHEN** 用户运行 `python tools/visualization/gradio_multimodal_viewer.py --config <config> --cache-dir <dir> --host 127.0.0.1 --port 7860`
- **THEN** 系统 MUST 根据配置构建数据集并处理所选 split 的全部样本
- **AND** 系统 MUST 将 `samples.json`、`manifest_meta.json` 和 processed assets 写入 cache
- **AND** 页面 MUST 显示第一个可用样本或空数据提示

#### Scenario: 复用已有 cache
- **WHEN** 用户再次使用相同配置和相同 cache 启动 viewer
- **THEN** 系统 MUST 校验配置摘要、CSV、样本源文件和 processed assets
- **AND** 若未发生变化，系统 MUST 直接读取已有 `samples.json`，不得重新处理全部样本

#### Scenario: 使用 manifest 启动 viewer
- **WHEN** 用户运行 `python tools/visualization/gradio_multimodal_viewer.py --manifest <path> --host 127.0.0.1 --port 7860`
- **THEN** 系统 MUST 读取指定 manifest 并启动 Gradio 页面
- **AND** 页面 MUST 显示第一个可用样本或空数据提示

#### Scenario: Viewer 不执行训练推理
- **WHEN** 用户启动 Gradio viewer 或准备 viewer cache
- **THEN** 系统 MUST 不构建训练器、不加载模型 checkpoint、不写入训练日志
- **AND** 系统 MUST 只在 viewer cache 目录写入可视化资产和 manifest

### Requirement: Manifest 数据格式与路径解析
系统 MUST 支持 JSON 数组或 JSONL manifest。每条样本记录 SHOULD 包含 `sample_id`、`raw`、`processed`、`label`、`prediction`、`confidence`、`quality`、`gate` 和 `extra` 字段，但 viewer MUST 允许除样本标识和模态路径外的字段缺失。相对路径 MUST 支持相对于 manifest 所在目录解析，并 SHOULD 支持相对于项目根目录解析。

#### Scenario: 读取 JSON 数组 manifest
- **WHEN** manifest 内容是样本 dict 组成的 JSON 数组
- **THEN** 系统 MUST 返回有序样本列表
- **AND** 系统 MUST 为每个样本补充内部 view index 或 global index

#### Scenario: 读取 JSONL manifest
- **WHEN** manifest 内容是每行一个样本 dict 的 JSONL
- **THEN** 系统 MUST 逐行解析并忽略空行
- **AND** 任一坏行 MUST 被记录为错误或警告，不得导致其他合法样本不可用

#### Scenario: 解析相对路径
- **WHEN** 样本记录中的图片或 JSON 路径是相对路径
- **THEN** 系统 MUST 优先按 manifest 所在目录解析
- **AND** 如果设置了项目根目录且该路径存在，系统 MAY 使用项目根目录兜底解析

#### Scenario: Manifest 字段缺失
- **WHEN** 某条样本缺少 `prediction`、`confidence`、`quality`、`gate` 或 `extra`
- **THEN** 系统 MUST 正常渲染可用字段
- **AND** 缺失诊断面板 MUST 显示空图、空表或空 JSON，而不是报错

### Requirement: 样本过滤与时间浏览
Viewer MUST 支持按 scene、split 和 show mode 过滤样本，并通过 slider、上一帧、下一帧和自动播放按过滤后的顺序浏览样本。Show mode MUST 至少支持 `all`、`correct only`、`wrong only` 和 `low quality only`。

#### Scenario: Scene 和 split 过滤
- **WHEN** 用户选择具体 scene 或 split
- **THEN** 系统 MUST 只展示匹配 `scene_id` 和 `split` 的样本
- **AND** slider 最大值 MUST 更新为过滤后样本数量对应的范围

#### Scenario: 正确与错误样本过滤
- **WHEN** 用户选择 `correct only` 或 `wrong only`
- **THEN** 系统 MUST 根据 `prediction.correct` 过滤样本
- **AND** 缺少 `prediction.correct` 的样本 MUST 不使过滤流程失败

#### Scenario: 低质量样本过滤
- **WHEN** 用户选择 `low quality only`
- **THEN** 系统 MUST 选择至少一个模态 quality 低于阈值或平均 quality 低于阈值的样本
- **AND** 缺少 `quality` 字段的样本 MUST 被安全跳过或按空质量处理

#### Scenario: Slider 切换样本
- **WHEN** 用户拖动 sample slider
- **THEN** 页面 MUST 渲染过滤后对应 view index 的样本
- **AND** index 超出范围时 MUST 被截断到合法范围

#### Scenario: 上一帧和下一帧
- **WHEN** 用户点击上一帧或下一帧按钮
- **THEN** 系统 MUST 更新当前 index 并重新渲染样本
- **AND** 到达首尾边界时 MUST 停留在合法样本或按设计循环，不得产生非法 index

#### Scenario: 自动播放
- **WHEN** 用户启用自动播放并设置播放速度
- **THEN** Timer tick MUST 按速度推进样本 index
- **AND** 到达末尾时 MUST 循环回到第一个过滤后样本

### Requirement: Raw 与 processed 多模态展示
Viewer MUST 同屏展示 raw 和 processed 的 image、LiDAR、radar、GPS、mmWave。图片类输入 MUST 通过安全图片加载返回 PIL Image 或空状态；GPS 和 mmWave JSON MUST 转换为 Plotly 图表；无法识别或缺失的数据 MUST 显示 Missing / Not Available 空图。Raw 与 processed MUST 表达不同处理阶段；当 raw 输入实际已经是预生成中间产物时，manifest 或页面标题 MUST 明确标注其数据空间。

#### Scenario: 展示 raw 图片模态
- **WHEN** 样本包含 `raw.image`、`raw.lidar` 或 `raw.radar` 图片路径
- **THEN** Viewer MUST 加载并展示对应图片
- **AND** 文件不存在或无法打开时 MUST 显示空状态而不是中断页面

#### Scenario: LiDAR raw 不复用 processed BEV
- **WHEN** manifest 导出启用 LiDAR 且样本 CSV 中包含 LiDAR 点云路径
- **THEN** `raw.lidar` MUST 指向由点云源文件独立生成的 raw 点云俯视预览
- **AND** `processed.lidar` MUST 指向 Dataset 返回的 LiDAR BEV 可视化
- **AND** 两者 MUST NOT 通过保存同一个 `sample["lidar"]` 张量生成

#### Scenario: Radar 预生成空间标注
- **WHEN** manifest 导出启用 radar 且 CSV 中的 radar 路径指向 RA/DA 预生成数组
- **THEN** raw radar 预览 MUST 在 manifest metadata 中标注 `space: precomputed_ra_da`
- **AND** processed radar MUST 标注 `space: dataset_ra_da`
- **AND** 页面标签 SHOULD 使用 Raw / Precomputed Radar，避免暗示其为原始 radar cube

#### Scenario: 展示 processed 图片模态
- **WHEN** 样本包含 `processed.image`、`processed.lidar` 或 `processed.radar` 图片路径
- **THEN** Viewer MUST 加载并展示对应处理后图片
- **AND** 处理后图片缺失时 MUST 不影响其他模态展示

#### Scenario: 展示 GPS 轨迹
- **WHEN** GPS 数据为包含 `x/y`、`east/north` 或 `lon/lat` 的 dict/list JSON
- **THEN** Viewer MUST 用 Plotly 绘制轨迹或散点图
- **AND** 当前点 SHOULD 使用 marker 标出

#### Scenario: 展示 processed GPS 特征
- **WHEN** processed GPS JSON 包含 `features` 和 `feature_names`
- **THEN** Viewer MUST 按 time index 绘制每个 GPS 特征的时间序列
- **AND** Viewer MUST NOT 把 relative-polar `[dist, sin_theta, cos_theta]` 当作二维坐标轨迹绘制

#### Scenario: 展示 mmWave beam power
- **WHEN** mmWave JSON 包含 `beam_power`
- **THEN** Viewer MUST 绘制 beam index 到 power 的 bar chart
- **AND** beam index MUST 与 power 数组顺序一致

#### Scenario: 展示 mmWave 时序 heatmap
- **WHEN** mmWave JSON 包含 `beam_power_seq`
- **THEN** Viewer MUST 绘制 time index 到 beam index 的 heatmap
- **AND** power 值 MUST 映射为 heatmap 颜色
- **AND** 如果 JSON 包含 `scale` 或 `normalized` 字段，图标题 MUST 标明 dB、z-score 或其它数值空间

### Requirement: 诊断信息展示
Viewer MUST 展示当前样本关键元信息、标签、预测、confidence、quality、gate 和 extra。Confidence、quality 和 gate MUST 同时支持 bar chart 与 DataFrame 表格；缺失时 MUST 显示空图或空表。

#### Scenario: 展示样本信息 JSON
- **WHEN** 用户浏览某个样本
- **THEN** Viewer MUST 显示包含 `sample_id`、`scene_id`、`split`、`sequence_id`、`time_index`、`timestamp`、`label`、`prediction` 和 `extra` 的 JSON
- **AND** 缺失字段 MUST 使用空值或省略，不得报错

#### Scenario: 展示 confidence
- **WHEN** 样本包含 `confidence` dict
- **THEN** Viewer MUST 展示每个模态 confidence 的 bar chart
- **AND** Viewer MUST 展示列为 `modality` 和 `confidence` 的表格

#### Scenario: 展示 quality
- **WHEN** 样本包含 `quality` dict
- **THEN** Viewer MUST 展示每个模态 quality 的 bar chart
- **AND** Viewer MUST 展示列为 `modality` 和 `quality` 的表格

#### Scenario: 展示 gate
- **WHEN** 样本包含 `gate` dict
- **THEN** Viewer MUST 展示每个模态 gate weight 的 bar chart
- **AND** Viewer MUST 展示列为 `modality` 和 `gate` 的表格

### Requirement: Future beam 分布诊断
Viewer MUST 在 Diagnostics Tab 提供 Future Beam Distribution Inspector，用于按 horizon 检查每个模态在所有 beam label 上的完整 probability 或 logit 分布。控件 MUST 至少包含 Future Horizon、Distribution View、Chart Type 和 Show Fusion。Future Horizon MUST 根据当前样本的 `label.future_beams` 长度兼容 `t+1...t+H`，不得强制假设固定 4 个 horizon 或固定 64 个 beam。

#### Scenario: 展示 probability heatmap
- **WHEN** 当前样本包含 `beam_distribution.<modality>.prob` 且用户选择 probability + heatmap
- **THEN** Viewer MUST 展示 rows=modalities、cols=beam index 的 heatmap
- **AND** 颜色范围 MUST 固定为 0 到 1
- **AND** Viewer MUST 用红色竖线标出当前 horizon 的 GT beam
- **AND** Viewer MUST 用蓝色 marker 标出每个模态的 top1 beam

#### Scenario: 展示 per-modality 分布
- **WHEN** 当前样本包含完整分布且用户选择 per_modality
- **THEN** Viewer MUST 为每个可用模态展示一个 distribution subplot
- **AND** 每个 subplot MUST 标出 GT beam、top1 beam、GT rank 和 GT probability

#### Scenario: 展示 logit 调试视图
- **WHEN** 用户选择 logit view 且样本包含 `beam_distribution.<modality>.logit`
- **THEN** Viewer MUST 展示 softmax 前 logits，颜色范围或 y 轴范围 MAY 自动缩放
- **AND** 如果 logits 不存在，Viewer MUST 显示空图并提示 `Logits not available`

#### Scenario: 生成分布摘要表格和 detail JSON
- **WHEN** 当前样本包含 beam distribution 或旧的 per-modality prediction summary
- **THEN** Viewer MUST 输出 summary dataframe，列包含 `modality`、`horizon`、`gt_beam`、`top1_beam`、`top1_value`、`gt_value`、`gt_rank`、`top1_minus_gt`、`top1_top2_margin`、`entropy`、`is_correct` 和 `distance_to_gt`
- **AND** Viewer MUST 输出 detail JSON，包含当前 horizon、gt_beam、view_type、每个模态的 top1/GT/rank/entropy/correct/distance 信息和必要 warning

#### Scenario: 分布字段缺失或 horizon 越界
- **WHEN** 样本缺少 `beam_distribution`、缺少某个模态、fusion 不存在、logit 不存在、future_beams 缺失或 horizon 越界
- **THEN** Viewer MUST 安全降级为 empty figure、空表、跳过缺失模态或 fallback 到 `t+1`
- **AND** Gradio 回调 MUST NOT 因这些缺失字段抛出未处理异常

### Requirement: 页面布局与空状态
Viewer MUST 使用 Gradio Blocks 构建页面。页面 MUST 包含顶部控制区和用于 Overview、Raw Modalities、Processed Modalities、Diagnostics 的分区或 Tabs。空过滤结果、缺失模态和加载错误 MUST 在对应区域明确展示，不得导致 Gradio 回调失败。

#### Scenario: 页面包含核心控制区
- **WHEN** Gradio 页面加载完成
- **THEN** 页面 MUST 包含 scene 下拉、split 下拉、show mode 下拉、sample slider、样本编号、上一帧、下一帧、自动播放和播放速度控件

#### Scenario: 页面包含核心展示区
- **WHEN** Gradio 页面加载完成
- **THEN** 页面 MUST 提供 raw modalities、processed modalities 和 diagnostics 展示区域
- **AND** 这些区域 MUST 能在样本切换时同步更新

#### Scenario: 过滤结果为空
- **WHEN** 当前 scene、split 和 show mode 组合下没有样本
- **THEN** 页面 MUST 显示 `No samples found`
- **AND** 所有图片、图表、表格和 JSON 输出 MUST 返回安全空状态

### Requirement: 文档与可选依赖
系统 MUST 提供 Gradio viewer 的 README、最小 manifest 示例和交互可视化依赖说明。依赖安装说明 MUST 使用 `conda run -n kd_mm_beam` 执行 Python 命令的项目约束。

#### Scenario: README 说明运行方式
- **WHEN** 用户打开 `tools/visualization/README.md`
- **THEN** README MUST 包含安装依赖、manifest 格式、运行命令、常见问题和最小样例

#### Scenario: Viewer 依赖隔离
- **WHEN** 项目新增 Gradio viewer 依赖
- **THEN** `gradio` 和 `plotly` MUST 被记录在 viewer 专用 requirements 或 `visualization` optional extra 中
- **AND** 训练核心依赖不应被强制引入 Web UI 包
