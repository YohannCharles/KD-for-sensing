## Context

当前项目已经支持 Scene 9/32 选择、统一 split、按模态加载、image motion mask cache、LiDAR BEV cache、GPS/mmWave 归一化和训练 metadata。近期 `add-sequence-split-strategies` 已把 Scene 32 从旧顺序 split 改为 `balanced_seq`，但 Scene 32 中只包含 image/radar/LiDAR 的组合仍停留在多数类基线附近。

现有代码能保存训练曲线，但缺少一个面向输入数据的诊断入口。尤其是 image 输入在训练中不是原始 RGB，而是经过 resize、灰度化、高斯平滑和相对阈值二值化的 motion mask；LiDAR 输入是 BEV 伪图像；radar 输入是 RA/DA map；GPS 和 mmWave 是数值序列。仅看训练曲线无法判断这些处理后的输入是否在 Scene 32 中过稀疏、过噪、train/test 漂移，或与标签关系弱。

## Goals / Non-Goals

**Goals:**

- 提供一个配置驱动的只读诊断入口，用于可视化 DeepSense6G 各模态“训练实际消费的处理后张量”。
- 支持 image、radar、GPS、LiDAR、mmWave 的任意启用组合，且默认复用训练配置中的 scene、CSV、split、cache policy 和预处理参数。
- 支持按 split、`seq_index`、future beam label、样本数量和 seed 抽样，便于比较 Scene 9/32、train/test 和特定失败区间。
- 输出人可读 PNG 网格和机器可读 JSON/CSV 摘要，记录样本索引、路径、标签、`seq_index`、输入 shape/dtype 和模态统计量。
- 保持诊断入口不影响训练、评估、cache 内容和已有实验结果。

**Non-Goals:**

- 不修改模型结构、loss、scheduler、指标或训练循环。
- 不引入新的 split 策略，也不自动重训任何实验。
- 不把诊断图作为训练 artifact 自动生成；诊断入口由用户显式运行。
- 不实现交互式 UI 或 notebook 服务；本变更只提供命令行/脚本和静态文件输出。
- 不用可视化路径替代 Dataset 预处理逻辑；可视化必须从 Dataset 返回的张量生成。

## Decisions

1. 新增独立诊断入口，而不是扩展训练入口。

   训练入口关注优化和 checkpoint，诊断入口关注抽样、图像布局和统计报告。独立入口可以避免训练配置变复杂，也不会让正常训练额外导入 matplotlib 或写出大体积图片。建议新增 `scripts/visualize_modalities.py` 和 `kd_sensing/diagnostics/modality_visualization.py`。

   备选方案是在每次训练开始自动保存若干样本图。这个方案会增加训练 I/O 和配置噪声，而且用户通常只在排查问题时需要这些图。

2. 复用 `build_dataset()` 和真实 Dataset `__getitem__()`。

   诊断工具应通过 `kd_sensing.engine.builders.build_dataset(cfg, split)` 构建数据集，并直接索引样本。这样 image motion mask、radar RA/DA、LiDAR BEV、GPS/mmWave 处理、cache policy、scene 默认路径和 CSV 选择都与训练一致。

   备选方案是直接读取 CSV 并调用各 transform helper。它可以更快画图，但容易遗漏 Dataset 里的归一化、cache policy、scene 解析和启用模态推导，导致诊断图与训练输入不一致。

3. 诊断配置以训练配置为主，新增 `diagnostics.visualization` 覆盖层。

   用户可以传入任意现有训练配置，再通过命令行覆盖 `data.dataset.scene`、`model.teacher.modalities`、`model.student.modalities` 或诊断字段。默认字段建议包括：

   - `output_dir`
   - `splits`
   - `sample_count`
   - `seed`
   - `seq_index`
   - `labels`
   - `modalities`
   - `layout`
   - `include_raw_image_preview`
   - `max_frames_per_sample`

   如果配置没有显式 `diagnostics.visualization`，入口使用保守默认值：train/test 各抽少量样本，输出到 `outputs/diagnostics/<scene>/<run_name_or_config_stem>/`。

4. 抽样基于 CSV 行索引，并记录可复现选择。

   Dataset 当前返回处理后张量，但不暴露每个样本的所有原始元数据。诊断工具应读取 `dataset.root_csv` 对应 CSV，用行号与 dataset 样本顺序对齐，在抽样前按 `seq_index` 和 label 过滤候选行，再用 seed 确定样本。输出摘要必须记录 `dataset_index`、CSV 行字段、`seq_index`、历史 beam、future beam、原始路径和 split metadata 摘要。

   备选方案是只按 dataset index 抽样。这个方案实现简单，但用户难以定位图像对应的原始轨迹、标签和路径。

5. 可视化分两类输出：单样本面板和聚合统计。

   单样本 PNG 用固定布局展示：

   - image：按时间排列 motion mask；可选 raw RGB thumbnail 仅作参考，不能替代 processed input。
   - radar：RA/DA 每个时隙 heatmap，可默认显示最后若干时隙。
   - LiDAR：BEV 三通道和通道拆分图。
   - GPS：历史轨迹折线和相对极坐标数值。
   - mmWave：时间 x beam-index heatmap。
   - label：历史 beam 和 future beam 文本/条形摘要。

   聚合统计输出 `summary.json` 和 `samples.csv`，并可额外生成每个 split 的统计 PNG，例如 image density 直方图、radar/LiDAR 非零率箱线图、label 分布条形图。

6. 统计字段优先保持简单稳定。

   每个样本和每个模态记录 `shape`、`dtype`、`min`、`max`、`mean`、`std`、`nonzero_fraction`。模态特定统计包括 image mask density、radar RA/DA mean/std、LiDAR channel nonzero fraction、GPS per-dimension range、mmWave per-time mean/std。复杂特征距离或模型 embedding 可留给后续变更。

## Risks / Trade-offs

- [Risk] 诊断工具复用 Dataset 后，启用 LiDAR/GPS/mmWave 归一化时可能需要 train-fitted scaler 或 normalizer。  
  Mitigation: 默认只构建单 split 时禁用需要跨 split fit 的测试归一化，或按 train 后 test 的顺序复用 `build_dataloaders()` 中已有的 scaler/normalizer 逻辑；任务中明确覆盖测试。

- [Risk] 输出 PNG 可能较大，批量样本会占用磁盘。  
  Mitigation: 默认样本数很小，支持 `sample_count`、`max_frames_per_sample` 和显式输出目录；摘要文件记录实际输出文件列表。

- [Risk] 可视化布局过于拥挤，导致诊断价值下降。  
  Mitigation: 采用每个样本一个总览图，同时为复杂模态输出可选单模态图；测试只约束文件存在和核心元数据，不固定像素级布局。

- [Risk] raw image preview 容易让用户误以为模型使用 RGB。  
  Mitigation: 图中明确分区命名为 `processed image motion mask` 和 `raw image reference`；摘要中记录 `raw_reference_only: true`。

- [Risk] 直接用 matplotlib 在无显示环境运行失败。  
  Mitigation: 强制使用非交互式 backend，并在 CLI smoke test 中覆盖 headless 环境。

## Migration Plan

1. 新增诊断模块、CLI/script、默认配置和 README 使用说明。
2. 用 Scene 32 小样本运行 smoke test，确认能从现有统一 split 和 cache 生成 PNG/JSON/CSV。
3. 用 synthetic 或临时小数据添加单元测试，覆盖抽样、输出文件、统计字段和缺失模态跳过行为。
4. 不需要数据迁移；回滚时删除新诊断入口和配置即可，不影响训练产物。
