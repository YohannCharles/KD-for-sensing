## Context

当前 `modality_visualization.py` 已复用 `build_dataset()` 与 Dataset `__getitem__()` 生成样本级 PNG、`samples.jsonl`、`samples.csv`、`summary.json` 和最终配置快照。它适合确认单个窗口的处理后张量是否能被读取，但默认随机抽样少、PNG 网格拥挤，且没有跨 `seq_index`、split 或 scene 的聚合统计。

Scene 32 的 image/radar/LiDAR 失败现象需要回答三个更具体的问题：每个 test `seq_index` 是否被覆盖、处理后张量的密度/对比度是否相对 Scene 9 或 train split 异常、label 分布是否让多数类 baseline 足以解释 top1。新变更应保持现有只读诊断入口和配置兼容，同时补足这些统计与布局能力。

## Goals / Non-Goals

**Goals:**

- 增加按 `seq_index` 分层抽样，支持每个序列输出固定数量样本。
- 输出 `split_stats.json`，按 scene/split/`seq_index` 汇总模态统计、label 分布、majority baseline 和 train/test 分布距离。
- 支持 `compare_scenes` 在同一运行中构建多个 scene 的诊断数据源，并把输出路径、摘要和统计按 scene 区分。
- 改进样本图布局，让 raw image、image motion mask、radar RA/DA 和 LiDAR BEV/通道统计更易读。
- 默认避免覆盖已有诊断元数据文件，让多次运行可以在同一个输出目录中保留历史 JSON/JSONL/CSV/YAML 记录。
- 保持旧配置可运行，新增字段缺省时不改变现有用户的核心使用方式。

**Non-Goals:**

- 不实现模型 checkpoint 推理叠加、top-k 预测、混淆矩阵或预测分布；这些作为后续可选扩展。
- 不改变 Dataset、训练 split、模型结构、预处理算法或 cache 写入策略。
- 不做交互式可视化 UI，也不引入新的图形或统计依赖。
- 不要求测试依赖真实 DeepSense6G 全量数据；单元测试继续使用临时小数据和 synthetic 张量。

## Decisions

1. 聚合统计从 Dataset 返回样本和 CSV 元数据计算。

   统计逻辑复用现有 `collect_candidates()`、`selected_csv_frame_for_dataset()` 和 `modality_statistics()`。每个样本的 image mask density、radar RA/DA std、LiDAR nonzero fraction 由 Dataset 返回张量计算；label 分布、`seq_index` 和 majority baseline 从 CSV 对齐后的候选集合计算。这样统计与训练输入一致，不需要新增旁路预处理。

   备选方案是直接扫描 CSV 原始文件和 cache 文件计算统计。它更快，但会绕过 Dataset 的归一化、启用模态判断和在线处理路径，容易与训练真实输入不一致。

2. `split_stats.json` 使用层级结构保存统计。

   输出结构按 scene slug、split、`seq_index` 分层，同时提供 split 级汇总和 train/test label 分布距离。路径由 `summary.json` 引用，并加入 `output_files`。这样既适合人工读，也方便后续脚本比较 Scene 9 与 Scene 32。

   备选方案是只扩展 `summary.json`。这会让运行摘要过大，且不利于后续单独读取聚合诊断结果。

3. 分层采样通过新增 `per_seq_sample_count` 实现。

   当配置该字段时，抽样先按过滤后的候选 `seq_index` 分组，再对每组使用稳定 seed 洗牌并截断；`sample_count` 作为全局兜底上限或在未配置分层采样时保持旧行为。输出摘要记录每个 `seq_index` 的可用数量与选中数量。

   备选方案是要求用户手工多次指定 `seq_index` 运行诊断。它能实现覆盖，但产物分散，不适合一次性比较 test split 的所有序列。

4. `compare_scenes` 在现有配置上逐 scene 覆盖运行。

   顶层入口解析 `compare_scenes` 后，为每个 scene 深拷贝配置并覆盖 `data.dataset.scene`，每个 scene 使用独立输出子目录。单 scene 配置继续走现有路径。跨 scene 总摘要只聚合各 scene 输出和 `split_stats.json`，不改变单 scene 诊断产物格式。

   备选方案是要求用户分别运行 Scene 9 和 Scene 32，再手动合并。它实现简单，但不能保证同口径配置、抽样参数和输出结构一致。

5. PNG 布局保持 matplotlib 静态文件，但按模态分区。

   渲染函数使用 `constrained_layout`、更大的 figure、短标题和按模态单独行/列计算。radar RA 与 DA 在各自序列内共享色标；LiDAR 展示 BEV 聚合图和通道非零率；image 在配置启用时把 raw RGB reference 与 motion mask 并排展示。测试只验证文件生成和元数据，不固定像素布局。

   备选方案是输出多张单模态 PNG。它能提高可读性，但会增加产物数量；本轮先优化总览图并保留后续扩展空间。

6. 元数据文件按批次选择非冲突文件名。

   单 scene 运行在写出 `samples.jsonl`、`samples.csv`、`split_stats.json`、`final_config.yaml` 和 `summary.json` 前，先检查整组目标文件是否已存在。如果任意一个基础文件已存在，系统为本轮选择统一后缀，例如 `_001`，并写出 `samples_001.jsonl`、`samples_001.csv`、`split_stats_001.json`、`final_config_001.yaml` 和 `summary_001.json`。如果 `_001` 已存在，则继续递增。干净目录的首轮运行仍使用无后缀文件名。

   备选方案是每次运行创建时间戳子目录。它也能防止覆盖，但会改变 PNG 和元数据的目录结构；当前用户反馈是图片输出已经可用，因此本轮只让机器可读元数据文件具备非覆盖命名。

## Risks / Trade-offs

- [Risk] 计算 split 级聚合统计需要遍历更多样本，真实数据上耗时可能增加。  
  Mitigation: 默认只统计诊断启用的 split 和 scene；实现保持按样本顺序遍历且只计算轻量统计，不保存额外大张量。

- [Risk] `compare_scenes` 可能与用户显式 `output_dir` 冲突，导致多 scene 产物混在一起。  
  Mitigation: 多 scene 模式下自动为每个 scene 追加 scene slug 子目录，并在 summary 中记录每个 scene 的实际输出目录。

- [Risk] 新增分层采样字段与 `sample_count` 的关系可能引起误解。  
  Mitigation: 在最终配置快照和 summary sampling 字段中同时记录 `sample_count`、`per_seq_sample_count`、候选数量和实际选中数量。

- [Risk] raw image reference 可能被误认为模型输入。  
  Mitigation: 图标题和样本记录明确标记 raw image 为 reference，统计仍以 processed image motion mask 为准。

- [Risk] 递增后缀会让用户不再总能从固定文件名读取最新结果。  
  Mitigation: CLI 返回值和 `summary.json` 继续记录本轮实际路径，`output_files` 包含全部元数据文件；首轮干净目录仍保持原文件名。

## Migration Plan

1. 更新可视化配置 dataclass、解析逻辑、默认诊断配置和最终配置快照。
2. 增加分层采样、split/scene 聚合统计与 `split_stats.json` 写出。
3. 调整样本图渲染布局，并补充 image/radar/LiDAR 面板信息。
4. 增加元数据非覆盖路径解析，并把实际路径写入返回值、summary 和 output_files。
5. 更新现有测试，覆盖 `per_seq_sample_count`、`split_stats.json`、非覆盖元数据命名和配置兼容性。
6. 使用 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 验证；必要时运行现有诊断配置做 smoke check。
