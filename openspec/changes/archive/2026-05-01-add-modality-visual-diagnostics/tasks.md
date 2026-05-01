## 1. 诊断配置与入口

- [x] 1.1 新增 `scripts/visualize_modalities.py`，提供 `--config/-c`、`--override/-o` 和 dotted override 兼容入口，并使用 `conda run -n kd_mm_beam python scripts/visualize_modalities.py ...` 作为文档命令。
- [x] 1.2 新增 `src/kd_sensing/cli/visualize_modalities.py` 或等价 CLI 模块，复用 `load_cli_config()` 和现有配置覆盖逻辑。
- [x] 1.3 定义 `diagnostics.visualization` 配置解析 helper，支持 `output_dir`、`splits`、`sample_count`、`seed`、`seq_index`、`labels`、`modalities`、`max_frames_per_sample` 和 `include_raw_image_preview`。
- [x] 1.4 新增默认诊断配置 `configs/diagnostics/modality_visualization.yaml` 或等价示例，默认使用 Scene 32 统一 split 和小样本输出。

## 2. 数据源与可复现抽样

- [x] 2.1 新增 `src/kd_sensing/diagnostics/modality_visualization.py`，实现诊断运行主函数和只读输出目录准备。
- [x] 2.2 复用 `build_dataset(cfg, split)` 构建 train/test dataset，确保启用模态、scene、CSV、cache policy 和 Dataset 预处理路径与训练一致。
- [x] 2.3 读取 `dataset.root_csv` 并按 Dataset 样本顺序对齐 CSV 行，提取 `seq_index`、camera/radar/lidar/mmwave/GPS 路径列、历史 beam 和 future beam 信息。
- [x] 2.4 实现按 split、`seq_index`、future beam label、`sample_count` 和 `seed` 的稳定抽样；当候选不足时输出全部候选并记录 requested/actual count。
- [x] 2.5 处理 train/test 归一化状态复用，确保启用 GPS、LiDAR 或 mmWave 归一化时 test split 能使用 train-fitted scaler/normalizer 或给出清晰错误。

## 3. 可视化与统计

- [x] 3.1 实现通用张量统计 helper，记录 shape、dtype、min、max、mean、std 和 nonzero fraction。
- [x] 3.2 实现 image motion mask 面板，展示 Dataset 返回的 processed `image` 张量，并可选读取 raw RGB thumbnail 作为 reference-only 区域。
- [x] 3.3 实现 radar 面板，分别展示 `radar_ra` 和 `radar_da` 的时间序列 heatmap 或最后若干时隙 heatmap。
- [x] 3.4 实现 LiDAR 面板，展示 BEV 三通道合成图和通道拆分图，并记录每通道非零率。
- [x] 3.5 实现 GPS 面板，展示 relative-polar 特征序列曲线或轨迹摘要。
- [x] 3.6 实现 mmWave 面板，展示 time x beam-index receive-power heatmap，并记录 per-time 均值和方差摘要。
- [x] 3.7 实现 label 面板，展示 `input_beam`、`target_beam`、future label 和对应 CSV 路径摘要。

## 4. 输出产物

- [x] 4.1 为每个样本生成 PNG 总览图，路径包含 scene、split 和 dataset index，避免不同场景或 split 产物混淆。
- [x] 4.2 写出 `samples.csv` 或 `samples.jsonl`，包含 split、dataset index、`seq_index`、label、原始路径、PNG 路径和启用模态统计。
- [x] 4.3 写出 `summary.json`，记录最终诊断配置、scene、CSV 路径、split metadata 摘要、抽样条件、seed、请求样本数、实际样本数和输出文件列表。
- [x] 4.4 写出最终配置快照，便于复现同一次诊断运行。
- [x] 4.5 使用 matplotlib 非交互式 backend，确保无显示环境下也能生成 PNG。

## 5. 测试与验证

- [x] 5.1 添加单元测试，验证相同 seed 和过滤条件选择相同 dataset index，且候选不足时不会报错。
- [x] 5.2 添加单元测试，验证 image/radar/LiDAR/GPS/mmWave 统计 helper 对典型张量输出稳定字段。
- [x] 5.3 添加端到端 smoke test，使用 synthetic 或临时小型 DeepSense6G 风格数据运行诊断入口并检查 PNG、`summary.json` 和样本列表存在。
- [x] 5.4 添加测试，验证未启用模态不会触发对应文件读取或 cache 访问。
- [x] 5.5 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_preprocessing_formats.py` 或新增诊断测试文件验证行为。
- [x] 5.6 使用 `conda run -n kd_mm_beam python scripts/visualize_modalities.py --config configs/diagnostics/modality_visualization.yaml diagnostics.visualization.sample_count=2` 运行 Scene 32 小样本诊断 smoke test。

## 6. 文档与 OpenSpec

- [x] 6.1 更新 README 或实验说明，记录如何生成 Scene 9/32、train/test 的各模态处理后可视化。
- [x] 6.2 文档中明确 image 面板展示的是 processed motion mask，raw RGB 仅是 reference，不是模型输入。
- [x] 6.3 文档中说明诊断入口默认只读，不会修改训练 checkpoint、训练日志、split CSV 或评估报告。
- [x] 6.4 运行 `openspec validate add-modality-visual-diagnostics --strict`，确认 proposal、design、spec 和 tasks 合法。
