# KD for Sensing

本仓库提供基于 `src/kd_sensing` 包的多模态感知知识蒸馏实验工作流，主要覆盖 DeepSense6G、MMW 和 Raymobtime 数据集家族中的训练、评估、预处理、诊断和可视化入口。

## 安装

所有项目相关 Python 命令都使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam python -m pip install -e .
conda run -n kd_mm_beam python -c "import kd_sensing"
```

安装后可用 console scripts：

```bash
conda run -n kd_mm_beam kd-sensing-train --help
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help
conda run -n kd_mm_beam kd-sensing-visualize-modalities --help
```

等价包内 CLI 入口形如：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.train --help
conda run -n kd_mm_beam python -m kd_sensing.cli.export_viewer_manifest --help
```

## 目录概览

```text
configs/          # 训练、评估和预处理配置；高级 fusion 优先由 canonical/overlay recipe 生成
docs/             # 实验矩阵、数据集说明、扩展指南和性能调优说明
openspec/specs/   # 当前需求和架构契约
scripts/          # 保留的薄 alias、研究诊断和数据准备脚本
src/kd_sensing/   # 包内 CLI、config、data、engine、models、diagnostics 等实现
tests/            # 架构边界、配置加载、训练/诊断单元测试
tools/analysis/   # 研究分析脚本
tools/visualization/ # Gradio viewer 和 viewer 支持工具
```

配置相对路径从项目根目录解析，因此可以在子目录中启动命令。

## 快速健康检查

窄改动优先运行相关测试。涉及架构、导入边界、CLI 或公共 workflow 时，先跑：

```bash
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help
conda run -n kd_mm_beam pytest tests/test_phase_1_5_utility_validation.py -q
conda run -n kd_mm_beam pytest tests/test_complementarity_analysis.py tests/test_gradio_complementarity_explorer.py -q
```

最终回归：

```bash
conda run -n kd_mm_beam pytest -q
```

## 主要入口

训练：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/image/teacher_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_student_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml
```

评估：

```bash
conda run -n kd_mm_beam kd-sensing-evaluate \
  --config configs/image/teacher_no_kd.yaml \
  --weights outputs/scene31/image_teacher_no_kd/checkpoints/best.pth
```

预处理：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra_gps_lidar.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
```

Viewer manifest 导出：

```bash
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32
```

`kd-sensing-visualize-modalities` 保留为包内薄 alias，只委托 manifest 导出 CLI，不恢复旧静态 PNG 总览图流程。推荐命令仍是 `kd-sensing-export-viewer-manifest` 或 `python -m kd_sensing.cli.export_viewer_manifest`。

## 配置和实验矩阵

单模态 canonical 配置使用：

- `configs/<image|radar|gps|lidar|mmwave>/teacher_no_kd.yaml`
- `configs/<image|radar|gps|lidar|mmwave>/student_no_kd.yaml`
- `configs/<image|radar|gps|lidar|mmwave>/logits_kd.yaml`
- `configs/<image|radar|gps|lidar|mmwave>/rkd.yaml`

Fusion canonical 配置按固定模态顺序 `image -> radar -> gps -> lidar -> mmwave` 解析，命名为：

```text
configs/fusion/<canonical_slug>_<teacher_no_kd|student_no_kd|logits_kd|rkd>.yaml
```

很多 fusion 路径是 virtual config：磁盘上没有实体 YAML 时，配置加载器会按 canonical/overlay recipe 生成完整配置；实体 YAML 仍优先于生成规则。高级 G2D、CRAF 和 MARF 推荐使用 `configs/fusion/overlay_*.yaml` recipe 入口；已删除的同名 legacy 实体路径会作为 virtual alias 解析。训练产物中的 `final_config.yaml` 和 `resolved_config.yaml` 保存完整解析结果。

G2D、CRAF、MARF、CSI hardening、snapshot next-frame、objective-aware fusion、Raymobtime 和推荐实验顺序见 [docs/experiment_matrix.md](docs/experiment_matrix.md)。

## 数据和产物边界

- `dataset/` 是本地数据输入，默认不提交；源码中只保留 `dataset/.gitkeep`。
- `outputs/`、`logs/`、cache、TensorBoard 产物和新生成 checkpoint 是本地运行产物，默认不提交。
- `All_models/` 中已跟踪权重是历史复现实验资料；新生成的 `.pth`、`.pt`、`.ckpt` 不应进入源码变更。
- 当前运行时优先使用 checkpoint registry，或通过 `distillation.teacher_model_name` / `--weights` 显式传入 checkpoint。

DeepSense6G 默认场景是 Scenario 31，数据根目录解析为 `dataset/DeepSense6G/scenario31`，输出默认写入 `outputs/scene31/<run_name>/`。可通过配置或 CLI override 切换场景：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/mmwave/teacher_no_kd.yaml data.dataset.scene=9
conda run -n kd_mm_beam kd-sensing-train --config configs/mmwave/teacher_no_kd.yaml data.dataset.scene=32
```

Raymobtime s008 current snapshot beam selection 见 [docs/Raymobtime_s008_selection.md](docs/Raymobtime_s008_selection.md)。MMW Town10 skybridge 本地数据准备使用：

```bash
conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py \
  --config configs/preprocess/mmw_town10_skybridge.yaml
```

## Viewer

Gradio 交互式 viewer 入口保留在 `tools/visualization/gradio_multimodal_viewer.py`。安装可选依赖、启动 viewer、后台运行、manifest 格式、prediction/quality/gate 合并和故障排查见 [tools/visualization/README.md](tools/visualization/README.md)。

离线 manifest 导出推荐：

```bash
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 32 \
  --predictions outputs/eval/predictions.json \
  --quality outputs/eval/quality.json \
  --gate outputs/eval/gate.json
```

## 文档索引

- 实验矩阵和推荐运行顺序：[docs/experiment_matrix.md](docs/experiment_matrix.md)
- Raymobtime s008：[docs/Raymobtime_s008_selection.md](docs/Raymobtime_s008_selection.md)
- 训练吞吐、cache 和并行建议：[docs/training_throughput.md](docs/training_throughput.md)
- 新组件扩展指南：[docs/extension_guide.md](docs/extension_guide.md)
- Viewer 详细说明：[tools/visualization/README.md](tools/visualization/README.md)
- 项目表面积 inventory：[docs/project_surface_inventory.md](docs/project_surface_inventory.md)
- 架构与需求契约：`openspec/specs/`

## 破坏性变更

旧的顶层脚本入口已移除；请使用 console script、包内 CLI 或保留的 `scripts/` 薄 alias。

| 旧命令 | 当前入口 |
| --- | --- |
| `python train_image.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/image/<mode>.yaml ...` |
| `python train_both.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/<mode>.yaml ...` |
| `python test_model_image.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/image/<mode>.yaml --weights <path>` |
| `python test_model_both.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/fusion/<mode>.yaml --weights <path>` |
| `python CSV_process.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/radar_ra.yaml` |
| `python gen_data_seq.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra.yaml` |
