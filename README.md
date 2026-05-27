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
conda run -n kd_mm_beam kd-sensing-runs --help
conda run -n kd_mm_beam kd-sensing-hist-beam-loso --help
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
docs/             # 实验矩阵、数据集说明、研究笔记、扩展指南和性能调优说明
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
conda run -n kd_mm_beam kd-sensing-visualize-modalities --help
conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py tests/test_modality_visual_diagnostics.py -q
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

快速调试训练时，可以启用 train epoch 子采样，只减少每个 epoch 的训练 step，不改 train CSV、不缩小 validation/test split，也不替代 `data.dataset.portion`：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_student_no_kd.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.fraction=0.1 \
  -o output.progress.enabled=false
```

也可以用固定样本数限制每个 epoch：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/student_no_kd.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.num_samples=256
```

`fraction` 和 `num_samples` 二选一；`seed` 为空时默认使用 `experiment.seed`。默认 `rotate_each_epoch=true`，会按绝对 epoch 轮换无放回抽样，resume 后同一 epoch 的样本选择仍可复现；设置 `rotate_each_epoch=false` 可固定同一小子集用于排障。运行产物会在 `train_log.json`、`final_config.yaml` 的 runtime metadata 中记录完整 train 样本数、每 epoch 有效样本数、seed、轮换设置和是否退化为完整 epoch。更完整的吞吐和 cache 说明见 [docs/training_throughput.md](docs/training_throughput.md)。

评估：

```bash
conda run -n kd_mm_beam kd-sensing-evaluate \
  --config configs/image/teacher_no_kd.yaml \
  --weights outputs/scene31/image_teacher_no_kd/checkpoints/best.pth
```

HiST-Beam 跨场景 LOSO 资源探针：

```bash
conda run -n kd_mm_beam kd-sensing-hist-beam-loso \
  --config configs/hist_beam/quick_smoke.yaml
```

完整 quick validation 方法矩阵需显式使用 `quick_validation` 配置；可先用 `--max-runs` 分段执行：

```bash
conda run -n kd_mm_beam kd-sensing-hist-beam-loso \
  --config configs/hist_beam/quick_validation.yaml \
  --variants v3_decoupled \
  --budgets 0,10 \
  --seeds 0 \
  --max-runs 2
```

默认 `quick_smoke` 只运行轻量 resource probe；`quick_validation` 使用 DeepSense6G scenarios 31-34、`image`/`radar`/`gps` 三模态和包内 `hist_beam_fusion` 模型。详细变体矩阵、target adapt/test 防泄漏和 prototype/adaptation 设计以 OpenSpec change `add-hist-beam-cross-scene-adaptation` 为准。

实验运行索引：

```bash
conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs
conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs --format json \
  --state running --state killed --output outputs/analysis/run_index.json
```

`kd-sensing-runs` 只读扫描本地 `outputs/`、`logs/`、当前 Python 进程和可用资源快照，不删除、不移动、不重写训练产物、日志、checkpoint、cache 或 TensorBoard 文件。状态分类包括 `running`、`complete`、`started_no_metrics`、`partial`、`failed`、`killed`、`waiting`、`stale` 和 `unknown`；JSON 输出稳定包含 `generated_at`、`roots`、`runs`、`resources` 和 `warnings`。

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

Multimodal-NF 作为独立数据集家族默认放在 `dataset/MultimodalNF/`，通常目录为：

```text
dataset/MultimodalNF/
  raw/        # 用户放置官方 HDF5、image/lidar zip 或解压后的本地数据
  codebooks/  # 用户放置 upa64x64_NF_codebook*.pkl 或等价 codebook metadata
  cache/      # 审计/index/cache 生成产物，默认不提交
```

审计和 index 构建使用包内预处理入口，不会自动移动、复制、删除或解压真实数据：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_audit.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_index.yaml
```

Multimodal-NF 配置入口按用途区分：

- dataset smoke：`configs/multimodal_nf/dataset_smoke.yaml`，用于小步验证 HDF5/index/profile/target 契约。
- 单任务 near-field beam：`*_beam.yaml`、`gps_only.yaml`、`csi_only.yaml`、`image_lidar.yaml`，objective 为 `near_field_beam_selection`，target 是三维 codebook flattened beam class。
- 单任务 LOS：`*_los.yaml`，objective 为 `current_los_classification`，主 target 是 `los_label`，不是 beam-only run。
- multitask/fusion：`fusion_all_tasks.yaml` 同时启用 beam、LOS 和 link quality；`fusion_beam.yaml`、`fusion_los.yaml`、`fusion_near_field.yaml` 是 fusion 入口。

真实训练前请确认 `data.dataset.codebook_path`、`codebook_shape` 或 `codebook_profile` 与本地 codebook 一致，且模型 beam head `num_classes` 与 codebook `num_beam_classes` 一致；HDF5、zip、codebook、cache、审计报告、训练日志和 checkpoint 都属于本地输入或运行产物，通常不进入源码变更。

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
- 研究结论和历史方案收束：[docs/research_notes.md](docs/research_notes.md)
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
