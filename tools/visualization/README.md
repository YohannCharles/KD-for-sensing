# Gradio 多模态可视化分析

这个目录提供新的交互式多模态时序样本浏览器，用于替代旧的静态 PNG 可视化报告。Viewer 可以直接读取数据集配置并生成可复用 cache，也可以读取已有 manifest；它不启动训练、不加载 checkpoint、不执行模型推理。

## 推荐运行流程

按下面顺序运行：

1. 安装 viewer 依赖。
2. 选择一个数据集/训练配置。
3. 用 Gradio viewer 的 `--config` 和 `--scenes` 启动程序。
4. 程序自动处理该配置下所选 split 的全部样本，写入可复用 cache。
5. 处理完成后在浏览器里按 scene、split、show mode 和 sample slider 交互查看样本。

完整命令：

```bash
conda run -n kd_mm_beam python -m pip install -r tools/visualization/requirements_viewer.txt

NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32 \
  --host 127.0.0.1 \
  --port 7860
```

打开：

```text
http://127.0.0.1:7860
```

如果只是想确认 viewer 能否读取 manifest，而不启动 Web 服务：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --manifest tools/visualization/sample_manifest_example.json \
  --check-only
```

需要定位切帧耗时时，可以加 `--profile-render`。该参数默认关闭；开启后不会新增页面组件，也不会调整顶部控件、Overview、Raw Modalities、Processed Modalities 或 Diagnostics 的布局，只在控制台输出每次回调的过滤耗时、静态渲染耗时、future distribution 渲染耗时、cache hit/miss 和实际更新组件数量：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --manifest tools/visualization/sample_manifest_example.json \
  --host 127.0.0.1 \
  --port 7860 \
  --profile-render
```

性能优化主要减少后端重复过滤、图片解码、Plotly/DataFrame 重建和隐藏 Tab 的同步更新。Gradio 前端在一次回调中渲染多张图片、多个 Plotly 图和表格时仍有固定成本；如果 profile 日志中 `callback_total_ms` 已较低，但浏览器仍感觉慢，瓶颈通常在前端组件渲染或浏览器绘制。

如果只是想确认 `--config` 能否完成数据处理和 cache 准备，而不启动 Web 服务：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32 \
  --check-only
```

## 安装依赖

推荐在项目环境中安装交互可视化依赖：

```bash
conda run -n kd_mm_beam python -m pip install -r tools/visualization/requirements_viewer.txt
```

如果使用 editable 安装，也可以安装可选 extra：

```bash
conda run -n kd_mm_beam python -m pip install -e ".[visualization]"
```

## 自动处理和 cache

推荐直接用 `--config --scenes 9,32` 启动 viewer。第一次运行会处理配置中所选 scene/split 的全部样本，生成：

- `samples.json`：Gradio 使用的样本 manifest
- `manifest_meta.json`：cache metadata，记录配置摘要、源文件 mtime/size 和样本数
- `viewer_assets/`：处理后的 image、radar、LiDAR、GPS、mmWave 可视化文件

再次运行时，如果配置、CSV、样本源文件和已生成资产都没有变化，程序会直接复用 cache，不重新处理数据。需要强制重处理时加 `--force-rebuild`：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32 \
  --force-rebuild \
  --check-only
```

临时调试时可以只处理少量样本：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32 \
  --sample-limit 20 \
  --check-only
```

## 一条命令：跑模型并启动 viewer

推荐直接用 `--run-models`。这条命令会按顺序完成：并行加载五个单模态 checkpoint、导出每个 beam label 的 confidence 曲线、生成 viewer manifest、启动 Gradio。

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 32 \
  --run-models \
  --model-workers 5 \
  --model-devices cuda \
  --model-batch-size 32 \
  --host 127.0.0.1 \
  --port 7860
```

`--model-devices` 默认就是 `cuda`，会使用所有可见 GPU；多张 GPU 时按模态轮转分配。CUDA 不可用时会直接报错，不会静默退回 CPU。单 GPU 机器上默认也可以并行跑多个小模型，如果显存紧张就把 `--model-workers` 调小到 `1` 或 `2`。只有明确想用 CPU 时才传 `--model-devices cpu`。

模型 checkpoint 按 scene 隔离，`--run-models` 一次只支持一个 scene；需要看两个场景的数据时，先用不带
`--run-models` 的 `--scenes 9,32` 生成统一 manifest。

快速检查可以先限制样本数并不启动服务：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 32 \
  --run-models \
  --sample-limit 5 \
  --check-only
```

如果某个模态不用默认 registry checkpoint，可以显式指定：

```bash
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 32 \
  --run-models \
  --model-checkpoint image=/path/to/image.pth \
  --model-checkpoint radar=/path/to/radar.pth
```

## 可选：只导出 manifest

如果想先离线处理数据、不启动 Gradio，可以单独运行导出脚本：

```bash
conda run -n kd_mm_beam python tools/visualization/export_viewer_manifest.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32
```

这个命令会读取配置中的 dataset、scene、split、启用模态、cache policy 和过滤字段，处理全部样本，并输出 Gradio viewer 可读取的 `samples.json`。多个 scene 会写入同一个 manifest，页面的 `Scene` 下拉框可直接切换。导出过程默认只写入 viewer cache 目录，不修改训练 checkpoint、训练日志、评估报告或 split CSV。

默认诊断配置启用 image、radar、gps、lidar、mmWave 五个模态，并使用 `model.modalities` 标记数据模态集合；导出 manifest 不会加载 `fusion_teacher` 或 `fusion_student` 模型。

旧入口 `scripts/visualize_modalities.py` 和 `kd-sensing-visualize-modalities` 现在也会导出 manifest，并提示改用 Gradio viewer，不再生成旧的静态 PNG 总览图作为主产物。

可选合并预测、质量分数和 gate 权重：

```bash
conda run -n kd_mm_beam python tools/visualization/export_viewer_manifest.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 32 \
  --run-models \
  --model-workers 5 \
  --quality outputs/eval/quality.json \
  --gate outputs/eval/gate.json
```

如果 `--predictions` 中每条记录同时包含 `prediction`、`confidence` 或 `confidence_curves` 字段，导出器会分别合并到 manifest 顶层，供 JSON、bar chart 和叠加曲线图展示。

## 启动 viewer

不需要模型曲线时，也可以只使用 `--config`，这样 viewer 只准备或复用数据 cache：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/diagnostics/gradio_viewer_cache \
  --scenes 9,32 \
  --host 127.0.0.1 \
  --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

`NO_PROXY` 这一段用于避免 Gradio 启动时的 localhost 自检被代理环境拦截。如果你的机器没有代理问题，也可以省略这些环境变量。

后台启动：

```bash
setsid bash -lc 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate kd_mm_beam && cd /root/projects/KD-for-sensing && env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= python tools/visualization/gradio_multimodal_viewer.py --config configs/diagnostics/modality_visualization.yaml --cache-dir outputs/diagnostics/gradio_viewer_cache --scenes 9,32 --host 127.0.0.1 --port 7860 >/tmp/kd_gradio_viewer.log 2>&1' >/dev/null 2>&1 < /dev/null &
```

查看端口和停止服务：

```bash
ss -ltn | grep ':7860'
ps -ef | grep gradio_multimodal_viewer.py
kill <pid>
```

## Manifest 格式

Viewer 支持 JSON 数组和 JSONL。每条样本推荐包含：

```json
{
  "sample_id": "000001",
  "scene_id": "scene32",
  "split": "test",
  "sequence_id": "seq_001",
  "time_index": 0,
  "raw": {
    "image": "data/raw/image/000001.jpg",
    "lidar": "data/raw/lidar/000001_bev.png",
    "radar": "data/raw/radar/000001_heatmap.png",
    "gps": "data/raw/gps/000001.json",
    "mmwave": "data/raw/mmwave/000001.json"
  },
  "processed": {
    "image": "data/processed/image/000001.png",
    "lidar": "data/processed/lidar/000001.png",
    "radar": "data/processed/radar/000001.png",
    "gps": "data/processed/gps/000001.json",
    "mmwave": "data/processed/mmwave/000001.json"
  },
  "label": {
    "current_beam": 12,
    "future_beams": [15, 16, 17]
  },
  "prediction": {
    "fusion_pred": 16,
    "topk": [16, 15, 17],
    "correct": true
  },
  "confidence": {
    "image": 0.81,
    "radar": 0.42,
    "fusion": 0.93
  },
  "confidence_curves": {
    "image": [0.01, 0.03, 0.12],
    "radar": [0.04, 0.11, 0.08],
    "gps": [0.02, 0.05, 0.2],
    "lidar": [0.03, 0.14, 0.09],
    "mmwave": [0.08, 0.19, 0.31]
  },
  "quality": {
    "image": 0.68,
    "radar": 0.41
  },
  "gate": {
    "image": 0.3,
    "radar": 0.1
  },
  "extra": {
    "loss": 1.24
  }
}
```

字段可以缺失。缺少模态、图片不存在、JSON 无法解析或过滤结果为空时，页面会显示空状态，不会中断浏览。

`confidence` 用于每个模态的单值 confidence bar chart；`confidence_curves` 或 `prediction.confidence_curves` 可提供每个 beam label 的曲线，viewer 会把所有模态曲线叠加，并用竖线标出当前样本的 `label.future_beams`。

`beam_distribution` 用于 Diagnostics Tab 中的 Future Beam Distribution Inspector。推荐格式：

```json
{
  "beam_distribution": {
    "image": {
      "prob": [[0.01, 0.02, 0.97], [0.04, 0.91, 0.05]],
      "logit": [[-1.0, -0.3, 3.4], [-0.2, 2.9, 0.1]]
    },
    "gps": {
      "prob": [[0.2, 0.7, 0.1], [0.1, 0.6, 0.3]]
    }
  }
}
```

`prob` 是 softmax 后概率，shape 为 `[H, num_beams]`；`logit` 是 softmax 前输出，可以缺失。Viewer 会从分布长度自动推断 `num_beams`，不会强制 64 类。没有完整分布但存在 `modality_prediction` 或 `prediction.modalities` 时，只生成 summary/detail，不会用 top1 confidence 伪造 heatmap。

## GPS 与 mmWave JSON

GPS 支持：

```json
{"x": [0.0, 0.2], "y": [0.0, 0.1]}
```

或：

```json
[{"x": 0.0, "y": 0.0}, {"x": 0.2, "y": 0.1}]
```

mmWave 支持：

```json
{"beam_power": [0.1, 0.2, 0.9]}
```

或：

```json
{"beam_power_seq": [[0.1, 0.2], [0.3, 0.4]]}
```

## 常见问题

- 页面显示 `No samples found`：检查 scene、split、show mode 过滤条件，或确认 manifest 非空。
- 图片不显示：检查路径是否存在。相对路径优先按 manifest 所在目录解析，然后按项目根目录解析。
- GPS/mmWave 为空图：检查 JSON 是否符合支持格式。
- 旧静态 PNG 文件不再生成：这是预期行为。推荐直接用 `--config` 启动 viewer，让程序自动处理数据并复用 cache。
