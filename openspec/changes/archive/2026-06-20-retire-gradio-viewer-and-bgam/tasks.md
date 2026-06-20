## 1. 规格与治理收口

- [x] 1.1 更新 current OpenSpec specs，归档本 change 的 BGAM、viewer manifest、TopK candidate supporting 和 lifecycle delta，确保 current specs 不再要求 BGAM/viewer 入口存在。
- [x] 1.2 更新 `docs/project_surface_inventory.md`，将 BGAM、GPS pseudo-history BGAM、BGAM-only TopK candidate 支撑、viewer manifest 和 Gradio viewer 从 current/supporting 支持面移除或标记为 retired-tombstone。
- [x] 1.3 更新 `docs/maintainer_context_index.yaml`，删除 BGAM/viewer entrypoint owner metadata、hotspot、validation command 和 routing 中的 current 要求，并新增退役防回流 guard。
- [x] 1.4 更新 README、主线模型目录、实验协议表、结果账本和实验矩阵中 BGAM/viewer 当前命令、状态和输出说明，保留历史说明时明确标记 retired/historical。

## 2. 入口、配置和依赖删除

- [x] 2.1 从 `pyproject.toml` 删除 `kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities` 和所有 DeepSense6G/MMW BGAM prepare/run/evaluate console scripts。
- [x] 2.2 删除 `configs/deepsense6g_gps_lidar_bgam.yaml`、`configs/mmw_town_gps_lidar_bgam.yaml` 和其它只服务 BGAM/viewer manifest 的实体配置或 config load allowlist。
- [x] 2.3 检查 dependency metadata 与文档，移除仅服务仓库级 Gradio viewer 的 Gradio 依赖或安装说明；不得影响 JEPA visual analysis、plotting 或保留诊断依赖。

## 3. BGAM 源码清理

- [x] 3.1 删除 `src/kd_sensing/cli/*bgam*`、BGAM data manifest/dataset、BGAM engine、BGAM model、BGAM loss 和 BGAM debug mask 相关专属模块。
- [x] 3.2 清理 registry、package `__init__`、engine/model/loss/data imports 和 generated metadata 引用，确保当前主线不导入 BGAM 专属模块。
- [x] 3.3 按语义保留通用 Top-K metrics、circular metrics、GPS v2、LiDAR preprocessing 和其它非 BGAM current helper；删除只服务 BGAM candidate manifest/dataset/loss 的代码。

## 4. Viewer Manifest 和 Gradio Viewer 清理

- [x] 4.1 删除 `src/kd_sensing/cli/export_viewer_manifest.py`、`kd-sensing-visualize-modalities` alias 和 `src/kd_sensing/diagnostics/viewer_manifest*`、`viewer_predictions.py` 等 viewer manifest 专属模块。
- [x] 4.2 检查 JEPA visual analysis、GPS shortcut benchmark 和其它保留诊断是否复用 viewer helper；如有通用 JSON/asset/statistics helper，先迁入非 viewer 命名模块并补最小测试。
- [x] 4.3 清理 README、docs 和架构测试中对仓库级 Gradio viewer、viewer manifest 导出和 viewer cache 目录的 current 推荐描述。

## 5. 测试和防回流

- [x] 5.1 删除或改写 BGAM focused tests：`tests/test_gps_lidar_bgam_geometry.py`、`tests/test_gps_lidar_bgam_model.py`、`tests/test_gps_lidar_bgam_dataset.py`、`tests/test_gps_lidar_bgam_runner.py`。
- [x] 5.2 更新 `tests/test_cli_help.py`，移除 BGAM/viewer manifest help smoke，并保留当前 CLI 的 help 覆盖。
- [x] 5.3 更新 `tests/test_architecture_boundaries.py`，将 BGAM/viewer manifest 从 allowlist/hotspot/current route 中移除，并增加退役入口、配置和模块不得回流的断言。
- [x] 5.4 更新配置加载、诊断和文档同步测试，使 `configs/*bgam*.yaml`、viewer manifest CLI 和旧 alias 不再被要求可用。

## 6. 验证

- [x] 6.1 运行 `openspec validate retire-gradio-viewer-and-bgam --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。（已运行；本 change 相关检查通过，当前失败来自工作区已有未跟踪 `知乎问答下载.md` 和未登记未跟踪 specs：`jepa-visual-architecture-sweep`、`real-perturbation-forward-evaluation`、`safe-residual-beam-rerank-fusion`。）
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 6.4 运行仍保留诊断的 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py tests/test_jepa_msac.py -q`。
- [x] 6.5 按实现风险运行 `conda run -n kd_mm_beam pytest -q`，并确认没有新增 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 generated metadata 源码变更。（已运行；894 passed，剩余 3 个失败与 6.2 相同，来自工作区已有未跟踪 `知乎问答下载.md` 和未登记未跟踪 specs。）
