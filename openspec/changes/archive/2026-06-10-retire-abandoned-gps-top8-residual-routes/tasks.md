## 1. 公开入口和配置删除

- [x] 1.1 从 `pyproject.toml` 删除 GPS coarse anchor、MMW standalone Top8、DeepSense6G residual、camera residual 和 DeepSense6G Top8 selector 相关 console scripts；保留 DeepSense6G/MMW BGAM console scripts。
- [x] 1.2 删除退役配置：`configs/deepsense6g_residual_fusion.yaml`、`configs/deepsense6g_camera_residual.yaml`、`configs/deepsense6g_top8_selector.yaml`、`configs/mmw_town_top8_selector.yaml` 和 `configs/gps/*coarse*.yaml`；保留 `configs/*bgam*.yaml`。
- [x] 1.3 删除或更新引用退役配置/命令的 orchestration 脚本、analysis 脚本和 shell 脚本；不得新增兼容 alias 或 stub CLI。
- [x] 1.4 刷新入口清单，确认 `kd-sensing-gps-coarse-anchor`、retired standalone `*top8*` 和 `*residual*` 命令不再作为当前安装入口出现，并确认 BGAM 入口仍作为当前安装入口保留。

## 2. 源码实现删除

- [x] 2.1 删除 `src/kd_sensing/cli/` 中退役路线专属 CLI：Top8 selector run/plot/compare、residual、camera residual 和 GPS coarse anchor 入口；保留 BGAM prepare/run/evaluate 入口。
- [x] 2.2 删除 `src/kd_sensing/engine/` 中退役路线专属 engine：`deepsense6g_residual_fusion.py`、`deepsense6g_camera_residual.py`、`deepsense6g_top8_selector.py` 和 `gps_coarse_anchor.py`；保留 BGAM engines。
- [x] 2.3 删除 `src/kd_sensing/data/` 中退役 manifest/dataset/target provider：DeepSenseG residual、camera residual 和 geometry residual 相关模块；保留 BGAM dataset/manifest 及其 TopK candidate manifest 支撑模块。
- [x] 2.4 删除 `src/kd_sensing/models/` 中退役模型：TopK selector、candidate attention selector、beam candidate attention、GPS anchored residual fusion 和 camera residual fusion；保留 GPSGuidedBGAM 和 BGAM predictor。
- [x] 2.5 删除 `src/kd_sensing/losses/` 中退役 loss：residual 和 camera residual loss；保留 BGAM loss 及其 TopK candidate loss 支撑代码。
- [x] 2.6 清理 `__init__.py`、registry/default component imports、architecture allowlist 和其它内部引用，确保保留主线不依赖已删除模块。
- [x] 2.7 保留通用 Top-K metric、circular metric、viewer top-k 展示、CSI candidate ranking、GPS-Rel-Polar、GPS v2/MMW GPS v2、BGAM 和 JEPA GPS conditioning 代码。

## 3. 测试和架构边界更新

- [x] 3.1 删除退役路线 focused tests：`test_top8_selector_runner.py`、TopK selector model 专属测试、`test_candidate_attention_selector.py`、`test_topk_reranker.py`、`test_residual_*`、`test_camera_residual_*`、`test_gps_coarse_anchor.py`、`test_beam_candidate_attention.py` 和 `test_target_shot_geometry_residual.py`；保留 `test_gps_lidar_bgam_*` 和 BGAM 依赖的 candidate manifest/loss 回归。
- [x] 3.2 更新 `tests/test_architecture_boundaries.py`，把 residual/Top8 包内入口正向导入断言改为退役入口不存在或不在当前 inventory 中，并断言 BGAM 包内入口保留。
- [x] 3.3 更新 CLI help、student config、component registry、run index 或其它受删除影响的测试，不为退役入口保留正向 smoke，并保留 BGAM smoke。
- [x] 3.4 增加或保留核心回归，确保普通 Top-K evaluation、GPS v2、BGAM、CSI、Raymobtime、viewer manifest 和 JEPA 相关测试仍覆盖当前主线。

## 4. 文档和 OpenSpec 同步

- [x] 4.1 更新 README 和 README_REPRODUCE，移除 retired Top8 selector、residual、camera residual 和 GPS coarse anchor 的 quickstart、命令块、章节或当前推荐描述；保留 BGAM 说明。
- [x] 4.2 更新 `docs/project_surface_inventory.md`、`docs/experiment_matrix.md`、`docs/research_notes.md` 和 residual/geometry 相关文档，把退役路线标记为历史或删除当前入口说明，并确保 BGAM 仍列为当前入口。
- [x] 4.3 更新 OpenSpec 当前 specs 或归档说明，确保退役路线不再以正向 MUST workflow 留在当前规范中。
- [x] 4.4 保留本地产物边界说明：本 change 不自动删除 `outputs/`、`logs/`、cache、checkpoint、dataset 或历史权重。

## 5. 引用扫描和验证

- [x] 5.1 运行引用扫描，确认 README、docs、OpenSpec 当前 specs、configs、scripts、tests、pyproject 和 `src/kd_sensing` 不再把退役路线声明为当前入口，同时确认 BGAM 仍保留。
- [x] 5.2 运行 `openspec validate retire-abandoned-gps-top8-residual-routes --strict`。
- [x] 5.3 运行 `openspec status --change retire-abandoned-gps-top8-residual-routes` 并确认 artifacts complete。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.5 运行保留核心入口 smoke：`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`、`conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`、`conda run -n kd_mm_beam kd-sensing-train --help` 和 `conda run -n kd_mm_beam kd-sensing-evaluate --help`。
- [x] 5.6 运行必要的 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py tests/test_circular_metrics.py tests/test_run_index.py -q`，并根据实际受影响文件补充 GPS v2、CSI、Raymobtime 或 JEPA smoke。
