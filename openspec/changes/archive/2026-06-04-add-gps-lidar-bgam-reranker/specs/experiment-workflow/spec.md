## ADDED Requirements

### Requirement: GPS+LiDAR BGAM 配置驱动工作流
项目 MUST 提供 `configs/deepsense6g_gps_lidar_bgam.yaml`，用于驱动 DeepSense6G GPS+LiDAR BGAM 的 manifest enrich、训练、评估、debug mask、ablation 和 comparison。配置 MUST 声明数据场景、GPS v2 artifact、Top8 manifest、LiDAR profile/cache、geometry、BGAM、model、loss、train、eval、ablation、metrics 和 outputs。

#### Scenario: 默认配置字段
- **WHEN** 开发者查看 `configs/deepsense6g_gps_lidar_bgam.yaml`
- **THEN** 配置 MUST 包含 scenario31-34、`mapping_disabled`、`num_beams=64`、`support_ratio=0.15`、`topk=8`、GPS v2 sweep root、Top8 manifest path、LiDAR BEV/grid defaults、BGAM default mode `single_soft` 和 output root
- **AND** 配置 MUST 包含 `anti_leakage.query_label_used_for_training=false`

#### Scenario: 命令行覆盖核心参数
- **WHEN** 用户通过 BGAM CLI 传入 `--support-ratio`、`--label-space`、`--topk`、`--bgam-mode` 或 `--output-dir`
- **THEN** 系统 MUST 使用命令行值覆盖配置默认值
- **AND** 输出目录 MUST 按 ratio tag、label space 和 run name 分离

#### Scenario: 运行产物保存配置快照
- **WHEN** BGAM workflow 完成一次运行
- **THEN** result dir MUST 保存 resolved config 或等价配置快照
- **AND** run metadata MUST 记录 GPS v2 artifact path、Top8 manifest path、LiDAR cache/profile、BGAM mode、beam angle source、support/query count 和 query label usage

### Requirement: GPS+LiDAR BGAM 验收命令
项目 MUST 记录并支持 GPS+LiDAR BGAM 的分层验收命令。所有项目相关 Python 命令 MUST 使用 `conda run -n kd_mm_beam` 环境运行。

#### Scenario: manifest enrich 验收命令
- **WHEN** 开发者运行 BGAM manifest enrich 验收
- **THEN** 推荐命令 MUST 为 `conda run -n kd_mm_beam kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`
- **AND** 命令 MUST 写出 BGAM manifest 和 metadata

#### Scenario: 训练评价验收命令
- **WHEN** 开发者运行 BGAM 训练评价验收
- **THEN** 推荐命令 MUST 为 `conda run -n kd_mm_beam kd-sensing-run-deepsense6g-gps-lidar-bgam --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`
- **AND** 命令 MUST 写出 metrics、summary、predictions、debug mask metadata 和 run metadata

#### Scenario: 独立评估验收命令
- **WHEN** 开发者运行已训练 checkpoint 的独立评估
- **THEN** 推荐命令 MUST 覆盖 `kd-sensing-evaluate-deepsense6g-gps-lidar-bgam`
- **AND** 命令 MUST 读取配置和 checkpoint
- **AND** 命令 MUST 写出 `metrics.json` 和 `predictions.csv`

#### Scenario: 测试验收命令
- **WHEN** 开发者运行 BGAM 单元测试
- **THEN** 推荐命令 MUST 覆盖 `tests/test_gps_lidar_bgam_geometry.py`、`tests/test_gps_lidar_bgam_model.py`、`tests/test_gps_lidar_bgam_dataset.py` 和 `tests/test_gps_lidar_bgam_runner.py`
- **AND** 最终回归仍 MUST 使用 `conda run -n kd_mm_beam pytest -q`

### Requirement: GPS+LiDAR BGAM README 工作流说明
README MUST 新增 GPS+LiDAR BGAM reranker 章节。该章节 MUST 保持 quickstart 风格，说明动机、输入 manifest、GPS/RSU coordinate assumption、beam-angle convention、BGAM modes、训练/评估命令、输出文件、debug mask 和结果判读。

#### Scenario: README 说明输入和假设
- **WHEN** 用户阅读 README 的 GPS+LiDAR BGAM 章节
- **THEN** 文档 MUST 说明需要 LiDAR path 或 BEV cache、GPS coordinate、RSU coordinate/yaw、GPS v2 logits/probs 或 Top8 manifest 和 64-beam label
- **AND** 文档 MUST 明确 future ground-truth beam 不用于 BGAM mask

#### Scenario: README 说明结果判读
- **WHEN** 用户阅读 README 的 BGAM 结果判读说明
- **THEN** 文档 MUST 指向 `metrics.json`、`summary_overall.csv`、`summary_by_scene.csv`、`summary_by_bgam_mode.csv`、`predictions.csv`、`debug_masks/` 和 comparison report
- **AND** 文档 MUST 说明如何比较 GPS-only、GPS+LiDAR no BGAM、soft/hard/topK BGAM 和 topK per-candidate rerank
