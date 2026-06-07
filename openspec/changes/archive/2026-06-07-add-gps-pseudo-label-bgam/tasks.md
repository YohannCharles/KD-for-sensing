## 1. MMW Label-space 与 GPS v2 Artifact

- [x] 1.1 将本 change 的主实验对象改为 MMW Town10 sunny scenes，并保留 DeepSense6G 代码为非默认后续迁移路径
- [x] 1.2 为 MMW Top8/BGAM 配置新增 `mapping_enabled` 默认配置和 `mapping_disabled` raw-label 对照
- [x] 1.3 扩展 MMW GPS v2 logits index、predictions、support manifest 和 run metadata，记录 scene-specific `label_space`、`beam_label_space` 和 mapping fingerprint
- [x] 1.4 在 config validation 中支持 MMW scene-specific fingerprint 校验

## 2. MMW GPS Top8 Candidate Manifest

- [x] 2.1 新增 `configs/mmw_town_top8_selector.yaml`
- [x] 2.2 新增 MMW Top8 manifest builder，从 frozen MMW GPS v2 logits/probs 重新计算 TopK candidates
- [x] 2.3 Top8 manifest 合并 MMW prepared CSV、frame/sample id、LiDAR path、beam power path、GPS logits row index 和 label-space metadata
- [x] 2.4 支持按 scene 校验 logits artifact、predictions、support manifest 和 Top8 manifest 的 mapping fingerprint
- [x] 2.5 输出 Top8 recall summary、candidate rank distribution 和 normalized gain 诊断字段
- [x] 2.6 新增 `kd-sensing-prepare-mmw-town-top8-candidate-manifest` CLI

## 3. MMW GPS Pseudo-history Builder

- [x] 3.1 扩展 pseudo-history builder，支持配置 `history.group_keys`
- [x] 3.2 MMW 默认按 `scene + agent + split` 分组，避免不同车辆/轨迹串历史
- [x] 3.3 输出 `history_pseudo_beams`、prob、entropy、confidence、valid mask、timestamps、source row indices 和 missing count metadata
- [x] 3.4 写出 `pseudo_history_summary.csv`，按 scene/history step/confidence bucket 统计 coverage、entropy 和 evaluation-only pseudo error
- [x] 3.5 保持 anti-leakage：target/query true beam label 不参与 pseudo-history 生成

## 4. MMW BGAM Manifest、Dataset 与 LiDAR

- [x] 4.1 新增 `configs/mmw_town_gps_lidar_bgam.yaml`
- [x] 4.2 新增 MMW BGAM manifest enrich，合并 MMW Top8、RSU LiDAR、BEV cache、pseudo-history 和 mapping metadata
- [x] 4.3 MMW manifest 默认优先使用 frame manifest 中 RSU/BS-side LiDAR path，并记录 `lidar_source`
- [x] 4.4 复用 `GPSLidarBGAMDataset` 解析 history tensor、candidate tensor、LiDAR BEV/raw point cloud 和 label-space metadata
- [x] 4.5 对 pseudo-history 缺失、LiDAR 缺失和 mapping fingerprint 冲突写清晰错误或 skipped reason
- [x] 4.6 新增 `kd-sensing-prepare-mmw-town-gps-lidar-bgam-manifest` CLI

## 5. BGAM Model、Runner 与 Evaluation

- [x] 5.1 复用并扩展 `GPSGuidedBGAM` 的 `history_pseudo_soft`、`history_pseudo_hard`、`history_pseudo_topk_union` 和 per-candidate history mask 模式
- [x] 5.2 扩展 `GPSLidarBGAMBeamPredictor` forward，消费 history pseudo label/prob/entropy/valid mask，并禁止 future label 进入 BGAM mask
- [x] 5.3 扩展 BGAM runner，支持 MMW 无 support-ratio 子目录输出结构
- [x] 5.4 新增 `kd-sensing-run-mmw-town-gps-lidar-bgam` 和 `kd-sensing-evaluate-mmw-town-gps-lidar-bgam` CLI
- [x] 5.5 predictions/summary 写出 pseudo-history、mapping fingerprint、BGAM mode、history source、oracle upper bound 标记和 delta vs GPS
- [x] 5.6 MMW predictions/summary 透传 `gps_normalized_gain`、`final_normalized_gain` 和 delta normalized gain
- [x] 5.7 确保 checkpoint selection 不使用 target query label，也不把 oracle-history upper bound 纳入默认 best_ablation

## 6. 文档、OpenSpec 与测试

- [x] 6.1 更新 OpenSpec proposal/design/specs，使主路径从 DeepSense6G 改为 MMW
- [x] 6.2 更新 README/运行说明，明确 MMW-first workflow、GPS logits 前置步骤和与 arXiv:2603.15093v1 对比边界
- [x] 6.3 增加 MMW tiny fixture 测试，覆盖 GPS logits → MMW Top8 manifest → MMW BGAM manifest/pseudo-history
- [x] 6.4 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q` 验证 MMW GPS/Top8/BGAM manifest 行为
- [x] 6.5 使用 `conda run -n kd_mm_beam pytest tests/test_topk_candidate_manifest.py tests/test_gps_lidar_bgam_dataset.py tests/test_gps_lidar_bgam_model.py tests/test_gps_lidar_bgam_runner.py -q` 验证通用 BGAM/DeepSense 回归
- [x] 6.6 使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 验证配置加载
- [x] 6.7 使用 `openspec validate add-gps-pseudo-label-bgam --strict` 校验 OpenSpec change

## 7. MMW 端到端实验

- [x] 7.1 重跑 MMW GPS v2 canonical `mapping_enabled` 输出，启用 `--save-logits --save-prior-probs`
- [x] 7.2 先跑 MMW 小样本 smoke：Top8 manifest、pseudo-history builder、BGAM manifest enrich、`gps_pseudo_history_soft_bgam`
- [x] 7.3 再跑 MMW Town10 `mapping_enabled` 主实验，输出 `summary_overall.csv`、`summary_by_scene.csv`、`predictions.csv`、`pseudo_history_summary.csv` 和 `run_metadata.json`
- [x] 7.4 单独跑 MMW `mapping_disabled` raw-label 对照，比较 mapped vs raw 的 pseudo-history continuity、BGAM DBA、mean circular error 和 normalized gain
- [x] 7.5 更新 comparison report，明确与 arXiv:2603.15093v1 的可比指标、失败场景、pseudo label 误差来源和后续调参建议
