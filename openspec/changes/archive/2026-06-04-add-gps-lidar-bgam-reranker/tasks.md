## 1. 配置与包内入口

- [x] 1.1 新增 `configs/deepsense6g_gps_lidar_bgam.yaml`，覆盖 data、candidate/topk、geometry、lidar、bgam、gps、model、loss、train、eval、ablation、metrics、anti_leakage 和 outputs 默认字段。
- [x] 1.2 在 `pyproject.toml` 增加 BGAM console scripts：`kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest`、`kd-sensing-run-deepsense6g-gps-lidar-bgam` 和 `kd-sensing-evaluate-deepsense6g-gps-lidar-bgam`。
- [x] 1.3 新增包内 CLI 文件：`src/kd_sensing/cli/prepare_deepsense6g_gps_lidar_bgam_manifest.py`、`run_deepsense6g_gps_lidar_bgam.py` 和 `evaluate_deepsense6g_gps_lidar_bgam.py`。
- [x] 1.4 扩展架构边界测试，确认不新增 `train_gps_lidar_bgam.py`、`eval_gps_lidar_bgam.py`、顶层 `datasets.*`、顶层 `models.*` 或 `src.run_*` 入口。
- [x] 1.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证入口边界。

## 2. Geometry 与 Beam Angle 工具

- [x] 2.1 新增 `src/kd_sensing/utils/geometry.py`，实现 `wrap_to_pi`、`gps_to_local_xy`、`gps_to_rsu_aod` 和 distance 计算，支持 local x/y 与 lat/lon。
- [x] 2.2 新增 beam angle table helper，优先读取配置表，缺失时使用 DFT-ULA approximation，并返回 `beam_angle_source` metadata。
- [x] 2.3 为 RSU yaw 单位、zero axis、coordinate frame 和 fallback 行为加入配置校验与清晰错误。
- [x] 2.4 新增 `tests/test_gps_lidar_bgam_geometry.py`，覆盖 wrap-around、local 坐标、lat/lon 近似、yaw subtraction 和 beam angle fallback。
- [x] 2.5 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_geometry.py -q`。

## 3. BGAM Manifest 与 Dataset

- [x] 3.1 新增 `src/kd_sensing/data/deepsense6g_gps_lidar_bgam_manifest.py`，复用 Top8 manifest，补齐 `theta_gps`、`distance_to_rsu`、LiDAR path/cache path、RSU pose、coordinate frame 和 beam angle convention。
- [x] 3.2 支持配置化 manifest column mapping，兼容 Top8 candidate manifest、DeepSense6G sequence CSV 和用户提供的 GPS+LiDAR manifest 字段。
- [x] 3.3 在 manifest enrich metadata 中记录输入 manifest、GPS v2 artifact、Top8 candidate source、LiDAR availability、RSU pose source、缺失字段和受影响样本数。
- [x] 3.4 新增 `src/kd_sensing/data/deepsense6g_gps_lidar_bgam_dataset.py`，返回 candidate beams/probs/log_probs、theta_gps、distance、LiDAR BEV 或 raw points、label 和 support/query metadata。
- [x] 3.5 实现 variable-size raw point cloud collate_fn，确保空点云和不同点数 batch 可安全迭代。
- [x] 3.6 实现 normalizer fit boundary，确保 source/support rows 可用于 fit，target query rows 只 transform，并写入 `query_label_used_for_training=false` metadata。
- [x] 3.7 新增 `tests/test_gps_lidar_bgam_dataset.py`，覆盖 manifest enrich、字段映射、LiDAR 懒加载、gps_only 不读 LiDAR、query 不参与 normalizer fit 和缺失 LiDAR skipped reason。
- [x] 3.8 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_dataset.py -q`。

## 4. LiDAR BEV/Pillar Spatial Encoder

- [x] 4.1 扩展或新增 LiDAR BEV grid metadata 读写，记录 ROI、BEV size、cell center convention、FoV、filter 参数、cache version 和参数 hash。
- [x] 4.2 新增 `src/kd_sensing/models/lidar_pillar_encoder.py` 或等价包内模块，实现 `LidarBEVSpatialEncoder`，消费 `[B,C,H,W]` BEV 并输出 `[B,C',H',W']` spatial feature。
- [x] 4.3 实现 `SimplePillarEncoder` fallback，从 raw point cloud 生成 point count、mean z、max z、mean intensity、mean x/y offset 的 fixed-size pseudo-image。
- [x] 4.4 支持 `freeze_lidar_encoder`，并确保 optimizer 不更新 frozen encoder 参数。
- [x] 4.5 记录 LiDAR 输入质量摘要，包括 raw/model input 非空率、通道均值/标准差、零值比例、ROI、BEV size 和 cache path。

## 5. BGAM 与 Cross Attention 模型

- [x] 5.1 新增 `src/kd_sensing/models/gps_lidar_bgam.py`，实现 `GPSGuidedBGAM`，预计算 `theta_cell` buffer，并支持 `single_soft`、`single_hard`、`topk_union_soft` 和 `topk_per_candidate`。
- [x] 5.2 实现 GPS uncertainty/adaptive sigma 逻辑，支持 GPS entropy 或外部 uncertainty 扩大 soft mask sigma。
- [x] 5.3 实现 debug mask 保存 helper，只保存抽样样本，并记录 sample id、theta_gps、sigma/half_width、BGAM mode 和 beam angle source。
- [x] 5.4 实现 `LidarBEVCrossAttention`，将 `[B,C,H,W]` tokens flatten 后由 learnable query 聚合为 `[B,d_model]` LiDAR embedding。
- [x] 5.5 实现 `GPSPriorEncoder`，编码 sin/cos theta、log distance、可选 GPS logits/probs、entropy 和 top1 margin。
- [x] 5.6 新增 `tests/test_gps_lidar_bgam_model.py`，覆盖 mask shape、soft mask peak、hard mask binary、TopK union、per-candidate shape、cross-attention shape、GPS encoder shape 和 debug mask metadata。
- [x] 5.7 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_model.py -q`。

## 6. Fusion/Rerank Predictor 与 Loss

- [x] 6.1 新增 `src/kd_sensing/models/gps_lidar_bgam_model.py`，实现 `GPSLidarBGAMBeamPredictor`，组合 GPS prior encoder、LiDAR spatial encoder、BGAM、LiDAR cross-attention、cross-modal attention 或 concat MLP。
- [x] 6.2 默认实现 TopK rerank head，输出 `final_candidate_scores [B,K]`、`candidate_probs [B,K]`、`selected_beam [B]`，并使用 `candidate_log_prob + lambda_lidar * score_i`。
- [x] 6.3 实现可选 full 64 head 与 sparse 64 logits helper，仅作为 metrics 兼容或 ablation，不替代主 TopK rerank 报告。
- [x] 6.4 新增 `src/kd_sensing/losses/gps_lidar_bgam_losses.py`，实现 candidate CE、candidate circular soft CE、TopK miss skip/nearest soft loss、optional full 64 CE 和 prior anchor loss。
- [x] 6.5 加入 anti-leakage assertions，确保 `gt_beam` 不传入 BGAM mask source，改变 label 不改变 mask。
- [x] 6.6 扩展 `tests/test_gps_lidar_bgam_model.py` 或新增 loss 测试，覆盖 forward/backward、TopK hit/miss loss、circular smoothing 和 no-future-label leakage。
- [x] 6.7 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_model.py -q`。

## 7. Runner、Evaluation 与 Ablation

- [x] 7.1 新增 `src/kd_sensing/engine/deepsense6g_gps_lidar_bgam.py`，实现 manifest 自动 enrich/读取、seed/device、DataLoader、AMP optional、AdamW、scheduler、checkpoint 和 train/eval loop。
- [x] 7.2 实现默认 ablation：`gps_only`、`lidar_only_no_bgam`、`gps_lidar_no_bgam`、`gps_lidar_soft_bgam`、`gps_lidar_hard_bgam`、`gps_lidar_topk_union_bgam` 和 `gps_lidar_topk_per_candidate_rerank`。
- [x] 7.3 确保 `gps_only` baseline 复用 frozen GPS v2 logits/probs 或 Top8 candidate prob，并与 GPS v2 r15 circular metrics 口径一致。
- [x] 7.4 实现 metrics/predictions 写出：`metrics.json`、`summary_overall.csv`、`summary_by_scene.csv`、`summary_by_bgam_mode.csv`、`predictions.csv`、`run_metadata.json` 和 `resolved_config.yaml`。
- [x] 7.5 实现独立 evaluation CLI，支持 `--config`、`--ckpt`、`--output-dir` 和 ablation/debug mask 开关。
- [x] 7.6 新增 `tests/test_gps_lidar_bgam_runner.py`，使用 synthetic manifest/BEV 完成 CPU smoke 的 manifest、train、eval、metrics 和 predictions 流程。
- [x] 7.7 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_runner.py -q`。

## 8. 文档、CLI Help 与验收

- [x] 8.1 更新 README，新增 GPS+LiDAR BGAM reranker 章节，说明输入 manifest columns、RSU coordinate assumption、beam-angle convention、训练/评估命令、debug masks、输出文件和结果判读。
- [x] 8.2 扩展 `tests/test_cli_help.py`，覆盖新增 BGAM console scripts 和 `python -m kd_sensing.cli.* --help`。
- [x] 8.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
- [x] 8.4 运行 `conda run -n kd_mm_beam kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`。
- [x] 8.5 运行 `conda run -n kd_mm_beam kd-sensing-run-deepsense6g-gps-lidar-bgam --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`。
- [x] 8.6 运行 `conda run -n kd_mm_beam kd-sensing-evaluate-deepsense6g-gps-lidar-bgam --config configs/deepsense6g_gps_lidar_bgam.yaml --ckpt <checkpoint> --output-dir outputs/analysis/deepsense6g_gps_lidar_bgam/eval_smoke`。
- [x] 8.7 运行 `conda run -n kd_mm_beam pytest tests/test_gps_lidar_bgam_geometry.py tests/test_gps_lidar_bgam_dataset.py tests/test_gps_lidar_bgam_model.py tests/test_gps_lidar_bgam_runner.py tests/test_cli_help.py -q`。
- [x] 8.8 运行 `openspec validate add-gps-lidar-bgam-reranker --strict`。
- [x] 8.9 实现完成后运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。
