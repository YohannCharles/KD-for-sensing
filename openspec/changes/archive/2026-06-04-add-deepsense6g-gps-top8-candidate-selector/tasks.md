## 1. 文件清单与配置入口

- [x] 1.1 新增 `configs/deepsense6g_top8_selector.yaml`，覆盖 data、candidate、image、model、attention、loss、train、experiment、metrics 和 outputs 默认字段。
- [x] 1.2 在 `pyproject.toml` 增加 Top8 selector console scripts：manifest、run、plot 和 compare 四个入口。
- [x] 1.3 新增包内 CLI 文件：`src/kd_sensing/cli/prepare_deepsense6g_top8_candidate_manifest.py`、`run_deepsense6g_top8_selector.py`、`plot_deepsense6g_top8_selector.py`、`compare_deepsense6g_top8_selector_with_gps_v2.py`。
- [x] 1.4 确认不新增长期维护的顶层 `src.data.*`、`src.models.*`、`src.losses.*` 或 `src.run_*.py` 入口，并更新架构边界测试期望。

## 2. Top8 Candidate Manifest

- [x] 2.1 新增 `src/kd_sensing/data/deepsense6g_topk_candidate_manifest.py`，实现 GPS v2 r15 目录、ratio tag、label space、protocol 和 output path 解析。
- [x] 2.2 实现 strict GPS logits loader，读取 `gps_logits.npy`、`logits.npy` 或 `pred_logits.npy` 以及 logits index，缺失时提示重跑 GPS v2 with `--save-logits`。
- [x] 2.3 从 GPS logits 重新计算 Top8 candidate beams/logits/probs/ranks/dist-to-top1，禁止从 predictions top5 字段截断推导。
- [x] 2.4 合并 GPS context、support/query role、target label、image path、camera AE feature index、LiDAR/radar feature path 和 availability 字段。
- [x] 2.5 计算 `target_in_top8`、`target_candidate_index`、`nearest_candidate_index`、`nearest_candidate_error`、`top8_oracle_error`、`top8_oracle_beam` 和 `top8_miss`。
- [x] 2.6 写出 `top8_candidate_manifest.csv`、metadata JSON、Top8 recall summary 和 warning，并与已有 TopK analysis CSV 做对齐检查。

## 3. Dataset 与 Normalization

- [x] 3.1 新增 `src/kd_sensing/data/deepsense6g_topk_candidate_dataset.py`，读取 manifest 并返回 candidate tensors、GPS context、Top8 labels 和 optional modality fields。
- [x] 3.2 实现 candidate feature 构造：beam sin/cos、rank norm、logit norm、prob、log prob、dist-to-top1 norm、is top1/top3/top5。
- [x] 3.3 实现 GPS context feature 构造：E/N norm、theta、range、heading、speed、GPS prob/margin/entropy 和 GPS pred beam sin/cos。
- [x] 3.4 实现 normalization artifact fit/load/save，确保 scaler 只从 source/support 或 support internal train rows 拟合，不使用 target query。
- [x] 3.5 接入 camera AE feature、image tensor、LiDAR feature 和 radar feature 的按需读取；缺失时返回可诊断标记并允许 GPS context-only selector 运行。

## 4. Selector 模型

- [x] 4.1 新增 `src/kd_sensing/models/topk_candidate_selector.py`，实现 candidate encoder、GPS context encoder、optional modality encoder、candidate scoring MLP、miss head 和 diagnostics。
- [x] 4.2 实现 GPS prior fusion：`final_score = candidate_log_prob + clamp(softplus(lambda_param), 0, lambda_max) * modality_score`，默认 lambda 约为 0.5。
- [x] 4.3 实现 `candidate_probs = softmax(final_candidate_scores)` 和 final beam selection helper。
- [x] 4.4 实现 sparse 64 logits helper，candidate beams 填 score，非 candidate beams 填 `-1e9`。
- [x] 4.5 新增 `src/kd_sensing/models/candidate_attention_selector.py`，实现 candidate query tokens attend GPS/camera AE/image tokens 的 attention selector ablation。
- [x] 4.6 在 `src/kd_sensing/models/__init__.py` 或现有注册机制中按需暴露新 selector，保持轻量导入边界。

## 5. Loss 与 Metrics

- [x] 5.1 新增 `src/kd_sensing/losses/topk_candidate_losses.py`，实现 `TopKCandidateSelectorLoss` 和配置 dataclass。
- [x] 5.2 实现 candidate circular soft CE，target 不在 Top8 时 nearest candidate 权重最大。
- [x] 5.3 实现 target-in-Top8 index CE、miss BCE、GPS prior anchor KL、entropy regularization 和 hard-rank sample weighting。
- [x] 5.4 确保 loss 只使用 source/support 可训练样本，返回 `query_label_used_for_training=False` diagnostics。
- [x] 5.5 复用或扩展现有 circular metrics helper，确保 wrap-around distance、DBA、pmN、P_error_lt4 与 GPS v2 baseline 口径一致。

## 6. Runner、Ablation 与输出

- [x] 6.1 新增 `src/kd_sensing/engine/deepsense6g_top8_selector.py`，实现 manifest 自动生成/读取、fold 遍历、source/support/query split 和 seed/device 管理。
- [x] 6.2 实现 `gps_top1_baseline`、`gps_top8_oracle`、`gps_candidate_prob`、`gps_context_only_selector`、`camera_ae_only_selector`、`camera_ae_gps_selector`、`camera_ae_gps_selector_anchor`、`candidate_attention_selector` 和 `top8_selector_no_gps_prior_fusion`。
- [x] 6.3 实现 `support_only` 与 `source_pretrain_target_finetune`，target query 只用于最终 eval，source 或 camera AE 缺失时记录降级/跳过原因。
- [x] 6.4 写出 `summary_overall.csv`、`summary_by_scene.csv`、`summary_by_top8_hit_miss.csv`、`predictions.csv`、`selection_events.csv`、`candidate_rank_distribution.csv`、`metrics.csv` 和 `run_metadata.json`。
- [x] 6.5 确保 `gps_top1_baseline` 能复现 GPS v2 r15 baseline，`gps_top8_oracle` exact acc 接近 manifest Top8 recall。

## 7. Plotter、Comparison 与 README

- [x] 7.1 新增 Top8 selector plotter，实现 ENU scatter、improvement、rank distribution、hit/miss spatial map、residual histogram、signed residual、label distribution、calibration 和 miss diagnostics figures。
- [x] 7.2 image 可用时生成 selector 成功修正、改坏样本和 Top8 miss montage；不可用时记录 skipped reason。
- [x] 7.3 新增 GPS v2 comparison helper，写出 `comparison_with_gps_v2.csv` 和 `comparison_report.md`，自动回答 scenario31/33、scenario32/34、Top8 oracle、camera AE、GPS prior fusion 和 miss head 诊断问题。
- [x] 7.4 更新 README，新增 “DeepSense6G GPS Top8 Candidate Selector” 章节，说明动机、输入输出、loss、GPS prior fusion、miss head、运行命令和结果判读。

## 8. 测试与验收

- [x] 8.1 新增 `tests/test_topk_candidate_manifest.py`，覆盖字段完整性、strict logits requirement、Top8 recall 和不从 top5 截断推导。
- [x] 8.2 新增 `tests/test_topk_candidate_selector.py`，覆盖 forward shape、candidate_probs sum、sparse 64 logits 和 lambda 为 0 时 GPS ranking。
- [x] 8.3 新增 `tests/test_topk_candidate_losses.py`，覆盖 candidate soft label、target-in-Top8 CE mask、miss BCE、prior anchor 和 hard-rank weighting。
- [x] 8.4 新增 `tests/test_candidate_attention_selector.py`，覆盖 camera AE pseudo-token、GPS token、image token 兼容和 output shape。
- [x] 8.5 新增或扩展 circular metrics 测试，覆盖 `target=1`、`candidate=63`、`num_beams=64` 时 distance 为 2。
- [x] 8.6 运行 `conda run -n kd_mm_beam pytest tests/test_topk_candidate_manifest.py tests/test_topk_candidate_selector.py tests/test_topk_candidate_losses.py tests/test_candidate_attention_selector.py tests/test_circular_metrics.py -q`。
- [x] 8.7 运行 `conda run -n kd_mm_beam kd-sensing-prepare-deepsense6g-top8-candidate-manifest --config configs/deepsense6g_top8_selector.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`。
- [x] 8.8 运行 `conda run -n kd_mm_beam kd-sensing-run-deepsense6g-top8-selector --config configs/deepsense6g_top8_selector.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`。
- [x] 8.9 运行 `conda run -n kd_mm_beam kd-sensing-plot-deepsense6g-top8-selector --results-dir outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled`。
- [x] 8.10 运行 `conda run -n kd_mm_beam kd-sensing-compare-deepsense6g-top8-selector-with-gps-v2 --gps-v2-root outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep --selector-root outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 8.11 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 8.12 实现完成后运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。
