## 1. Baseline Audit and Test Locks

- [x] 1.1 复核现有 `pooler_gps_query_k2_tokens`、`pooler_gps_query_k2_frame`、`pooler_mean` 的 generated config、runtime metadata 和 P0-P5 summary，记录当前 Scene31、S31-S34、S32-S34 baseline 数值到 change notes 或结果摘要。
- [x] 1.2 为现有 `GPSQueryPool(output_mode="tokens")` 增加 focused test，锁定 `[B,T,K,D]` 输出、`[B,T,K,N]` attention map、`k_queries` metadata 和默认 frame 输出不变。
- [x] 1.3 为 `ModularSequenceModel` token feature 汇合逻辑增加 synthetic test，覆盖 image K-token feature 与 GPS frame feature 拼接后进入 token-aware core 的 shape。
- [x] 1.4 为 legacy `token_aware_transformer` metadata 增加测试，确认未声明 readout 时标记为 `legacy_uniform_mean` 或等价语义，且旧配置不被误标记为 learned readout。

## 2. Query Diagnostics

- [x] 2.1 在 GPS-query attention diagnostics 中增加 query entropy、effective patch count、query diversity 和 attended-latent similarity summary。
- [x] 2.2 将 query diagnostics 接入 `jepa_visual_analysis` 或现有 GPS-query evidence helper，输出 machine-readable CSV/JSON 字段，不要求真实 dataset 的单元测试。
- [x] 2.3 增加 synthetic diagnostics tests，覆盖 attention map 存在、attention map missing、token grid missing 和 diagnostics unavailable 的降级行为。
- [x] 2.4 确认 diagnostics 张量 detach，不参与训练 loss 或 readout forward 决策。

## 3. Minimal Token Readout Candidate

- [x] 3.1 设计并实现最小 opt-in token readout candidate，优先为 `gps_query_k2_tokens_weighted_readout` 或等价 learned query-weighted readout，输出 `[B,T,D]`。
- [x] 3.2 为 readout candidate 记录 `token_readout_type`、`k_tokens`、`readout_trainable_params`、readout weight summary 和 output shape metadata。
- [x] 3.3 保持默认 mean、GPS-query frame、legacy token-aware transformer、hybrid residual query 和 Predictive GPS-query++ 配置行为不变。
- [x] 3.4 增加 readout shape、metadata、oracle 信息拒绝和 backward smoke tests。

## 4. Sweep Manifest and Config Generation

- [x] 4.1 扩展 `cnn_hybrid_jepa_visual_prior_sweep` manifest/generator，加入最小 readout candidate，并保留现有 `pooler_gps_query_k2_tokens` variant_id。
- [x] 4.2 为 readout ablation 记录 pooler type、output mode、`k_queries`、readout type、representation core type、checkpoint policy、run tier 和 strict comparability fields。
- [x] 4.3 增加 manifest/generator tests，确认 readout 候选与 `pooler_mean`、`pooler_gps_query_k2_frame`、`pooler_gps_query_k2_tokens` 同组出现。
- [x] 4.4 确认所有新增训练、评估、summary 命令使用 `conda run -n kd_mm_beam`，输出路径位于 ignored `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/` 或显式 output root。

## 5. Paired Gate and Reporting

- [x] 5.1 扩展 summary/gate 逻辑，输出 readout candidate 相对 `pooler_gps_query_k2_frame` 和 `pooler_mean` 的 paired DBA/Top-k delta。
- [x] 5.2 gate 输出 clean/P0、P1-P5 mean、Scene31、S31-S34、S32-S34、P3/P4 delta、threshold、pass/fail、missing evidence 和 caveats。
- [x] 5.3 支持 seeds 17、23、42 的 per-seed 和 mean/std 聚合；缺失 seed 时明确标记为 incomplete，不静默升级 claim。
- [x] 5.4 将 query diagnostics 字段汇入 summary，缺失 attention 或 readout weights 时保留候选行并标记 `missing` 或 `unavailable`。

## 6. Validation and Documentation

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_jepa_visual_architecture_sweep.py tests/test_config_load_characterization.py -q`。
- [x] 6.2 如触碰 `jepa_visual_analysis` 或 evidence helper，追加运行 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q`。
- [x] 6.3 如触碰架构边界、CLI 或配置生命周期，追加运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py -q`。
- [x] 6.4 运行 `openspec validate improve-gps-query-token-readout --strict`。
- [x] 6.5 在最终实现说明中列出新增 readout candidate、paired baseline 数值、gate 状态、未跑完的 seed 或 missing evidence。
