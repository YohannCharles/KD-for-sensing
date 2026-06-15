## 1. 基线骨架与测试夹具

- [x] 1.1 新建 `src/kd_sensing/baselines/jepa_msac/` 包，放置 workflow orchestration、manifest/report schema、metrics helper 和公开 `__init__`。
- [x] 1.2 新建 `src/kd_sensing/models/jepa_msac.py` 或等价窄模块，预留 JEPA-MSAC 模型组件并通过当前 registry 暴露必要模型/组件。
- [x] 1.3 新建 synthetic JEPA-MSAC fixture helper，用随机 tensor 构造 image/radar/lidar/gps/RF history 和 localization/beam/RSSI targets，不读取真实 `dataset/`。
- [x] 1.4 增加最小 config fixture 或 smoke YAML，覆盖 `workflow.family: jepa_msac`、Scenario 32、`T_hist=8`、`T_pred=5`、64 beams 和 output boundary。
- [x] 1.5 添加初始 focused test 文件，先验证配置可加载、fixture shape、包导入轻量性和没有 runtime artifact 写入源码目录。

## 2. 数据协议与 manifest

- [x] 2.1 实现 JEPA-MSAC manifest schema，记录 scene、CSV/manifest 来源、window length、history/prediction length、split seed、70/30 ratio、enabled modalities、target schema 和 output path。
- [x] 2.2 实现 Scenario 32 dry-run/audit manifest builder，只读检查本地 DeepSense6G 字段可用性，缺字段时输出 blocked reason。
- [x] 2.3 实现 13 帧 sliding window sample assembly，产生历史输入和未来 localization/beam/RSSI targets 的 flat sample schema。
- [x] 2.4 实现 RF history workflow-local mapping，把 beam-power/RSRP vector 映射为 `[B,T,K]` 输入，并在 metadata 中记录 `paper_modality: RF`。
- [x] 2.5 添加测试：manifest dry-run 不读取大数组、不写源码目录，RF 不进入 canonical modalities，缺字段报告包含字段名和修复提示。

## 3. 多模态 tokenizer 与 JEPA backbone

- [x] 3.1 实现 Image tokenizer，默认 paper-aligned 输出 `[B,T,9,D]`，并提供 smoke-friendly 轻量 fallback 或可配置 vision backbone。
- [x] 3.2 实现 Radar tokenizer，将 range-angle map 或现有 radar feature 转为 `[B,T,16,D]`，并对输入 shape 给出清晰错误。
- [x] 3.3 实现 LiDAR tokenizer，将 depth/BEV/projection feature 转为 `[B,T,16,D]`，并记录输入 profile。
- [x] 3.4 实现 GPS 与 RF state tokenizer，分别输出 `[B,T,1,D]`，包含 linear projection 和 LayerNorm。
- [x] 3.5 实现 factorized temporal/modality/intra-frame positional embeddings，并在超出 `max_frames`、`max_modalities` 或 `max_tokens` 时失败。
- [x] 3.6 实现 JEPA-MSAC context encoder、EMA target encoder、mask token predictor 和 predictive latent pooling。
- [x] 3.7 添加 synthetic forward tests，验证 token counts、concat index metadata、position embedding、encoder输出和 `training_strategy_metadata()`。

## 4. Stage 1 temporal block-masked JEPA

- [x] 4.1 实现 per-modality contiguous temporal block mask sampler，支持 `rho=0.5`、random/checkerboard/ablation pattern 和可复现 seed。
- [x] 4.2 实现 masked latent SmoothL1 loss，只在 `I_mask` 上计算，并记录 mask ratio、target token count、EMA momentum 和 latent norm diagnostics。
- [x] 4.3 增加 JEPA-MSAC pretraining objective metadata 或 workflow-local loss adapter，确保 early stopping metric 为 JEPA loss 而不是 beam 指标。
- [x] 4.4 增加 training extension 或 workflow runner hook，在 optimizer step 后更新 target encoder EMA，并保存/恢复 EMA 状态。
- [x] 4.5 添加 Stage 1 smoke test：`conda run -n kd_mm_beam pytest tests/test_jepa_msac*.py -q` 能完成 forward、loss、backward、EMA update 和 checkpoint metadata 检查。

## 5. Stage 2 frozen inference 与 task heads

- [x] 5.1 实现从 Stage 1 checkpoint 加载 frozen backbone 的 helper，默认冻结 context encoder、target encoder 和 predictor，并汇总 trainable parameter metadata。
- [x] 5.2 实现 future latent inference：历史 tokens 作为 keep，未来 slots 作为 mask，输出 `S_pred` 形状 `[B,T_pred,D]`。
- [x] 5.3 实现 localization head，支持 constant-velocity coarse estimate、bootstrap MLP fallback、residual correction 和 L1 loss。
- [x] 5.4 实现 beam prediction head，支持 `[S_pred; predicted_location]` cascading、projection+GRU+MLP logits 和 cross-entropy loss。
- [x] 5.5 实现 RSSI/link-strength head，支持 beam-power residual/profile output、scalar RSSI output 和 SmoothL1/MAE/RMSE 相关 loss 配置。
- [x] 5.6 添加 Stage 2 tests，验证 backbone 冻结、optimizer 只含 heads、三任务输出 shape、关闭/开启 localization guidance metadata。

## 6. 指标、report 与 ablation

- [x] 6.1 实现 ADE、FDE、Top-1、Top-3、L1-RSRP diff、RSSI RMSE 和 RSSI MAE metrics，并支持 horizon-wise/aggregate 输出。
- [x] 6.2 实现 RRankMe 和 RLDA representation quality metrics，RLDA 缺增强视图时标记 unavailable。
- [x] 6.3 实现 report writer，输出 JSON/Markdown/CSV summary 到 ignored `outputs/analysis/` 或用户显式输出目录。
- [x] 6.4 实现 ablation manifest writer，记录 latent dim、mask ratio、mask pattern、modality ablation、untrained/E2E/frozen-head、loc aux 和 missing-history rows。
- [x] 6.5 添加 metric/report tests，覆盖缺 beam-power reference、未运行 ablation row、claim status 和 caveat 字段。

## 7. CLI、配置与入口声明

- [x] 7.1 新增包内 CLI `kd_sensing.cli.run_jepa_msac`，支持 `--config`、`--stage pretrain|heads|evaluate|report|all`、`--dry-run`、`--pretrained-checkpoint`、`--output-dir`。
- [x] 7.2 在 `pyproject.toml` 增加 `kd-sensing-run-jepa-msac` console script，并确保 help smoke 使用 `conda run -n kd_mm_beam kd-sensing-run-jepa-msac --help`。
- [x] 7.3 新增 paper-aligned 配置和 smoke/lowmem 配置，默认输出限定在 ignored `outputs/` 或 `logs/`。
- [x] 7.4 确保 CLI 不新增 root-level 脚本；如确需 thin alias，先同步 allowlist、README 和架构边界测试。
- [x] 7.5 添加 CLI/config tests：help、dry-run、stage dispatch、invalid stage、retired field guard 和 output path metadata。

## 8. 文档与生命周期同步

- [x] 8.1 更新 `docs/mainline_model_catalog.md`，新增 JEPA-MSAC local-ready/unverified workflow 行，说明配置、入口、数据场景、指标和 caveat。
- [x] 8.2 更新 `docs/experiment_protocols.md`，记录 paper-aligned 与 smoke/lowmem 口径、epochs、batch size、learning rate、mask ratio、split 和输出目录。
- [x] 8.3 更新 `docs/result_claims_registry.md`，添加 JEPA-MSAC claim placeholders，默认状态为 `unverified`、`local-ready`、`blocked` 或 `mock/smoke`。
- [x] 8.4 更新 `docs/experiment_matrix.md` 和 README 简短索引，指向 JEPA-MSAC workflow 入口和 caveat，不复制完整结果账本。
- [x] 8.5 更新 `docs/project_surface_inventory.md` 和 `tests/test_architecture_boundaries.py`，登记新增 package CLI、configs、baseline package 和文档 lifecycle。

## 9. 验证与收尾

- [x] 9.1 运行 `openspec validate reproduce-jepa-msac --strict` 并修复 OpenSpec 问题。
- [x] 9.2 运行 JEPA-MSAC focused tests：`conda run -n kd_mm_beam pytest tests/test_jepa_msac*.py -q`。
- [x] 9.3 运行架构/CLI/config focused checks：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 9.4 运行既有 JEPA 回归：`conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_jepa_visual_analysis.py -q`。
- [x] 9.5 检查 `git status --short`，确认没有 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、metrics 或 TensorBoard event 进入源码变更。
- [x] 9.6 在实现总结中记录未运行的长训练、真实数据依赖、claim status 和剩余复现风险。
