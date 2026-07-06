## 1. 代码定位与边界确认

- [x] 1.1 定位当前训练入口、Scene31-34 训练/eval 脚本、TinyViT/TinyViT+JEPA 配置、上一轮 PCPG launcher/summary、fusion 逻辑、prototype/unimodal diagnostics、drop-k eval 输出、checkpoint 命名和现有 tests。
- [x] 1.2 确认新增脚本为 local/manual surface，输出边界为 ignored `outputs/bprr_reliability_router_v1/`，不新增 package console script、不覆盖旧 outputs。

## 2. BPRR、raw gate 与 oracle 实现

- [x] 2.1 新增 `raw_conf_gate` fusion，支持 masked softmax、单模态 gate=1、logits 层融合和 gate diagnostics。
- [x] 2.2 新增 `BeamPrototypeReliabilityRouter` / `bprr` fusion，支持 reliability feature 构造、pattern bias、masked softmax、`bprr_fuse_level: logits` 和 prototype feature fallback。
- [x] 2.3 新增 BPRR temperature calibration diagnostics，至少支持 `bprr_calibration: temperature`、正温度和 per-modality 独立参数。
- [x] 2.4 新增 BPRR gate balance regularization 和 radar gate floor regularization，默认关闭，仅训练启用。
- [x] 2.5 补齐 eval-only oracle gate，输出 oracle metrics 和 chosen modality distribution，并明确标注 oracle upper bound。

## 3. Launcher 与 summary

- [x] 3.1 新增 `scripts/launch_bprr_reliability_router_v1.py`，覆盖 e3/e7/e8/e9/e10/e11/e12、GPU 0-7、每 GPU 1 job、总并发 8、dry-run、skip_completed、force、max_epochs override、log 和 manifest。
- [x] 3.2 新增 `scripts/summarize_bprr_reliability_router_v1.py`，输出 summary.csv、summary.md、drop_count_summary.csv、gate_diagnostics.csv 和 oracle_summary.csv，并合并 e5/e6 baseline delta。
- [x] 3.3 确保 e3 自动查找 `outputs/pcpg_radar_balance_v1/e5_pcpg_low_encoder_lr_seed1` 和兼容候选路径的 best checkpoint；找不到时 fail fast 并给出清晰错误。

## 4. 测试与验证

- [x] 4.1 新增或更新 focused tests，覆盖 BPRR masked softmax、temperature calibration、radar gate regularization、oracle gate、launcher dry-run 和 summary parser。
- [x] 4.2 运行 `openspec validate add-bprr-reliability-router --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest -q tests/test_bprr_reliability_router.py`。
- [x] 4.4 运行 BPRR launcher dry-run，并按可用资源运行 e7/e8 `--max_epochs 1` smoke。
