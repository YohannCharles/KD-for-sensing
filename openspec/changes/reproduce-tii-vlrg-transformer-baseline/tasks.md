## 1. Workflow Scaffold

- [ ] 1.1 新增 `kd_sensing.baselines.tii_vlrg_transformer` 窄 owner，包含 manifest schema、dry-run payload 和输出路径 helper。
- [ ] 1.2 新增包内 CLI 或 console script，所有项目 Python 命令示例必须使用 `conda run -n kd_mm_beam ...`。
- [ ] 1.3 添加配置或 manifest 模板，声明 source repo、source commit、checkpoint path、scene set、enabled modalities 和 output root。

## 2. External Run And Ingestion

- [ ] 2.1 实现 dry-run 命令构造，记录外部预处理/推理命令但不启动真实训练。
- [ ] 2.2 实现 artifact availability 检查，缺失 repo/checkpoint/prediction 时写出 `pending`、`unavailable` 或 `blocked` 状态。
- [ ] 2.3 实现 TII metrics/prediction CSV ingestion，输出统一 DBA summary row 和 provenance 字段。

## 3. Comparability And Output Boundary

- [ ] 3.1 为 summary row 写出 split、scene set、label space、metric profile、history window、GPS source window、prediction horizon、seed 和 difficulty digest。
- [ ] 3.2 comparability mismatch 时将 row 标记为 `external_reference` 或 `not_comparable`，禁止进入 strict ranking。
- [ ] 3.3 确认 checkpoint、cache、prediction、metrics 和 logs 默认写入 ignored `outputs/analysis/tii_vlrg_transformer_reproduction/`。

## 4. Tests And Documentation

- [ ] 4.1 添加 synthetic manifest / fixture metrics focused tests，不读取真实 `dataset/`、外部 repo 或 checkpoint。
- [ ] 4.2 运行 `conda run -n kd_mm_beam pytest <tii focused tests> -q`。
- [ ] 4.3 更新 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和必要 README 索引，标记 TII 为 external workflow baseline。

## 5. Validation

- [ ] 5.1 运行 `openspec validate reproduce-tii-vlrg-transformer-baseline --strict`。
- [ ] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
