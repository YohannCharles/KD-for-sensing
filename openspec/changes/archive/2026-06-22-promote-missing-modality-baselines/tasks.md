## 1. Baseline 配置

- [x] 1.1 将 AMBER-lite 默认 image encoder 改为不下载外部权重，并保留 missing-modality dropout。
- [x] 1.2 将 WCL-style local baseline 的训练期 modality dropout 迁移到 `difficulty.profiles`。
- [x] 1.3 新增 TII-VLRG-style 本地可训练 baseline 配置，使用 `modular_sequence` 和本地 scratch 权重。

## 2. 文档与边界

- [x] 2.1 更新主线模型目录、实验协议、claim 账本和实验矩阵，明确三者是 local experimental baseline。
- [x] 2.2 保留 TII external wrapper 和 WCL source audit 作为可选 external/audit 路径，不作为训练前置。

## 3. 测试与验证

- [x] 3.1 更新 focused tests，覆盖 AMBER/WCL/TII local baseline 配置可加载、可构建且不默认外部权重。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_amber_lite_missing_modality.py tests/test_wcl2025_missing_modality.py tests/test_tii_vlrg_transformer.py -q`。
- [x] 3.3 运行 `openspec validate promote-missing-modality-baselines --strict` 和必要架构边界测试。
