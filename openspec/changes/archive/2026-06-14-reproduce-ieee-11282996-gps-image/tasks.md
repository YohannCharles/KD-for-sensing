## 1. Source audit 与协议固化

- [x] 1.1 新增 IEEE `11282996` source audit schema/helper，记录 title、DOI、venue/year、IEEE URL、PDF/code/source availability、dataset scene、split、modalities、target label、metric profile、official weight 状态和 blocked reason。
- [x] 1.2 审计用户提供的 IEEE PDF/BibTeX/作者代码或公开作者页面，生成 machine-readable audit digest；若 IEEE 仍不可访问，明确标记 `blocked_official`。
- [x] 1.3 将 source audit 结果接入 paper runner/report，禁止缺少关键官方协议时标记 `official_reproduction`。
- [x] 1.4 添加 source audit 单元测试，覆盖 metadata 完整、IEEE blocked、缺失官方权重和 claim status gating。

## 2. DeepSense6G Scenario 23 场景描述符

- [x] 2.1 在 DeepSense6G scene descriptor/解析逻辑中新增 Scenario 23，支持 `23`、`scene23` 和 `scenario23`。
- [x] 2.2 为 Scenario 23 声明默认数据根目录、legacy data_root、train/test CSV 名和 scene-scoped output slug。
- [x] 2.3 确保显式 `data_root`、`train_csv_name` 和 `test_csv_name` 覆盖默认值，同时 runtime metadata 仍记录规范 Scenario 23。
- [x] 2.4 添加配置/场景解析测试，覆盖整数、别名、默认路径、显式路径和未知场景错误。

## 3. GPS+Image-only 配置与 LiDAR 禁用护栏

- [x] 3.1 新增 IEEE `11282996` paper preset/manifest，默认使用 DeepSense6G Scenario 23、`modalities: [image, gps]`、paper-audited target source 和 paper-audited GPS feature mode。
- [x] 3.2 实现 paper preset 的模态校验：拒绝 `lidar`、`use_lidar: true`、radar/mmWave/CSI/all-modalities、GPS+LiDAR BGAM checkpoint 或需要未启用模态的 fallback。
- [x] 3.3 确保 CSV 文件名包含 `LIDAR` 时不会被误判为使用 LiDAR；实际输入以 enabled modalities、dataset flags 和 batch keys 为准。
- [x] 3.4 添加 config loading 和 LiDAR override 测试，使用 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 或新增 focused test。

## 4. Paper model groups 与 workflow

- [x] 4.1 在 Vision-Position baseline suite 中登记 IEEE `11282996` model groups：image-only、GPS-only 和 Image+GPS fusion。
- [x] 4.2 优先用 `modular_sequence`/现有 encoder/core/head 表达 image-only、GPS-only 和 Image+GPS fusion；如需特殊两阶段或作者代码协议，新增 `src/kd_sensing/baselines/ieee11282996/` 窄 workflow helper。
- [x] 4.3 为每个 model group 输出 `training_strategy_metadata()` 或等价 run metadata，记录 enabled modalities、image encoder、GPS feature mode、fusion type、num beams、target source、claim status 和 `uses_lidar: false`。
- [x] 4.4 添加 synthetic forward smoke，覆盖 image-only、GPS-only、Image+GPS fusion 的 logits shape、metadata 和普通 runtime output adaptation。

## 5. Metrics、report 与产物边界

- [x] 5.1 实现 paper-aligned metrics aggregation，至少输出 Top-1、Top-3、Top-5；可用时追加 DBA、beam-distance、overhead reduction 或 paper 等价字段，并用不同字段名记录口径。
- [x] 5.2 实现 report/manifest writer，记录 source audit digest、命令、git status 摘要、scenario、enabled modalities、checkpoint provenance、metric profile、warnings 和文件清单。
- [x] 5.3 默认输出到 `outputs/analysis/ieee_11282996_gps_image/` 或 `outputs/scene23/<run_name>/`，并确保 checkpoint、predictions、cache、plots、reports 均不进入源码变更。
- [x] 5.4 添加 mock/synthetic smoke，确保 mock metrics、checkpoint metadata 和 report 标记 `mock_data: true` 且 claim status 为 `mock_smoke`。

## 6. CLI、文档与结果账本

- [x] 6.1 如需要用户入口，新增包内 CLI 或当前 allowlist 薄 alias，并同步 `pyproject.toml`、README、`docs/project_surface_inventory.md` 和架构边界测试。
- [x] 6.2 更新 `README_REPRODUCE.md`、`BASELINE_REPORT.md`、`docs/mainline_model_catalog.md` 或 `docs/result_claims_registry.md`，记录 IEEE `11282996` 的命令、claim status、blocked/local substitute caveat 和不使用 LiDAR 的边界。
- [x] 6.3 文档不得提交或引用新生成真实 metrics/checkpoint/cache；只记录本地产物路径模式、source audit digest 和可复现命令。
- [x] 6.4 若 source audit 后发现 article `11282996` 并非 Scenario 23 或公开作者页面对应论文，先更新本 change 的 OpenSpec artifact，再继续实现。

## 7. Validation

- [x] 7.1 运行 `openspec validate reproduce-ieee-11282996-gps-image --strict`。
- [x] 7.2 运行 focused tests：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`，并追加新增 IEEE `11282996` focused tests。
- [x] 7.3 运行 CLI help smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`，如新增 CLI 则运行对应 `--help`。
- [x] 7.4 若触碰 model/forward/metadata，运行对应 model focused tests；若触碰 dataset scene descriptor，运行对应 dataset/config focused tests。
- [x] 7.5 最终说明中记录未运行的真实长训练、缺失的 IEEE/官方数据或权重、claim status 和剩余风险。
