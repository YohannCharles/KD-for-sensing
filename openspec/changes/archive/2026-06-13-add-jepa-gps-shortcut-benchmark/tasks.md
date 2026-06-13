## 1. Benchmark Manifest 与配置骨架

- [x] 1.1 设计并实现 benchmark manifest schema，覆盖 models、protocol、perturbation_suites、metrics、figures、seeds、outputs 和 comparability 字段
- [x] 1.2 增加 manifest 解析与校验 helper，未知模型 key、缺失 config/weights、未知 suite type 和非法 severity 必须报清晰错误
- [x] 1.3 增加最小 smoke benchmark manifest fixture，使用 mock/synthetic 数据，不读取真实 `dataset/`
- [x] 1.4 增加 canonical JEPA vs GPS shortcut benchmark 配置示例，引用现有 Vision-Position baseline 与 JEPA downstream 配置但不提交 checkpoint

## 2. Deterministic Perturbation Suite

- [x] 2.1 实现 GPS clean、Gaussian jitter、cumulative drift、missing/dropout 和 GPS distractor intervention transform
- [x] 2.2 实现 image fog/rain、night、occlusion 和 motion blur transform，并保持 image batch shape、dtype 和 metadata
- [x] 2.3 实现 image/GPS temporal delay 与 sampling-rate mismatch transform，记录 delay 单位、帧偏移和 fallback 策略
- [x] 2.4 为所有 transform 增加 seed、sample id、suite id、severity 驱动的 deterministic replay 机制
- [x] 2.5 为 missing GPS、delay 不足、attention 不可用等情况统一 warning schema

## 3. Benchmark Runner 与模型可比性

- [x] 3.1 新增包内 benchmark runner 或扩展现有 analysis/diagnostics runner，不新增仓库根旧入口
- [x] 3.2 runner 复用现有配置加载、dataset runtime、model registry、checkpoint loading 和 evaluation metrics
- [x] 3.3 实现 evaluation-only 协议，确保只读 config、weights、split CSV 和训练 run 目录
- [x] 3.4 实现可选 train-then-evaluate 协议，训练命令和验证命令必须使用 `conda run -n kd_mm_beam ...`
- [x] 3.5 实现模型可比性校验，覆盖 split、sample_count、label_space、metric_profile、normalization artifact、checkpoint provenance 和 enabled modalities

## 4. 指标聚合与 Benchmark 产物

- [x] 4.1 生成 `benchmark_manifest.json`，记录输入 digest、模型配置/权重、seeds、suite 参数、split metadata、warnings 和输出文件清单
- [x] 4.2 生成 `metrics_by_condition.csv`，每行包含 model、suite、condition、severity、seed、split、sample_count、primary metric 和 clean delta
- [x] 4.3 生成 `robustness_summary.csv`，包含 relative drop、collapse slope、area-under-robustness-curve 和可比较性状态
- [x] 4.4 实现 drop GPS、drop image、misleading GPS 和 GPS-only collapse slope 的 shortcut reliance 汇总
- [x] 4.5 确保所有 benchmark 输出写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定本地产物目录

## 5. JEPA Visual Analysis 集成

- [x] 5.1 扩展 `kd-sensing-jepa-visual-analysis` 配置解析，使其可引用 benchmark manifest 或 runner 输出 manifest
- [x] 5.2 从 benchmark 指标表生成 robustness matrix、GPS collapse curve、image degradation curve 和 temporal delay curve
- [x] 5.3 在 `report.md` 中新增 GPS shortcut reliance 小节，区分性能结果、counterfactual intervention 和解释性诊断
- [x] 5.4 增加 benchmark case study 选择，覆盖 `jepa_recovery`、`gps_shortcut_failure` 和 `shared_failure`
- [x] 5.5 对缺失 attention、embedding、case payload 或 optional visualization dependency 的场景实现可审计降级

## 6. 测试与回归

- [x] 6.1 添加 manifest schema 单元测试，使用 `conda run -n kd_mm_beam pytest <focused-tests> -q` 运行
- [x] 6.2 添加 synthetic image/GPS batch perturbation determinism 和 shape 保持测试
- [x] 6.3 添加 benchmark metrics aggregation 测试，覆盖 clean delta、relative drop、collapse slope 和 unavailable warning
- [x] 6.4 添加 JEPA visual analysis benchmark ingestion 测试，验证 analysis manifest、summary 表和 report skeleton
- [x] 6.5 添加 CLI help smoke，使用 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help` 以及新增 runner help 命令
- [x] 6.6 运行架构边界相关回归：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 6.7 OpenSpec 实现完成后运行 `openspec validate add-jepa-gps-shortcut-benchmark --strict`

## 7. 文档与复现说明

- [x] 7.1 在实验文档中记录 benchmark 入口、manifest 字段、推荐模型矩阵和最小 smoke 命令
- [x] 7.2 记录论文图产物路径、表格字段和不能过度声称的 caveat
- [x] 7.3 明确本地产物边界：真实数据、checkpoint、cache、metrics、figures 和 reports 默认不提交
- [x] 7.4 更新项目 surface/inventory 文档中新增包内入口或配置位置，避免把 outputs 中的运行产物写成源码需求

## 8. Scenario C 异步位置反馈补强

- [x] 8.1 扩展 benchmark manifest schema，支持 Scenario C / Asynchronous Position Feedback suite 或 preset，并声明 `C0_sync`、`C1_mild_stale`、`C2_low_rate`、`C3_random_async`、`C4_severe_async` 的 delay、stride、dropout 和 fallback 参数
- [x] 8.2 实现 Scenario C GPS transform，输出 `gps_async`、`gps_valid_mask`、`gps_delay_steps` 或等价字段，并保证 beam label、power target、sample id 和未启用退化的 image sequence 保持不变
- [x] 8.3 实现低采样率 GPS 和 random async drift，支持 deterministic stride sampling、dropout mask、forward-fill/zero-fill fallback，并保留 stale 或 invalid metadata
- [x] 8.4 支持 timestamp-based delay；当 timestamp 不可用时降级为 frame-index delay，并在 warnings 或 perturbation metadata 中记录 fallback
- [x] 8.5 更新 canonical smoke 与 beambench fair benchmark 配置，使它们包含 Scenario C C0-C4 preset、Protocol A/B/C 或明确的 evaluation-only 子集
- [x] 8.6 扩展指标和图表聚合，覆盖 Top-1/Top-3/Top-5、mean beam index error、accuracy vs delay/dropout/stride、`accuracy(Ck)/accuracy(C0)` 鲁棒性比值，以及 image-only missing GPS 切片
- [x] 8.7 添加 Scenario C focused tests：toy GPS `[0,1,2,3,4]` 在 delay=2 下得到 `[invalid, invalid, 0,1,2]`、label 不变、无未来 GPS 泄漏、mask 正确、固定 seed deterministic
- [x] 8.8 使用 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q` 运行 focused 测试，并运行 `openspec validate add-jepa-gps-shortcut-benchmark --strict`
