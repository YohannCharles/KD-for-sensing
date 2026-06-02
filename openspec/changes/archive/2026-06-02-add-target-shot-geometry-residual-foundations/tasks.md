## 1. Split 配置与 artifact 基础

- [x] 1.1 新增 target-shot split 配置解析，支持 `split.domain_type`、`source_domains`、`target_domains`、`target_label_fraction`、`target_label_selection`、`seed`、`allow_target_unlabeled` 和 artifact 输出路径。
- [x] 1.2 实现 domain key 构造工具，支持 scenario、weather/condition、scenario_weather、town_scenario_weather，并在字段缺失时输出包含 dataset type 与缺失字段的清晰错误。
- [x] 1.3 实现 source、target_labeled、target_unlabeled、target_test 拆分逻辑，确保 sample id 无交集，并保留序列窗口 guard band/leakage diagnostics。
- [x] 1.4 实现 target_labeled 采样策略：random、stratified_by_beam、stratified_by_geo_sector、stratified_by_weather，记录每个分层桶候选数、选中数和 fallback reason。
- [x] 1.5 写出 split artifact JSON/NPZ，包含 sample ids/indices、配置摘要、输入 fingerprint、domain metadata、统计、sampling manifest、leakage diagnostics 和 strict eligibility。
- [x] 1.6 实现 split artifact 复用校验，输入 fingerprint、seed、domain、target fraction 或 sample ids 不匹配时拒绝复用并提示 regenerate/overwrite。

## 2. Geometry-residual label 工具

- [x] 2.1 新增或扩展轻量 geometry/beam utility 模块，实现 relative position、azimuth、angle_to_beam、circular_beam_distance、beam_to_residual、residual_to_beam。
- [x] 2.2 支持 full circular residual 与 signed circular residual convention，并在 metadata 中固定 convention。
- [x] 2.3 实现 `max_residual` clipped residual class、overflow/ignore/boundary 策略、`make_residual_class` 和 `residual_class_to_delta`。
- [x] 2.4 实现 geo_sector 构造，支持按 `num_geo_sectors` 从 `geo_angle` 或 `beam_geo` 生成 sector，并记录 boundary 或 beam-to-sector mapping。
- [x] 2.5 复用或适配 MMW direct relative geometry 字段，缺失 geometry 时按 `label_space.geometry.required` 失败或记录 unavailable reason。

## 3. Dataset runtime 与加载集成

- [x] 3.1 在 target provider 或 dataset runtime 中接入 `label_space.type: geometry_residual`，按需暴露 `beam_abs`、`beam_geo`、`beam_residual`、`residual_class`、`geo_angle`、`geo_sector`。
- [x] 3.2 确保默认 absolute label 配置不要求 GPS/pose/relative geometry，且 sample keys 与既有训练、评估路径兼容。
- [x] 3.3 在 dataloader/runtime metadata 中记录 target-shot split artifact、source/target domains、target_label_fraction、各 split 样本数、strict eligibility 和 target schema。
- [x] 3.4 实现 target_labeled/target_unlabeled/target_test subset 标记传递，训练 payload 对 target_unlabeled 的 beam/residual/physical/path/radio supervision 访问必须触发 guard。
- [x] 3.5 确保 target_test label 只在 evaluation scope 可用于 metrics，不参与 adaptation、threshold、prototype、temperature 或 early stopping。

## 4. MMW 协议接入

- [x] 4.1 将 target-shot split builder 接入 MMW availability/manifest，支持 scenario-level、town-level 和 weather/condition-level split。
- [x] 4.2 保留 MMW group-safe sequence split 的 frame/window overlap、guard band 和 strict eligibility diagnostics。
- [x] 4.3 为 MMW geometry-residual label 统计接入 RSU-CAV direct relative geometry、uniform angle quantization 或 codebook mapping metadata。
- [x] 4.4 在 MMW split/summary metadata 中记录 `beam_geo_source`、geometry unavailable reason、target_label_fraction 和 target_labeled selected sample ids。
- [x] 4.5 更新 MMW eligibility audit，使 target_labeled beam/residual 监督合法，但 target_unlabeled/target_test oracle 使用仍会被排除。

## 5. 分布诊断命令与产物

- [x] 5.1 新增 distribution shift analysis CLI 或脚本，输入 config、split artifact 和 output_dir，不训练模型即可运行。
- [x] 5.2 输出 source、target_labeled、target_unlabeled 可选、target_test 的 absolute beam、geometry beam、residual beam histogram。
- [x] 5.3 实现 KL、JS、Wasserstein/EMD、total variation distance，支持 smoothing 与 ordered/circular class 语义记录。
- [x] 5.4 写出 `distribution_shift_metrics.json`、histogram CSV/JSON、summary metadata 和可选 PNG/PDF；可视化依赖不可用时仍写 JSON/CSV。
- [x] 5.5 在 summary 中解释 `emd_absolute` 与 `emd_residual`，只报告分布距离事实，不自动声明模型性能提升。

## 6. 配置、示例与文档

- [x] 6.1 新增最小示例配置，覆盖 5% target-shot split、geometry_residual label_space 和 distribution diagnostics 输出路径。
- [x] 6.2 新增示例命令：生成 split、打印分布诊断、构建 geometry-residual dataset one batch，所有 Python 命令必须使用 `conda run -n kd_mm_beam`。
- [x] 6.3 更新相关 README/docs 或 OpenSpec 注释，说明本变更只提供 split/label/diagnostics 基础，不包含新 residual neural network。

## 7. 测试与验证

- [x] 7.1 新增 split 单元测试，验证 target_labeled 约 5%、固定 seed 可复现、split sample ids 无交集、artifact mismatch 会被拒绝。
- [x] 7.2 新增 geometry-residual 单元测试，验证 wrap-around distance、residual_to_beam 可逆、clipped residual 范围和 overflow metadata。
- [x] 7.3 新增 dataset/runtime smoke test，验证 geometry_residual sample keys、默认 absolute path 兼容和 target_unlabeled guard。
- [x] 7.4 新增 distribution diagnostics 测试，验证 metrics JSON/CSV 产物存在、空 bin smoothing 可用、EMD absolute/residual 字段齐全。
- [x] 7.5 运行相关测试：`conda run -n kd_mm_beam pytest <新增测试文件> -q`。
- [x] 7.6 运行架构快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.7 运行 OpenSpec 校验：`openspec validate add-target-shot-geometry-residual-foundations --strict`。
- [x] 7.8 运行状态检查：`openspec status --change add-target-shot-geometry-residual-foundations`，确认 artifacts complete 且 ready for apply。
