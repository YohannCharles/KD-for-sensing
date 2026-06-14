## 1. 基础结构与 schema

- [x] 1.1 新增 difficulty 窄模块结构，放置 profile schema、context、result、metadata/warning 类型和 no-op pipeline 入口。
- [x] 1.2 实现 difficulty profile 标准化与 digest 生成，覆盖 profile id、operator、stage/split、condition、severity、seed、fallback 和 affected modalities。
- [x] 1.3 新增 target preservation guard，显式阻止 operator 改写 beam target、beam power、soft target、auxiliary target、sample id 和 split metadata。
- [x] 1.4 为 schema 和 digest 添加 synthetic/unit tests，使用 `conda run -n kd_mm_beam pytest <focused-tests> -q` 验证。

## 2. Operator registry 与内置 operators

- [x] 2.1 新增 `DIFFICULTY_OPERATORS` 或等价窄 registry，并保持 registry 导入不触发 dataset、model、diagnostics renderer 或训练循环。
- [x] 2.2 实现默认 difficulty operator 显式注册函数，并接入构建前注册流程。
- [x] 2.3 实现 GPS operators：clean、Gaussian jitter、cumulative drift、missing/dropout、distractor、fixed/random delay、low-rate stride、forward-fill/zero-fill fallback、timestamp delay。
- [x] 2.4 实现 image operators：clean、fog/rain、night、occlusion、motion blur，并记录输入空间和作用帧范围。
- [x] 2.5 添加 operator determinism、shape/dtype preservation、GPS no-future-leak、timestamp fallback 和 unknown operator error tests。

## 3. 配置解析与 runtime metadata

- [x] 3.1 在配置 validation/normalization 中解析 difficulty profiles，支持实体 YAML、virtual/overlay 后的命令行覆盖和稳定 digest。
- [x] 3.2 拒绝未知模态、伪模态名称、非法 stage/split、非法 severity 和 target-shift 配置。
- [x] 3.3 在 run metadata、dataloader metadata 或 artifact writer 中记录 resolved difficulty profiles、stage/split、operator digest、seed 和 warnings summary。
- [x] 3.4 为 clean/no-op 配置添加兼容测试，确认未配置 difficulty 时现有训练和评估 resolved config 不变。

## 4. 训练与评估集成

- [x] 4.1 在 supervised/JEPA batch step 的 `prepare_task_batch` 后、label/target 提取前接入 train-stage difficulty hook，未配置时 no-op。
- [x] 4.2 在 evaluation pass 接入 evaluation/test-stage difficulty hook，并确保 evaluation sweep 不隐式影响训练 dataloader。
- [x] 4.3 确保 difficulty hook 保持 `target_beam`、`beam_power`、sample id、split metadata 和 auxiliary targets 不变。
- [x] 4.4 为 train-only profile、evaluation-only profile 和 no-op path 添加 focused tests。

## 5. JEPA benchmark 迁移

- [x] 5.1 保留现有 benchmark manifest suite type 和 Scenario C preset 解析，新增映射层把 suite 转为 shared difficulty profile/operator。
- [x] 5.2 将 `apply_benchmark_perturbation` 或等价 benchmark 应用路径迁移到 shared difficulty pipeline。
- [x] 5.3 保持 `metrics_by_condition.csv`、`robustness_summary.csv`、`shortcut_reliance_summary.csv` 和现有 visual analysis ingest 兼容。
- [x] 5.4 在 `benchmark_manifest.json` 写入 difficulty provenance、profile digest、operator registry name、condition/severity、seed、warnings 和 replay metadata。
- [x] 5.5 扩展 benchmark tests，验证旧 manifest 兼容、Scenario C no-future-leak、benchmark/evaluation 同 profile 同 seed 输出一致。

## 6. 文档、示例与验证

- [x] 6.1 新增或更新示例配置，覆盖 clean baseline、GPS mild async training、GPS severe async evaluation、GPS/image dropout training 和 image hard degradation sweep。
- [x] 6.2 更新 README 或相关 docs，说明 difficulty profile 不是新模态，如何新增 operator，以及输出 metadata/产物边界。
- [x] 6.3 运行 `openspec validate refactor-modality-difficulty-pipelines --strict` 并修复所有 OpenSpec 问题。
- [x] 6.4 运行 focused tests：`conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。
- [x] 6.5 运行 CLI smoke：`conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`、`conda run -n kd_mm_beam python scripts/train.py --help`、`conda run -n kd_mm_beam python scripts/evaluate.py --help`。
