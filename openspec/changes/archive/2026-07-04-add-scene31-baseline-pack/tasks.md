## 1. 现有管线定位

- [x] 1.1 阅读现有 Scene31 runner、fresh eval、missing bucket summary、model registry 和 baseline 配置路径，确认最小复用点。
- [x] 1.2 确认 `proto_sampler_uniform_es40` 旧 reference 与可用 proto natural checkpoint 的读取位置，不覆盖既有输出 root。

## 2. Random Modality Dropout

- [x] 2.1 在训练 batch 难度/缺失模态路径中实现 `random_modality_dropout` 配置，支持 `bernoulli` 与 `random_nonempty_subset`。
- [x] 2.2 保证 dropout 只作用于输入模态和 mask metadata，不修改 target、split、sample id 或 label。
- [x] 2.3 为每 epoch 输出 `random_dropout_pattern_stats.csv`，包含 epoch、available set、样本数、比例和 missing_count。
- [x] 2.4 增加 focused tests 覆盖 ensure-at-least-one、subset 覆盖 miss1/miss2/miss3、与 uniform sampler 的实现区分。

## 3. 轻量 Baseline 模型

- [x] 3.1 实现或补齐 AMR-lite imputation + mask-aware gate，并输出 `amr_lite_gate_stats.csv`。
- [x] 3.2 实现或补齐 AMBER-lite baseline-pack 最小 transformer fusion 配置，支持 natural/randomdrop 与 uniform 训练策略。
- [x] 3.3 实现 FeatureMod-lite missing-modalities adapter，或在 runner/summary 中以 skipped/quick_screen 明确标记。
- [x] 3.4 增加 synthetic forward、metadata、参数量统计和防泄漏 focused tests。

## 4. 配置与 Runner

- [x] 4.1 新增 baseline pack run 配置或生成逻辑，覆盖 proto natural、randomdrop、AMR-lite、AMBER-lite 和 FeatureMod-lite group。
- [x] 4.2 新增 `scripts/run_scene31_baseline_pack.sh`，支持 group、gpus、train-only、eval-only、auto-eval、overwrite、overwrite-eval 和默认跳过。
- [x] 4.3 runner 必须通过 `conda run -n kd_mm_beam kd-sensing-train` 和现有 fresh eval helper 执行，不复制 DataLoader、模型加载或指标计算。
- [x] 4.4 为 runner 增加 dry-run 或 shell/static focused tests，确认 group 展开、输出 root 和 skip/failed list 行为。

## 5. Summary 与结论

- [x] 5.1 新增 `scripts/summarize_scene31_baseline_pack.py`，读取本轮 root、uniform reference root 和可选 proto baseline 旧结果。
- [x] 5.2 输出 per-run、method mean/std、delta vs uniform、backbone training comparison、rank markdown、params comparison 和 `baseline_conclusion.txt`。
- [x] 5.3 summary 过滤 missing_config、missing_checkpoint 和 failed run，保守标记 quick screen 与不可比项。
- [x] 5.4 增加 summary fixture tests，覆盖字段映射、排序、status 过滤、参数量和结论逻辑。

## 6. 验证与运行

- [x] 6.1 运行 `openspec validate add-scene31-baseline-pack --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest` 的相关 focused tests：配置加载、模型 forward、summary、runner dry-run 和架构边界。
- [x] 6.3 视可用 GPU 时间运行 baseline pack 对应 group；若无法完成长训，记录已实现命令、未运行原因和剩余风险。当前未启动 24 个 40 epoch GPU 长训；代码、runner、eval 与 summary 已验证，正式运行命令记录在 6.4。
- [x] 6.4 运行或准备正式命令：`bash scripts/run_scene31_baseline_pack.sh --group all_core --gpus <ids> --auto-eval` 与 baseline pack summary 命令。
