## Why

当前 Scene32 MARF 结果里，`all` 与 `gps+mmwave` 强模态路径的关系还不能解释弱模态到底是无信息、未被 MARF 学会利用，还是只在特定通信状态或 horizon 下有条件性价值。继续调整 router、loss 或通信特征前，需要先用同一 checkpoint 做系统化 subset evaluation、逐样本 delta 和 teacher complementarity 审计，避免把后续 Phase 2 建在平均指标的误判上。

## What Changes

- 新增 Conditional Utility Audit 分析能力，用一个已训练 MARF checkpoint 独立输出 `all`、`strong_only`、`strong_plus_image`、`strong_plus_radar`、`strong_plus_lidar`、`single_best_mmwave`、`weak_only` 的 Top1、Top3、Top5、DBA 和 CE 指标。
- 新增统一 subset registry，复用 `src/kd_sensing/modalities.py` 的固定模态顺序，避免 validator、脚本和报告各自硬编码 subset。
- 新增逐样本、逐 horizon、逐 subset prediction dump，用于计算 `strong_only` 对比 `strong_plus_<weak>` 的边际 CE、Top1、Top3 和 DBA delta。
- 新增 teacher complementarity 审计，从 `outputs/scene32/teacher_registry.json` 严格加载单模态 teacher，统计 strong path 错误时弱模态 teacher 的 rescue 能力和 ground-truth probability advantage。
- 新增通信状态 bucket 特征和分桶报告，覆盖 mmWave uncertainty、GPS 相对运动、beam transition，以及按 horizon 的条件性增益。
- 新增 subset oracle 与 teacher complementarity oracle，并在 summary 中输出面向 Phase 2 的初步 diagnosis，不自动修改 MARF 结构、router、loss、encoder 冻结策略或训练流程。
- 新增分析配置和命令入口，默认针对 `configs/fusion/scene32_marf.yaml` 与 `outputs/scene32/scene32_marf/checkpoints/best_top1.pt` 生成可复现实验产物。

## Capabilities

### New Capabilities

- `conditional-utility-audit`: 定义 MARF Phase 1 审计的 subset registry、逐样本输出、teacher complementarity、oracle、通信状态 bucket、summary diagnosis、图表和验收标准。

### Modified Capabilities

- 无。

## Impact

- 影响评估与诊断代码：新增 `src/kd_sensing/evaluation/subset_specs.py`、`src/kd_sensing/diagnostics/conditional_utility.py`、`src/kd_sensing/diagnostics/communication_state_features.py`，并让 validator 或分析入口复用统一 subset 定义。
- 影响脚本与配置：新增 `tools/analysis/run_conditional_utility_audit.py`、`tools/analysis/analyze_conditional_utility.py` 和 `configs/analysis/scene32_marf_conditional_utility_audit.yaml`。
- 影响数据输出：新增 `outputs/scene32/<run_name>/conditional_utility/` 下的 prediction dump、delta 表、bucket CSV、oracle summary、总 summary 和 figures。
- 可能需要对 dataset 做最小 metadata 扩展，返回稳定 `sample_id`、`dataset_index`、可用时的 `seq_id`、`frame_idx` 或 beam 路径派生信息；该扩展不得改变训练主流程语义。
- 测试覆盖新增 subset 定义、逐样本指标、oracle、teacher rescue、bucket feature 和 end-to-end dummy audit。项目相关测试命令使用 `conda run -n kd_mm_beam`。
