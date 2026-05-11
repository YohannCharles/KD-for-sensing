## Why

Phase 1 的 Scene32 Conditional Utility Audit 已经显示 `gps+mmwave` 强模态主导：`all` 略低于 `strong_only`，弱模态加入后的 DBA 增益只有约 `+0.0002 ~ +0.0003`，teacher rescue 约 `6.48%`，oracle DBA gain 也只有 `+0.00093`。在继续设计 MARF-v2、MARF-Comm、GPS 引导 image mask 或复杂跨模态交互前，需要用统计置信度、跨 checkpoint 复核和 dedicated fixed-subset 训练把“弱模态是否真的值得救”判清楚。

## What Changes

- 新增 Phase 1.5 Utility Validation 协议，明确本阶段只做验证、训练矩阵、统计分析和决策报告，不修改 MARF 主结构、router 输入、loss 目标或 encoder 冻结策略。
- 基于现有 `conditional_utility/` 逐样本产物新增 cluster bootstrap 95% CI，按 `seq_id` 可用时优先按序列重采样，否则退化为 `sample_id/dataset_index` 重采样并在报告中声明。
- 对 `strong_plus_image - strong_only`、`strong_plus_radar - strong_only`、`strong_plus_lidar - strong_only` 和 `all - strong_only` 输出 Top1、Top3、DBA、CE 的均值差、CI、p-like sign rate 和 horizon 分解。
- 扩展 Phase 1 audit 运行矩阵，至少覆盖 `scene32_marf` 的 `best_top1.pth`、`best.pth` 和 `last.pth`，并按 checkpoint 汇总 subset metrics、marginal utility、oracle gain、teacher complementarity 和 diagnosis。
- 建立 dedicated fixed-subset baseline 矩阵，训练 `gps+mmwave`、`gps+mmwave+image`、`gps+mmwave+radar`、`gps+mmwave+lidar` 和五模态 `all`，至少 3 seeds，并用相同训练预算、checkpoint 选择规则、encoder 初始化、loss 和评估协议比较。
- 生成 Phase 1.5 总报告，明确两类决策出口：若 dedicated strong+weak 仍无显著收益，则转向 strong-path 精度和 safe fusion；若特定弱模态在 bucket/horizon 上稳定显著，则进入 MARF-Comm 条件效用 router 设计。
- 收紧 diagnosis：弱模态全局 useful 不能只看正数 delta，至少需要达到配置阈值，例如 `global_delta_dba >= 0.001` 且 bootstrap CI 下界大于 0；条件性 useful 也必须同时记录 bucket 样本数和 CI。

## Capabilities

### New Capabilities

- `phase-1-5-utility-validation`: 定义 Phase 1.5 的统计显著性、跨 checkpoint audit、dedicated fixed-subset baseline、汇总报告和后续路线决策标准。

### Modified Capabilities

- `conditional-utility-audit`: 在已有逐样本 audit 产物之上新增 bootstrap CI、multi-checkpoint audit 汇总和带置信度约束的 diagnosis 阈值。

## Impact

- 影响分析代码：预计新增或扩展 `tools/analysis/*phase_1_5*`、`src/kd_sensing/diagnostics/conditional_utility.py` 中的统计汇总 helper，以及读取既有 `conditional_utility` 表的报告生成逻辑。
- 影响实验配置与运行脚本：新增 Phase 1.5 audit/checkpoint matrix 配置，复用现有 `configs/fusion/scene32_marf.yaml` 和 canonical fusion 配置/命令行覆盖运行 5 个 fixed-subset baseline 的 3-seed 矩阵。
- 影响输出产物：新增 `outputs/scene32/phase_1_5_utility_validation/` 下的 bootstrap CI 表、checkpoint comparison 表、fixed-subset baseline 汇总、decision report 和可选 figures。
- 不新增模型结构依赖，不改变普通训练、评估和 Phase 1 audit 的默认行为；所有 Phase 1.5 重计算必须由显式脚本或配置触发。
- 测试覆盖 bootstrap 统计、cluster fallback、checkpoint matrix 汇总、baseline run manifest 和 decision rule，项目相关命令使用 `conda run -n kd_mm_beam`。
