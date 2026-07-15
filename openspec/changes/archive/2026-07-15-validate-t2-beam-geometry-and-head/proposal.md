## Why

T2 在 seed1 固定时序掩码筛选中相对 S1 的五档 Top1 均为正，但当前训练沿用了未经物理核验的 circular beam 标签假设和 prototype head。进度汇报红字要求确认 beam 应为线性还是环形；本地 18,479 个有效 DeepSense6G Scene31-34 beam sweep 又显示相邻码字与首尾码字的响应关系明显不同，因此需要在补齐 T2 多随机种子证据的同时，受控验证几何和 head 假设。

本 change 接续仍 active 的 `improve-s1-lightweight-temporal-robustness`：复用其 seed1 筛选与 T2 实现，并取代其 tasks 7.2-7.3 的多 seed 执行和汇总；旧 change 不再单独启动 J2、T1 或 T1+T2 多 seed。

## What Changes

- 保留 S1/T2 当前 circular-prototype 口径作为历史可比对照，并补齐 seeds2/3；结果明确标注 circular metric caveat。
- 在现有 H5/P1 参数化 launcher 中增加四个 local/manual 方法：S1/T2 的 linear-Gaussian prototype，以及 S1/T2 的 classifier/prototype-loss-off；不新增模型注册名、实体 YAML 或独立训练循环。
- 第一轮固定使用 GPU0-7 一卡一进程：四个 current 多 seed job 与四个新候选 seed1 screening job 并行；只有通过 Drop0 guardrail 且改善主要时序指标的候选才补 seeds2/3。
- 所有方法复用相同 Scene31-34 split、history=5、prediction=1、temporal sampler、固定 mask cache、checkpoint 策略和评估脚本，并分别记录 beam geometry、head、prototype 和 metric profile。
- 输出三随机种子 mean/std、同 mask paired delta、Top1/Top3/ADBA/MAE 与 T2 gate diagnostics；结果保持 local experimental，不自动升级正式 claim。
- 新增中文 Markdown，逐页落实 PDF 红字替换建议，说明 T2 当前状态、beam geometry 审计证据、PPT 叙事调整、实验门禁和最终结果。

## Capabilities

### New Capabilities

- `t2-beam-geometry-head-validation`: 定义 T2 在 circular/linear beam 标签与 prototype/classifier head 下的受控筛选、多 seed 晋级、固定 mask 评估和汇报证据契约。

### Modified Capabilities

- `temporal-window-missing`: 扩展现有 local/manual H5/P1 workflow 的可选方法表和阶段式 GPU0-7 调度，不改变默认五方法或 S1 lightweight 默认八方法。

## Impact

- 主要影响 H5/P1 launcher/eval、两条 supervised-router oracle loss 路径及其 focused tests、本 change 的 OpenSpec artifacts 和一份 `docs/` 中文报告。
- 不改变 U-Mask 默认 forward、checkpoint schema、模型注册表、公共 package CLI 或 canonical config；只通过已有配置字段生成 ignored local run config。
- 训练、checkpoint、日志、固定 mask cache、评估和 summary 全部写入 ignored `outputs/`，不纳入源码变更。
