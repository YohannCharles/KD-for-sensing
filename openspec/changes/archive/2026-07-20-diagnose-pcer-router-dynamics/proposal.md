## Why

现有 PCER 快速验证只能说明端到端指标变化，无法区分样本级动态路由、缺失模式条件先验和静态融合，也不足以判断反事实 target 的实现、可学习性及 S3 整模态缺失崩溃来源。需要一套只读、固定身份、可复现的离线诊断流程，在不重训或覆盖历史产物的前提下给出可审计证据。

## What Changes

- 新增统一 PCER 离线诊断入口，自动定位 A0-A3 的历史最佳 checkpoint 与 resolved config，并冻结模型参数运行。
- 对 A1/A3 实施 D0-D4 权重替换、router 动态性、counterfactual target、梯度和 S3 分模态诊断。
- 新增 counterfactual target 的独立 synthetic sanity tests，覆盖符号、屏蔽和 `[M,T]` 展平顺序。
- 将所有 CSV、JSON、文本与结论写入忽略目录 `outputs/quick_pcer_diagnostics/`，不修改历史结果或正式 claim 文档。

## Capabilities

### New Capabilities

- `pcer-router-diagnostics`: 定义 PCER 历史 checkpoint 的只读权重替换、target 可学习性、梯度及 S3 崩溃诊断契约。

### Modified Capabilities

无。

## Impact

新增诊断脚本、shell 入口、单元测试和本地忽略产物；复用现有 MMW dataloader、固定 mask、模型 forward、checkpoint 与评测指标。模型、训练入口、配置 schema、历史 checkpoint 和正式 evidence registry 均不改变，也不引入新依赖。
