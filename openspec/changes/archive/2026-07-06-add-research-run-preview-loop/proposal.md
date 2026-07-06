## Why

当前仓库验证和研究产物治理很强，但缺少一个面向 agentic/vibe-coding 的“从当前研究问题到可见证据”的低摩擦闭环：一键运行/汇总、预算声明、静态预览 QA 和环境 recipe 仍分散在 README、scripts、outputs 和人工经验中。需要把研究运行闭环规格化，让 agent 能安全地生成可检查证据，而不是直接跳到长训练或手动拼表。

## What Changes

- 新增 research run preview loop 能力，定义当前主线/诊断 workflow 的 happy path：默认只做无副作用检查、汇总或 dry-run；真实训练必须显式 opt-in。
- 定义静态预览/证据 QA：对 HTML、CSV、figure data、paper table、checklist 和 conclusion draft 做结构化检查，发现空图、缺字段、candidate caveat 丢失、远程依赖和 HTML escaping 问题。
- 定义实验预算 manifest：长跑前记录 GPU、预计时长、数据读取、输出 root、checkpoint 写入、fresh eval、清理策略和停止条件。
- 定义环境/run recipe 复现要求：console script 不可用时提供 `python -m` fallback；记录 smoke/dev 与 GPU/full training 的边界。
- 保持真实数据、outputs、logs、cache、checkpoint 和 TensorBoard 产物在 ignored 本地产物边界内。

## Capabilities

### New Capabilities
- `research-run-preview-loop`: 定义研究 workflow 的一键汇总/预览、静态证据 QA、实验预算和环境 recipe 契约。

### Modified Capabilities

无。该能力会引用现有 research dashboard、mainline documentation、project health guardrails 和 experiment workflow，但本 change 不直接修改既有 requirement。

## Impact

- 可能新增 package CLI、薄脚本或 Makefile target，用于汇总当前研究证据、生成 checklist、运行预览 QA 和输出 budget manifest。
- 可能新增静态 HTML/CSV/figure/table validator 和 focused tests。
- 可能更新 README、`docs/experiment_matrix.md`、`docs/training_throughput.md`、inventory、agent navigation 和 relevant diagnostics docs。
- 不新增默认真实训练入口，不改变模型、dataset split、metric schema、checkpoint schema 或默认输出分区。
