## Why

BCACL U2 的 inner/development 结果已经证明 private/shared Beam 监督能改善弱模态和缺失组合，但仓库同时保留了 PGCD、PCER、动态 Router、PR-SQDF、residual recovery、feature/prototype fusion、availability fallback 和 BT-SCL 等已证伪或未启动路线。外围实验代码已超过核心包规模，current specs、脚本白名单和 canonical 配置也仍在保护历史分支。

CMSBL 是当前唯一主线：在 BCACL U2 上补足模态自身未学习容量，并有限强化困难缺失组合。实现前必须先退役前序路线，使模型、loss、配置、脚本、测试和文档重新形成一个可审计闭包。

## What Changes

- 将 CMSBL 收敛为三个训练期机制：单一 linear decay 辅助调度、固定 standalone Top-1 容量参考、15 个 canonical availability mask 的有界 loss 重加权。
- 只保留 V0 U2、V1 调度、V2 容量、V3 mask、V4 三者组合；不实现 residual distillation、额外 schedule/capacity 模式、sampling reweighting 或 V5/V6。
- 将 BCACL 收缩为 U2 所需的 modality projection、private head、shared head、observed/fusion mask 和逐模态统计；删除 relation prototype、teacher、quality matrix 和 detached two-stage。
- 删除已证伪或未启动研究路线的源码、配置、analysis、脚本和测试，并将其结论保留在 OpenSpec archive、`docs/retired_routes.md` 和既有本地产物中。
- 将 public workflow 保持为 train/evaluate/preprocess，保留 MMW all-weather、BPA/CMA 和结果汇总所需的最小本地 helper。
- 不删除、移动或改写 `outputs/`、`outputs/cache/`、dataset、日志或 checkpoint。

## Capabilities

### New Capabilities

- `capacity-aware-modality-sufficient-beam-learning`: 规定 CMSBL 的 U2 边界、线性课程、固定容量缺口、困难 mask loss、状态恢复和无泄漏要求。

### Modified Capabilities

- `project-architecture`: 将包导入和源码闭包收敛到 T2/baseline、BCACL U2 与 CMSBL。
- `t2-baseline-surface`: 将 CMSBL 设为唯一 active 研究扩展，并退役失败/停止的实验路线。
- `u-mask-beam-jepa`: 删除 PCER、PGCD、候选动态 Router 与 BCACL U3--U5，只保留 current T2 和 U2/CMSBL 所需 payload。
- `training-evaluation-runtime`: 保存 CMSBL 训练状态并保持 15-pattern validation；不增加独立 trainer。
- `project-entrypoint-lifecycle`: 缩减历史脚本白名单，不增加 CMSBL console script 或 runner。

## Impact

- 大量删除 `analysis/`、`scripts/`、route-specific `src/`、tests、旧 YAML 和失效 current specs。
- 修改 U-MaskBeamJEPA model/loss/config、BCACL U2、Beam prototype per-sample loss、canonical T2 配置和维护文档。
- 不新增依赖，不读取或修改本地 cache，不自动运行 outer test、多 seed 或正式 claim。
