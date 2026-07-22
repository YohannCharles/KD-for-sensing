## Why

当前 C0-static 在冻结的四模态表示上采用 late logit 加权求和，尚不能判断性能瓶颈来自融合算子、beam-specific query 设计，还是非 LiDAR 模态证据不足。需要一次 single-seed、inner-only、claim-ineligible 的冻结特征筛选，在不重复训练 backbone 的前提下公平比较六种融合方向。

## What Changes

- 新增冻结 C0 checkpoint 的分片特征缓存与一致性 gate；缓存只包含四模态五时间步特征、64-beam logits、availability 和必要的非输入元数据。
- 新增 F0 static logit sum、F1 concat MLP、F2 fusion-token Transformer、F3 learned 64-query、F4 prototype 64-query、F5 prototype-query balanced-evidence 六个本地实验方向。
- 新增统一的 inner-train/inner-validation 训练、validation selection、固定 missing/S3 评测、attention beam-specificity、模态 shuffle、天气/sector、LiDAR 依赖和效率汇总。
- 新增 GPU0--5 独立编排脚本；单任务失败不终止其他任务，完成后不自动启动 outer test、multi-seed 或端到端训练。
- 不新增公共 CLI、canonical recipe、动态 Router、quality/reconstruction branch，也不修改冻结 encoder 和 prototype bank。

## Capabilities

### New Capabilities

- `feature-fusion-quick-search`: 规定冻结缓存、F0--F5 公平筛选、mask 安全、inner-only 评测诊断、GPU 编排与停止边界。

### Modified Capabilities

无。

## Impact

- 新增独立的 feature-fusion 模型组件、缓存/训练评测 analysis 入口、GPU 编排脚本和 focused tests。
- 本地产物写入 `outputs/feature_fusion_quick_search/`，不纳入源码、不成为 canonical 配置或 package import 的依赖。
- 复用现有 `BeamPrototypeBank`、64-beam topology、C0 resolved config/checkpoint、四模态数据与固定 mask 语义，不引入新依赖。
