## Why

上一轮冻结特征筛选表明，F4 prototype query 虽改善 Missing LiDAR，却因 prototype 仅初始化 query、最终 score 由共享标量头生成而损害 Within-3、MAE 与 Missing Image。需要一次受控的 inner-only 搜索，验证将同一 prototype 用于 query 和最终 score 锚定能否修复该拓扑失真，并检验其与 F1 全局融合的互补性。

## What Changes

- 新增 G0--G5 topology-anchored prototype-query quick search：G0/G1 仅复现既有 F1/F4 checkpoint，G2 使用 prototype-compatible anchored score，G3 增加 query Gram 保持，G4/G5 以冻结 F1 作全局锚点并训练受限 local query 修正。
- 新增统一的缓存/checkpoint 身份 gate、anchored score、Gram loss、固定 mask-group 均衡损失、Full preserve loss、topology/模态依赖/global-local 替换诊断与 6 GPU 本地编排。
- 保持冻结四模态 encoder、64-beam prototype bank、既有 feature-fusion cache 和 inner train/validation split；所有结果固定为 single-seed、claim-ineligible，且不会自动启动 outer test、multi-seed 或端到端训练。

## Capabilities

### New Capabilities

- `topology-anchored-prototype-query-search`: 规定以冻结 feature-fusion cache 和 F1/F4 identity 为输入的 G0--G5 原型查询融合筛选、拓扑锚定、global-local 诊断及停止边界。

### Modified Capabilities

无。

## Impact

- 修改 `src/kd_sensing/models/feature_fusion_search.py` 的本地实验模型，并新增受限训练/评测入口与 GPU0--5 编排脚本。
- 增加 focused tests；不增加公共 CLI、canonical YAML、registry 项、第三方依赖或正式 claim。
- 运行产物仅写入 `outputs/topology_anchored_query_search/`，不纳入源码。
