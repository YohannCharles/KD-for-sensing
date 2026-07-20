## Why

现有 PCER 诊断已排除符号、索引、KL 和梯度 bug，并显示 equal-coalition target 近似均匀且与部署路由效用失配。需要在相同 MMW inner split、mask、seed 和 16 epoch 预算下并行筛选八个机制方向，判断后续应保留简单直接监督、结构化路由还是转向缺失 evidence 学习。

## What Changes

- 在现有 U-Mask/PCER opt-in 边界内增加 evidence-only、beam-only flat block、hierarchical modality-time 和 mask-prior residual 路由，不新增模型注册名或训练循环。
- 增加 standalone-quality、on-policy block、on-policy modality-group 三类可 detach route target，以及 balanced leave-one-modality-out distillation/evidence auxiliary loss。
- 生成 B0-B7 八组 claim-ineligible resolved config，保持共同 backbone/prototype 初始化、数据 split、mask curriculum、optimizer、scheduler、batch 和 checkpoint 选择规则。
- 新增真实单 batch preflight、一次自动量级审计、GPU0-7 fail-independent launcher、固定 S0-S5 evaluator、机制诊断、成本统计和 Pareto 汇总。
- 直接复用 A0-A3 历史指标，不重跑历史基线，不启动 multi-seed、outer evidence 或完整缺失组合矩阵。

## Capabilities

### New Capabilities

- `pcer-direction-search`: 定义八方向单 seed 快速筛选、GPU 映射、preflight、固定评测、机制诊断和方向选择边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 扩展 PCER opt-in router/fusion/target/evidence-learning 语义，并保持 canonical 默认路径和历史 checkpoint 兼容。
- `training-evaluation-runtime`: 增加方向筛选所需的同模型 full/masked/LOMO 训练扩展、成本观测和固定评测流程。

## Impact

影响 `src/kd_sensing/models/pcer_temporal_fusion.py`、`u_mask_beam_jepa.py`、对应 loss/config、训练 extension、方向筛选脚本和聚焦测试。生成配置、日志、checkpoint 与统计仅写入 ignored `outputs/pcer_direction_search/`；canonical recipe、数据集、beam prototype、历史 A0-A3 产物和正式 claim 均不修改，不新增第三方依赖。
