## Why

F1 feature concat MLP 已是当前综合最强的缺失模态基线，但冻结 token 适配未能显著改善困难子集，尚不能区分 encoder under-learning、共享子集优化冲突和输入信息上限。需要一次 single-seed、inner-only、claim-ineligible 的 encoder-side 受控实验，在严格保持 Full 路径的同时比较信息上限 probe、可用模态证据保留、嵌套拓扑单调性和按缺失严重度调整的一致性。

## What Changes

- 新增六个固定模态子集 specialist probe，从同一上游 encoder 与 F1 checkpoint 初始化，并仅按对应 inner-validation task loss 选择 checkpoint。
- 新增 F1 pre-prototype encoder-tail 低秩 residual、确定性 availability regime 和 missing-only fusion residual；Full 在新增模块前物理 bypass，基础 F1 与 prototype bank 在 Stage B 全部冻结。
- 新增 14-pattern 两级均衡 schedule、全部合法嵌套 pair、固定 beam sector manifest，以及 AER、NTM、SCFC 三个可组合 loss。
- 新增 V0--V5 公平训练、统一 pattern/monotonicity/representation/weather/sector/error-distance/efficiency 评测、success gates 和唯一推荐方向。
- 新增 GPU0--5 的 Stage A、Stage B 独立编排；任务失败互不终止，不自动调参、重跑、outer test、multi-seed 或下一轮实验。
- 新增 reproducibility repair：保留旧目录并审计 V0--V5 的真实执行状态，以 Availability Fallback U0 直接引用的 F1 为唯一 canonical 身份，锁定 split/sample/mask/metric hash，双次复现后才允许正式训练。
- 重新用独立加载的 canonical F1 评测六个 specialist，避免训练后的 fusion 对象污染 baseline；V1--V5 按同一预注册 encoder 尾部与 F1 fusion scope 端到端训练，并统一使用 Full/single/double/triple 四组等权 validation loss 选择 checkpoint。
- 不新增公共 CLI、canonical recipe、动态 Router、模态权重、attention/MoE、重建或 channel/path/power/历史 beam 输入。

## Capabilities

### New Capabilities

- `beam-topology-subset-consistency-learning`: 规定 F1/encoder 身份、specialist probe、Full bypass、14-pattern schedule、AER/NTM/SCFC、公平训练评测、GPU 编排和 inner-only 停止边界。

### Modified Capabilities

无。

## Impact

- 新增独立的 BT-SCL 模型/loss 组件、训练评测 analysis 入口、GPU 编排脚本和 focused tests。
- 原始本地产物保留在 ignored 的 `outputs/bt_subset_consistency/`，repair 产物写入 `outputs/bt_scl_repair/`；二者均不纳入源码，也不成为 canonical config 或 package import 的依赖。
- 复用 F1 validation-best checkpoint、上游 encoder checkpoint、既有 train/validation/eval 身份、`BeamPrototypeBank` 和 64-beam topology；不引入新依赖，也不修改 current T2/baseline 公共路径。
