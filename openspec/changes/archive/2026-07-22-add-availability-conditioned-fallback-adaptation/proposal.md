## Why

F1 feature concat MLP 已是当前冻结特征综合最强基线，但其 Missing LiDAR 与多模态组合缺失表现仍明显落后于 Full，尚不能判断瓶颈来自可用 token 未按当前模态集合调整，还是冻结表示本身的信息上限。需要一次 single-seed、inner-only、claim-ineligible 的受控实验，在严格保持 Full 路径的前提下覆盖全部 14 种非空缺失组合。

## What Changes

- 新增 F1 token cache 的身份、schema、Full/四种单缺失 parity gate 与六分片预计算流程。
- 新增 U0 frozen F1、U1 组合专属 SSF、U2 共享 availability-conditioned SSF、U3 contextual residual、U4 prototype auxiliary、U5 unimodal teacher 六个方向，并冻结 F1 fusion、prototype bank 与 encoder。
- 新增 14-pattern 两级均衡 schedule、group-balanced validation selection、四个单模态 teacher、统一训练评测及 mean/zero/shuffle、表示、weather、sector、error-distance 与效率诊断。
- 新增 GPU0--5 独立编排和完整汇总；单任务失败不影响其他任务，不自动调参、重跑、outer test、multi-seed、encoder 解冻或下一轮训练。
- 不新增公共 CLI、canonical recipe、伪缺失 token、动态 Router、Transformer、MoE、重建/残差恢复或 channel/path/power/历史 beam 输入。

## Capabilities

### New Capabilities

- `availability-conditioned-fallback-adaptation`: 规定冻结 F1 token cache、14 种缺失组合、U0--U5 适配/教师协议、mask 与 Full bypass 安全、统一诊断、GPU 编排和 inner-only 停止边界。

### Modified Capabilities

无。

## Impact

- 新增独立的 AC-CFA 模型组件、cache/训练/评测 analysis 入口、GPU 编排脚本和 focused tests。
- 本地产物写入 ignored 的 `outputs/availability_fallback_search/`，不纳入源码，不成为 canonical config 或 package import 的依赖。
- 复用 F1 validation-best checkpoint、token cache、冻结 fusion/prototype/topology、固定 split 与 mask 语义，不引入新依赖，也不改动当前 T2/baseline 公共路径。
