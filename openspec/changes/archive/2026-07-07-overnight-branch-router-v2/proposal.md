## Why

上一轮 `pcpg_radar_balance_v1` 与 `bprr_reliability_router_v1` 已经把问题拆成 branch weakness 与 routing failure。现在需要一次性补齐更温和的 hard subset、受监督 pattern-aware router、固定 GPU1-2 overnight 矩阵和可复盘 summary，让下一轮结果能直接判断 e5/e6/soft-hard/router 的去留。

## What Changes

- 新增 `soft_static` hard subset weighting，并在 run config 与 summary 中记录权重模式。
- 新增显式 opt-in 的 `supervised_router` fusion 路径，支持 oracle/pattern-best/none router supervision、focus pattern distillation、masked gate、router diagnostics 与 oracle target distribution。
- 保持默认行为不变；所有新增训练行为都必须通过显式 flag 或 launcher 配置启用。
- 新增 `scripts/launch_overnight_branch_router_v2.py`，固定生成并运行 A/B/C 三组共 40 个 job，支持 dry-run、skip/force、GPU1-2 并发限制、manifest 与 failed jobs。
- 新增 `scripts/summarize_overnight_branch_router_v2.py`，聚合当前输出与历史 baseline roots，写出 CSV/Markdown 和 router diagnostics。
- 新增 focused tests 覆盖 hard subset、supervised router helper、launcher dry-run 与 summary parser。

## Capabilities

### New Capabilities
- `overnight-branch-router-v2`: 定义 soft hard-subset weighting、supervised pattern-aware router、overnight launcher、评估/summary artifact 与 focused test 契约。

### Modified Capabilities
- 无。

## Impact

- 影响训练 CLI 参数、PCPG/BPRR fusion 相关实现、run config metadata、评估/summary 解析、本地研究 launcher 和 focused tests。
- 不新增长期 package CLI，不复制训练框架，不修改旧 outputs，不把训练日志、checkpoint 或 outputs 产物纳入源码变更。
- 所有项目 Python 命令继续通过 `conda run -n kd_mm_beam ...` 执行。
