## Why

P1 对应中等风险的重复诊断入口和薄 wrapper。它们不像 P0 那样纯粹是一次性报告脚本，但当前代码中存在多个“单独 CLI/脚本只包装同一个诊断 owner”的形态：predictive GPS query visualizations、MMW Town GPS v2 plot/compare、training IO profile/recommendation、以及少量 `scripts/mmw/` 数据准备 wrapper。

这些入口让用户看起来有更多功能，实际增加的是文档、help、inventory、测试和回归维护面。更懒也更稳的做法是：保留真实诊断 owner，把 plot、compare、visualize、recommend 等模式作为同一个 package CLI 的子命令或显式 flag。

## What Changes

- predictive GPS query explanatory visualizations 合并进 predictive JEPA robustness diagnostics bundle；删除独立可视化 CLI 或降为 bundle 的 mode。
- MMW Town GPS v2 runner、plotter 和 comparator 收敛到单一 package CLI 或单一 owner module；旧 plot/compare 薄 CLI 不再作为 current entrypoint。
- training IO profiling 与 parallel recommendation 收敛到一个 profiling owner；`scripts/profile_training_io.py` 和 `scripts/recommend_parallel_training.py` 只在无法折入 package CLI 时保留 retained-with-reason。
- `scripts/mmw/prepare_town10_skybridge.py`、`scripts/mmw/build_sequence_splits_from_manifest.py` 等薄 wrapper 回到 package preprocess/data owner 或 documented command recipe。
- 更新 CLI help tests、project surface inventory、docs 和 OpenSpec current specs，确保用户仍能找到同等行为。

## Capabilities

### New Capabilities

- 无。本 change 合并现有诊断/包装入口，不新增研究能力。

### Modified Capabilities

- `project-entrypoint-lifecycle`：增加薄诊断 wrapper 删除规则和 consolidated package CLI 入口规则。
- `predictive-jepa-robustness`：将 GPS query explanatory visualizations 归入 diagnostics bundle，而不是独立 CLI。
- `mmw-town-gps-adapter-v2`：允许 runner、plot、compare 由同一 owner CLI 覆盖，旧薄入口可删除。
- `training-throughput-optimization`：允许 profiling 和 parallel recommendation 由统一 profiling owner 覆盖。

## Impact

- 影响范围：predictive GPS query visualization CLI/module、MMW Town GPS v2 plot/compare CLI、training profiling/recommendation scripts、`scripts/mmw/` 薄 wrapper、CLI help tests、docs/inventory 和 surface guardrail。
- 不影响范围：预测 JEPA claim 口径、MMW Town GPS v2 adapter 训练/评估语义、profiling 指标含义、数据 split 语义、本地运行产物。
- 兼容性：旧薄入口路径不再承诺可用；替代路径是 consolidated package CLI、subcommand、mode flag 或 documented canonical command。
- 验证：OpenSpec strict validate、CLI help tests、predictive diagnostics focused tests、MMW Town GPS v2 smoke tests、training throughput focused tests 和 scripts surface doctor。
