## Why

当前仓库同时维护 final C2、DeepSense6G、CSI、物理 MMW、历史诊断和多条已退役路线，实际 T2 MMW 主实验却依赖一条更小的代码闭包。继续保留这些并行 surface 会掩盖 T2 的真实依赖，并让已删除的蒸馏、兼容和实验路线继续消耗维护成本。

## What Changes

- **BREAKING** 将唯一 current 研究 surface 收敛为 MMW `T2` 及可比较的 `S1`、`AMBER-Full`、`RMBP-MM` baseline，保留其训练、评估、预处理、固定掩码和多 seed 证据链。
- **BREAKING** 将 T2 的 resolved recipe 从 ignored output 提升为 tracked canonical config；launcher、hyperparameter screen 和 baseline matrix 不再以 `outputs/` 中的历史配置作为输入。
- **BREAKING** 删除不属于上述闭包的模型、损失、数据流程、CLI、脚本、实体 YAML、测试与兼容/migration facade，包括 DeepSense6G/final-C2/CSI/physics/历史诊断路线和外部 teacher-guidance、full-to-partial KD 等未启用蒸馏支线。
- 保留 T2 同一 primary model 的在线 no-grad full/superset consistency、BPA、prototype/router 与 CMA ablation；它们不是独立 teacher-student runtime。
- 用一份集中历史说明和 OpenSpec archive 保留退役路线的目的与存在证据，不保留 stub、alias、迁移配置或拒绝兼容层。
- 缩减 public CLI、registry、配置加载和架构测试，使它们只声明并验证 T2/baseline current surface。
- **BREAKING** 停用 GitHub Actions，删除 GitHub、Cursor 和 Kiro 专属协作适配；验证与协作提示只保留本地 Codex 文档和命令。

## Capabilities

### New Capabilities
- `t2-baseline-surface`: 定义唯一 T2、S1、AMBER-Full、RMBP-MM 研究闭包、canonical recipe 和退役边界。

### Modified Capabilities
- `project-architecture`: 将 current 包边界收敛到 T2/baseline 最小运行链。
- `project-entrypoint-lifecycle`: 将 console script 与 local/manual script surface 收敛到 T2/baseline workflow。
- `canonical-config-resolution`: 要求 T2/baseline 使用 tracked recipe，不再支持旧配置迁移或 compatibility guard。
- `distillation-free-project-surface`: 删除未使用的外部 teacher-guidance 和 full-to-partial KD，同时保留 T2 same-model consistency。
- `training-evaluation-runtime`: 删除非 T2/baseline 训练扩展与运行时分支。
- `u-mask-beam-jepa`: 将 retained U-Mask contract 明确限定为 T2 所需机制与 active BPA/CMA ablation。
- `mmw-baseline-multiseed-robustness-evidence`: 将 T2/baseline matrix 绑定到 tracked canonical recipes。
- `retired-route-summary`: 将退役范围扩展为所有非 T2/baseline source surface，并将历史信息集中到说明与 archive。
- `project-health-guardrails`: 以 T2/baseline surface 验证取代历史路线的兼容 guard，并移除 GitHub Actions CI 要求。

## Impact

影响 `src/kd_sensing/`、`configs/`、`scripts/`、`tests/`、`pyproject.toml`、`.github/`、`.cursor/`、`.kiro/`、`envs/smoke-dev.yml`、README、研究/claim 文档和 current OpenSpec specs。现有 active T2 调参及 BPA/CMA change 的输入、实现和任务保持可达；其余 active runtime change 不会被作为本变更的兼容目标。
