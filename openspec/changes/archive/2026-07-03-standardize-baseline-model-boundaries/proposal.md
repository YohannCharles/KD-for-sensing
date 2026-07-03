## Why

当前代码和文档里 “baseline” 同时指普通可训练模型、整模型例外、本地论文风格对照和外部复现 workflow，容易让后续维护者误以为应该按名称把所有 baseline 机械搬到 `src/kd_sensing/baselines/`。这会破坏现有 registry、通用训练 runtime 和 workflow/package CLI 的边界。

本变更把边界固定为“模型能力放 `models/`，复现/审计/多阶段 workflow 放 `baselines/`”，并用文档和架构测试防止后续漂移。

## What Changes

- 明确 `src/kd_sensing/models/`、`src/kd_sensing/baselines/`、`configs/fusion/` 和 `configs/baselines/` 的放置规则。
- 修正 baseline package marker 的描述，避免把 `baselines/` 误写成 “non-neural baseline” 容器。
- 在维护文档和模型目录中加入短规则表，说明普通 baseline、component baseline、whole-model exception 和 workflow/paper reproduction 的归属。
- 增加架构边界测试，拒绝在 `src/kd_sensing/baselines/` 新增 registry 注册，拒绝 `models/` 反向依赖 `baselines/`。
- 不移动当前已有模型文件；当前 `MODELS` 注册名和 CLI 入口保持兼容。
- **BREAKING**：无。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `model-architecture-extension-contract`：补充 baseline/model 源码与配置放置规则，明确 workflow baseline 与通用可训练模型的边界。
- `project-health-guardrails`：新增架构边界 guardrail，阻止 registry 模型注册进入 `baselines/` 或 `models/` 反向依赖 workflow package。

## Impact

- 影响文档：`docs/model_architecture_inventory.md`、`docs/project_surface_inventory.md`、`docs/extension_guide.md`。
- 影响源码轻量 marker：`src/kd_sensing/baselines/__init__.py`。
- 影响测试：`tests/test_architecture_boundaries.py`。
- 不新增依赖，不改变训练入口、配置解析、registry build 或 CLI 行为。
