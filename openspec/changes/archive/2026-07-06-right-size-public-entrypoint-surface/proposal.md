## Why

当前 `pyproject.toml` 暴露的 `kd-sensing-*` console scripts 已经覆盖训练、评估、诊断、paper export、dataset audit、baseline reproduction 和若干研究 helper。缺失模态主线表面收口后，如果 package CLI 不同步分层，入口噪声会从 `scripts/` 转移到 public console scripts，导致协作者无法判断哪些命令是长期 API、哪些只是 local/manual 或 paper helper。

本 change 目标是右尺寸化 public entrypoint surface：保留核心和当前诊断入口，删除或降级不应长期暴露的 console scripts，补齐 help smoke 和 inventory 分类，避免实验性 helper 偷偷变成 public API。

## What Changes

- 建立 package console script 生命周期分类，至少区分 core workflow、current diagnostic、paper/export helper、baseline reproduction、local/manual、internal-only 和 delete/defer。
- 审计 `pyproject.toml` 中所有 `kd-sensing-*` entry points，与 `tests/test_cli_help.py`、README/docs、OpenSpec specs 和 `docs/project_surface_inventory.md` 对齐。
- **BREAKING**：对无 current 文档/spec/测试契约、只服务本地研究或已有更清晰 package/owner 入口覆盖的 console script，删除 public entry point 或降级为 internal module-only helper；不提供旧命令 wrapper。
- 补齐仍保留 public CLI 的 help smoke、owner module、输出边界和推荐入口说明。
- 保持 CLI glue thin：`src/kd_sensing/cli/*.py` 只负责参数解析、轻量 IO、调用 owner module 和 exit code，不复制训练、评估、诊断聚合或报告生成主逻辑。
- 更新 health guardrails，使新增/删除/降级 console scripts 时必须同步 pyproject、CLI help smoke、inventory、docs 和 OpenSpec current surface。

## Capabilities

### New Capabilities

- 无。本 change 只收缩和治理现有 public entrypoint surface。

### Modified Capabilities

- `project-entrypoint-lifecycle`：增加 package console script lifecycle、internal-only CLI 降级规则、public help smoke 覆盖要求，以及删除 public entrypoint 时不得新增 wrapper 的约束。
- `project-health-guardrails`：增加 console script surface doctor/architecture checks，要求 pyproject、help smoke、inventory 和 current docs 保持一致。

## Impact

- 影响范围：`pyproject.toml` entry points、`src/kd_sensing/cli/` 下入口模块、`tests/test_cli_help.py`、`tests/test_architecture_boundaries.py`、`docs/project_surface_inventory.md`、README/docs 中的推荐命令。
- 不影响范围：训练数学语义、模型 forward、数据 split、配置解析语义、runtime outputs、checkpoint 和本地数据。
- 兼容性：部分 `kd-sensing-*` 命令可能被删除或不再作为 public API；保留的 internal helper 可通过 owner module 或明确文档入口访问，但不得添加旧命令 alias。
- 验证：OpenSpec strict validate、CLI help smoke、architecture boundary、surface doctor 和受影响 CLI 的 `--help`/dry-run 检查。
