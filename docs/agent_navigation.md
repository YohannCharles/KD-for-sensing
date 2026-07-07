# AI / Maintainer Navigation

本文件是非平凡改动前的薄导航层；需求契约看 active OpenSpec 和 `openspec/specs/`，操作规则看 `AGENTS.md`，表面积事实看 `docs/project_surface_inventory.md` 与 `docs/maintainer_context_index.yaml`。

## 当前一屏摘要

- 当前主线：final C2 / U-MaskBeamJEPA 缺失模态 beam prediction。
- 保留支线：MMW/CSI future/current supporting workflow，包括 MMW Town GPS v2、physics-informed MMW 和 CSI hardening。
- 推荐入口：`kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-eval-u-mask-matrix`、`kd-sensing-mmw-town-gps-v2`、`kd-sensing-inspect-mmw-physics`、`kd-sensing-project-surface-doctor`。
- 已退役历史面：Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、旧 RBMA/KD/BTAPA/weakKD sweep、HiST-Beam、Top8 selector、GPS residual、camera residual、Raymobtime s008、BGAM、viewer manifest、Gradio viewer、CRAF、MARF、Multimodal-NF。它们只可作为退役/历史/防回流语境，不可恢复为 current wrapper、CLI 或 YAML。
- 判断口径：tracked/source/current lifecycle 优先；不要从 archive、generated metadata、ignored `outputs/`、`logs/`、cache、checkpoint 或本地数据反推当前支持面。

## 检查顺序

1. 读用户请求、`AGENTS.md` 和本文件。
2. 对 active change 运行 `openspec list --json`、`openspec status --change <change> --json`、`openspec instructions apply --change <change> --json`。
3. 读 `docs/project_surface_inventory.md`，确认 protected inventory、public CLI、scripts、config lifecycle 和 OpenSpec lifecycle。
4. 按任务加载 scoped context：`docs/agent_context/README.md`、`docs/agent_context/models.md`、`docs/agent_context/data.md`、`docs/agent_context/configs.md`、`docs/agent_context/cli.md`、`docs/agent_context/diagnostics.md`、`docs/agent_context/openspec.md`、`docs/agent_context/documentation.md`、`docs/agent_context/claims.md`、`docs/agent_context/atlas.md`。
5. 最后看源码、测试和 `git status --short`；不要修改 `dataset/`、`outputs/`、`logs/`、cache 或 checkpoint。

## 任务路由表

| Route id | Scoped context | 常见触发 |
| --- | --- | --- |
| `model` | `docs/agent_context/models.md` | 模型、forward、registry、组件扩展 |
| `data` | `docs/agent_context/data.md` | dataset、batch contract、modality profile、split |
| `config` | `docs/agent_context/configs.md` | YAML、virtual config、canonical recipe、migration guard |
| `cli` | `docs/agent_context/cli.md` | console scripts、package CLI、`scripts/` |
| `diagnostics` | `docs/agent_context/diagnostics.md` | U-Mask eval matrix、run index、doctor、paper export、MMW/CSI diagnostics |
| `openspec` | `docs/agent_context/openspec.md` | proposal/spec/tasks/archive |
| `documentation` | `docs/agent_context/documentation.md` | README、inventory、导航、文档健康 |
| `claims` | `docs/agent_context/claims.md` | claim registry、paper tables、provenance |
| `atlas` | `docs/agent_context/atlas.md` | spec/config/claim owner 和 lifecycle 快速扫视 |

## 验证命令

| 触碰范围 | 推荐命令 |
| --- | --- |
| 常规 quick verify | `make verify-quick` |
| CLI/config | `make verify-cli-config` |
| scripts/package CLI compile | `make verify-compile` |
| OpenSpec change | `openspec validate <change> --strict` |
| 全量 OpenSpec | `openspec validate --all --strict` |
| 架构边界 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` |
| MMW/CSI | `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmw_town_gps_adapter_v2.py tests/test_csi_modality.py tests/test_physics_informed_mmw.py -q` |

## 项目技能

常用项目技能仍登记在 `.codex/skills/kd-add-model/SKILL.md`、`.codex/skills/kd-add-config/SKILL.md`、`.codex/skills/kd-update-claim/SKILL.md`、`.codex/skills/kd-diagnose-run/SKILL.md`、`.codex/skills/kd-archive-change/SKILL.md`。所有技能仍遵守 OpenSpec、`kd_mm_beam`、`dataset/`、`outputs/` 和 `logs/` 边界。
