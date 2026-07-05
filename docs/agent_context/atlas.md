# Agent Atlas

本 atlas 是人工维护的高信号索引，来源为 README、`docs/project_surface_inventory.md`、`docs/result_claims_registry.md` 和当前 OpenSpec specs。它只引用权威路径、lifecycle、owner、focused validation 和 caveat，不复制完整 requirement、完整 config 数据库或完整 claim 表。

Last reviewed: 2026-07-05
Source: manual curation for `improve-agent-context-routing`

## 字段

| Field | Meaning |
| --- | --- |
| `kind` | `spec`、`config` 或 `claim` |
| `id` | Agent routing 可读 id，不要求等同文件名 |
| `lifecycle` | `current`、`supporting`、`retired-tombstone`、`reviewed`、`draft`、`blocked` 等权威状态 |
| `authority_path` | 权威文档、spec、config family 或 registry path |
| `owner` | 主要源码 owner、文档 owner 或人工维护 owner |
| `focused_validation` | 最小验证命令或 focused test |
| `caveat` | 使用时必须保留的限制 |
| `source` | 本条来自 inventory、README、claim registry、OpenSpec 或手工 change |

## Spec atlas

| id | lifecycle | authority_path | owner | focused_validation | caveat |
| --- | --- | --- | --- | --- | --- |
| `model-extension` | `current` | `openspec/specs/model-architecture-extension-contract/spec.md` | `src/kd_sensing/models/` | `conda run -n kd_mm_beam pytest tests/test_component_registry.py -q` | 先判断 config-only、component、whole-model exception 或 workflow reproduction |
| `config-resolution` | `current` | `openspec/specs/canonical-config-resolution/spec.md` | `src/kd_sensing/config/` | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` | virtual config 不接管 retired route |
| `entrypoint-lifecycle` | `current` | `openspec/specs/project-entrypoint-lifecycle/spec.md` | `pyproject.toml` and `src/kd_sensing/cli/` | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` | CLI 保持 thin glue，workflow 逻辑回 owner |
| `diagnostic-surface` | `current` | `openspec/specs/jepa-visual-analysis-suite/spec.md` | `src/kd_sensing/diagnostics/` | `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q` | 诊断产物默认 ignored runtime root |
| `claim-governance` | `current` | `openspec/specs/mainline-experiment-documentation/spec.md` | `docs/result_claims_registry.md` | `conda run -n kd_mm_beam kd-sensing-paper-export --input docs/result_claims_registry.md --output-dir outputs/paper_artifacts/current` | draft/candidate 不进入 reviewed main claim |
| `maintainer-index` | `supporting` | `openspec/specs/maintainer-context-index/spec.md` | `docs/maintainer_context_index.yaml` | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | index 只放最小结构化事实 |
| `retired-routes` | `supporting` | `openspec/specs/retired-route-summary/spec.md` | migration guards and inventory | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | 只作为退役和防回流边界 |

## Config atlas

| id | lifecycle | authority_path | owner | focused_validation | caveat |
| --- | --- | --- | --- | --- | --- |
| `single-modality-canonical` | `current` | README configuration section | `configs/` and `src/kd_sensing/config/` | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` | 文件名可稳定，语义由 current `model.primary` 表达 |
| `fusion-canonical-virtual` | `current` | `docs/project_surface_inventory.md` config lifecycle | `src/kd_sensing/config/canonical.py` | `make verify-cli-config` | virtual recipe 不恢复 KD/HiST/residual/BGAM/viewer |
| `diagnostic-manifests` | `current` | `configs/diagnostics/` | `src/kd_sensing/diagnostics/` | `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q` | checkpoint 和 figure/report 输出不提交 |
| `scene31-local-manual` | `current` | `configs/scene31/` | `scripts/generate_scene31_next_round.py` | `conda run -n kd_mm_beam pytest tests/test_scene31_next_round.py -q` | generator-backed YAML 默认本地生成，不纳入源码变更 |

## Claim atlas

| id | lifecycle | authority_path | owner | focused_validation | caveat |
| --- | --- | --- | --- | --- | --- |
| `reviewed-claims` | `reviewed` | `docs/result_claims_registry.md` | docs maintainer | `conda run -n kd_mm_beam kd-sensing-paper-export --input docs/result_claims_registry.md --output-dir outputs/paper_artifacts/current` | 需要 metric、split、seed/checkpoint 和 provenance |
| `protocol-facts` | `current` | `docs/experiment_protocols.md` | docs maintainer | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | 只维护参数口径，不覆盖 runtime spec |
| `candidate-ledger` | `draft` | `outputs/analysis/` | local diagnostics | related diagnostic focused tests | ignored 本地产物，只能作为待审 candidate |
| `blocked-official` | `blocked` | `docs/result_claims_registry.md` | docs maintainer | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | blocked reason 必须和 official artifact / data availability caveat 对齐 |

## 维护方式

- 新增 atlas row 前，先确认该事实无法直接从 single owner path 低成本推导，且确实服务高频 agent routing。
- lifecycle 以 `docs/project_surface_inventory.md`、当前 OpenSpec specs 或 `docs/result_claims_registry.md` 为准；冲突时视为治理漂移，不能任选一处。
- Atlas 只补 route 级摘要。完整 requirement、完整 config 列表、claim 表格正文和历史 caveat 留在权威文档。
- 修改本文件后运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
