# KD for Sensing

当前默认主线是 final C2 / U-MaskBeamJEPA 缺失模态波束预测。仓库只把核心训练、评估、预处理、U-Mask eval matrix、MMW/CSI workflow 和必要治理入口作为 current public surface；Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、旧 RBMA/KD/BTAPA/weakKD sweep 和一次性诊断入口已退役或降级为历史说明。

## Quickstart

所有项目 Python 命令都通过 `kd_mm_beam` 环境运行：

```bash
conda run -n kd_mm_beam kd-sensing-train --help
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix --help
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --help
conda run -n kd_mm_beam kd-sensing-inspect-mmw-physics --help
conda run -n kd_mm_beam kd-sensing-project-surface-doctor --help
```

常用主线 smoke：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/u_mask_beam_jepa_smoke.yaml
conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix --config configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml --checkpoint outputs/local/best.pth
```

MMW/CSI 仍保留为 future/current supporting dataset workflow：

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --help
conda run -n kd_mm_beam kd-sensing-inspect-mmw-physics --help
conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmw_town_gps_adapter_v2.py tests/test_csi_modality.py tests/test_physics_informed_mmw.py -q
```

## Project Boundaries

- 数据输入在 `dataset/`，默认不提交真实数据。
- 训练输出、日志、cache、TensorBoard、checkpoint 和分析产物写入 ignored 的 `outputs/`、`outputs/cache/`、`logs/` 或显式本地路径，默认不提交。
- 当前保留 YAML/manifest 以 final C2、U-MaskBeamJEPA、MMW/CSI、claim/evidence 和 focused tests 为准；无法确认是否主线使用的文件标为 `pending-confirmation` 或 `protected-until-next-audit`，本轮不删。
- U-MaskBeamJEPA 的 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 及既有 loss/forward 开关本轮保留，后续若删必须另开 OpenSpec change。

## Retired Surface

已退役历史路线只保留防回流语境：HiST-Beam、Top8 selector、GPS residual、camera residual、Raymobtime s008、BGAM、viewer manifest、Gradio viewer、CRAF、MARF、Multimodal-NF、旧 KD、Image+GPS JEPA 诊断、BeamBench/BEV-Fusion 2604/Vision-Position 复现、旧 RBMA/KD/BTAPA/weakKD sweep 和相关一次性 CLI/runbook 不再作为 current 推荐入口，也不提供兼容 stub、alias、virtual config 或 package facade。

## Documentation

- 操作规则：`AGENTS.md`
- AI 导航：`docs/agent_navigation.md`
- 表面积 inventory：`docs/project_surface_inventory.md`
- 最小机器索引：`docs/maintainer_context_index.yaml`
- scoped context：`docs/agent_context/README.md`、`docs/agent_context/models.md`、`docs/agent_context/data.md`、`docs/agent_context/configs.md`、`docs/agent_context/cli.md`、`docs/agent_context/diagnostics.md`、`docs/agent_context/openspec.md`、`docs/agent_context/documentation.md`、`docs/agent_context/claims.md`、`docs/agent_context/atlas.md`
- 当前研究简报：`docs/current_research_brief.md`
- Claim/protocol：`docs/result_claims_registry.md`、`docs/experiment_protocols.md`、`docs/experiment_matrix.md`、`docs/mainline_model_catalog.md`
- 只读协作角色与记忆账本：`docs/readonly_agent_roles.md`、`docs/agent_memory_ledger.md`

## Verification

```bash
openspec validate --all --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q
conda run -n kd_mm_beam python scripts/verify_compile.py
```
