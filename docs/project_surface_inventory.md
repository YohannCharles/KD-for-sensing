# 项目表面积 Inventory

本 inventory 是 post-C2 清理后的可审计基线。权威顺序仍是用户请求、`AGENTS.md`、active OpenSpec、本文、README/docs、源码和测试。默认只统计 tracked source/config/docs/OpenSpec；`dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard 和本地训练产物不属于源码变更。

## 项目健康护栏基线

`prune-post-c2-nonmainline-surface` 于 2026-07-07 刷新 on-disk tracked-only 统计口径；可用 `git ls-files` 复核。扫描口径覆盖 `src/kd_sensing`、`tests/`、`scripts/`、`configs/`、README、docs 和 OpenSpec。排除项包括 `dataset/`、`outputs/`、`logs/`、`outputs/cache/`、legacy `cache`、`.pytest_cache/`、`__pycache__/`、`.pyc`、checkpoint、权重和其它本地运行产物。这些数字和 AST/CodeGraph 大小只作为趋势信号和右尺寸化上下文，非硬 KPI。

推荐验证：

- `openspec validate --all --strict`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
- `conda run -n kd_mm_beam python scripts/verify_compile.py`
- MMW/CSI touched 时追加 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmw_town_gps_adapter_v2.py tests/test_csi_modality.py tests/test_physics_informed_mmw.py -q`

当前 AST 热点清单如下；预算用于阻止热点静默扩大，不表示必须在本 change 拆分：

| Owner | Action | Focused validation |
| --- | --- | --- |
| `src/kd_sensing/data/datasets/deepsense6g.py` | `monitor` dataset orchestration | `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py -q` |
| `src/kd_sensing/data/difficulty/operators/image.py` | `keep-and-test` difficulty operators | `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` |
| `src/kd_sensing/diagnostics/apples_to_apples_evaluation.py` | `accepted-size` read-only diagnostic owner | `conda run -n kd_mm_beam pytest tests/test_missing_modality_stress.py -q` |
| `src/kd_sensing/diagnostics/scene31_34_final_analysis/main_summary.py` | `accepted-size` final evidence summary owner | `conda run -n kd_mm_beam pytest tests/test_missing_modality_stress.py -q` |
| `src/kd_sensing/engine/mmw_town_gps_v2.py` | `accepted-size` protected MMW workflow | `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q` |
| `src/kd_sensing/engine/objectives/metadata.py` | `keep-and-test` objective metadata owner | `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py -q` |
| `src/kd_sensing/engine/run_metadata.py` | `monitor` metadata writer | `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` |
| `src/kd_sensing/losses/u_mask_beam_jepa.py` | `keep-and-test` mainline loss branch owner | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py -q` |
| `src/kd_sensing/models/architecture_summary.py` | `keep-and-test` instance/startup summary schema; public CLI retired | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_amr_net.py tests/test_amber_full_architecture.py -q` |
| `src/kd_sensing/models/jepa.py` | `keep-and-test` retained JEPA component support | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` |
| `src/kd_sensing/models/jepa_downstream.py` | `keep-and-test` retained JEPA downstream support | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` |
| `src/kd_sensing/models/modular.py` | `accepted-size` shared modular model | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` |
| `src/kd_sensing/models/u_mask_beam_jepa.py` | `keep-and-test` final C2 mainline model | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py -q` |

## Post-C2 Protected Inventory

| Class | Protected surface | Notes |
| --- | --- | --- |
| `protected_mainline` | `src/kd_sensing/models/u_mask_beam_jepa.py`, `src/kd_sensing/losses/u_mask_beam_jepa.py`, `configs/fusion/u_mask_beam_jepa_*.yaml`, `configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml`, `scripts/launch_final_c2_ablation_v1.py`, `scripts/summarize_final_c2_ablation_v1.py` | final C2 / U-MaskBeamJEPA current mainline |
| `protected_umask_branches` | `pcpg`, `bprr`, `raw_conf_gate`, `weighted_sum`, `concat_mlp`, `supervised_router` | 本 change 只登记后续审计触发条件，不删实现 |
| `protected_mmw` | `src/kd_sensing/data/mmw/`, `src/kd_sensing/data/datasets/mmw*.py`, `src/kd_sensing/engine/mmw_town_gps_v2*.py`, `src/kd_sensing/models/physics/`, `configs/mmw_town_gps_adapter_v2.yaml`, `configs/fusion/physics_informed_mmw*.yaml`, `scripts/mmw/visualize_town_label_distribution.py` | MMW future/current supporting workflow |
| `protected_csi` | `configs/csi/`, `configs/fusion/csi_hardening_matrix/`, CSI models/tests | CSI hardening workflow |
| `protected_claim_docs` | `docs/result_claims_registry.md`, `docs/experiment_protocols.md`, `docs/experiment_matrix.md`, `docs/mainline_model_catalog.md`, `docs/mainline_experiment_history.md` | current claim/evidence gate |
| `pending-confirmation` | generated final C2 config manifests under ignored `outputs/` | 不纳入源码，不作为删除候选 |

## Deletion Candidate Ledger

| Candidate group | Current references audited | Replacement | Validation | Rollback |
| --- | --- | --- | --- | --- |
| non-mainline package CLI: JEPA visual/GPS shortcut, throughput, target-shot, distribution-shift, WCL, dataset audit, BeamBench, TII, model summary | pyproject, README/docs, CLI smoke, current specs | retained public CLI table below | `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` | restore deleted CLI modules and pyproject entries from git only if a new OpenSpec change re-promotes them |
| one-shot scripts and old Scene31 runbooks | tracked `scripts/`, inventory, docs | retained final C2 / PCPG / BPRR / Scene31-34 main scripts | `conda run -n kd_mm_beam python scripts/verify_compile.py` | restore specific script from git and add lifecycle row |
| BeamBench, BEV-Fusion 2604, Vision-Position, Image+GPS JEPA diagnostics, old RBMA/WCL/TII source | imports, registries, tests, docs, specs | U-MaskBeamJEPA mainline, MMW/CSI workflows, historical notes | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | restore module family from git and undo removed registry guard only through a new change |
| historical YAML/manifest families | configs, docs, claim registry, tests, OpenSpec | protected root U-Mask/MMW/CSI configs and generator-backed Scene31 template | `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` | restore exact YAML from git if a current claim/protocol needs it |

## Public CLI Lifecycle

| Command | Lifecycle | Owner | Responsibility | Output boundary | Focused validation |
| --- | --- | --- | --- | --- | --- |
| `kd-sensing-train` | `core_workflow` | `kd_sensing.engine.trainer` | config-driven training entrypoint | ignored outputs/ and logs/ run roots | `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q` |
| `kd-sensing-evaluate` | `core_workflow` | `kd_sensing.engine.evaluation_pass` | checkpoint evaluation entrypoint | ignored evaluation/output roots or user path | `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q` |
| `kd-sensing-preprocess` | `core_workflow` | `kd_sensing.preprocessing` | config-driven preprocessing entrypoint | dataset preparation targets or ignored cache/output roots | `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q` |
| `kd-sensing-runs` | `core_workflow` | `kd_sensing.diagnostics.run_index` | read-only local run index | stdout or explicit ignored analysis path | `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_runtime_artifact_cleanup.py -q` |
| `kd-sensing-clean-runtime-artifacts` | `current_diagnostic` | `kd_sensing.diagnostics.runtime_artifact_cleanup` | runtime artifact cleanup manifest workflow | ignored outputs/cleanup_manifests/ or explicit manifest/report path | `conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_cli_help.py -q` |
| `kd-sensing-organize-runtime-outputs` | `current_diagnostic` | `kd_sensing.diagnostics.runtime_artifact_cleanup` | runtime output organize manifest workflow | ignored outputs/cleanup_manifests/ or explicit manifest/report path | `conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_cli_help.py -q` |
| `kd-sensing-paper-export` | `paper_export` | `kd_sensing.diagnostics.paper_artifact_export` | reviewed claim table and figure-data export | ignored outputs/paper_artifacts/ or explicit output dir | `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q` |
| `kd-sensing-eval-u-mask-matrix` | `current_diagnostic` | `kd_sensing.eval.u_mask_beam_jepa_eval_matrix` | U-MaskBeamJEPA missing-modality evaluation matrix | ignored outputs/eval/ or explicit output dir | `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa_eval_matrix.py tests/test_cli_help.py -q` |
| `kd-sensing-mmw-town-gps-v2` | `current_diagnostic` | `kd_sensing.engine.mmw_town_gps_v2` | MMW Town GPS-only v2 run, plot and compare workflow | ignored outputs/analysis/mmw_town_gps_adapter_v2/ or explicit output dir | `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_cli_help.py -q` |
| `kd-sensing-inspect-mmw-physics` | `current_diagnostic` | `kd_sensing.models.physics` | physics-informed MMW sample inspection | stdout only unless explicit output path is added by caller | `conda run -n kd_mm_beam pytest tests/test_physics_informed_mmw.py tests/test_cli_help.py -q` |

## Script Lifecycle

| Script | Lifecycle | Owner | Retained reason | Public/recommended relation | Output boundary | Focused validation | Delete when |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `scripts/analyze_mmw_fused_feature_geometry.py` | `local/manual` | MMW T2 evidence | active BPA/CMA feature evidence | not public; active change helper | ignored `outputs/analysis/` | `conda run -n kd_mm_beam pytest tests/test_mmw_fused_feature_geometry.py -q` | BPA/CMA change is archived and evidence is migrated |
| `scripts/eval_h5_p1_temporal_matrix_v1.py` | `research_diagnostic` | H5/P1 temporal | group-safe final test matrix | not public; launcher companion | ignored `outputs/h5_p1_temporal_models_v1/` | `conda run -n kd_mm_beam pytest tests/test_h5_p1_temporal_matrix_v1.py -q` | temporal matrix capability is retired |
| `scripts/eval_mmw_all_weather_matrix.py` | `local/manual` | MMW all-weather | current all-weather evaluation | not public; campaign evaluator | ignored `outputs/mmw_all_weather_h5p1_seed1_v2/` | `conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py -q` | all-weather capability is retired and claims are migrated |
| `scripts/generate_scenes31_34_encoder_ablation.py` | `local/manual` | Scene31-34 | tracked template generator | not public; documented helper | ignored generated configs and `outputs/` | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | encoder ablation workflow is retired |
| `scripts/generate_scenes31_34_main.py` | `local/manual` | Scene31-34 | mainline config generator | not public; documented helper | ignored generated configs and `outputs/` | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | Scene31-34 workflow has a package owner |
| `scripts/inspect_dataset.py` | `local/manual` | dataset diagnostics | read-only local inspection | not public; optional diagnostic | stdout unless caller selects local output | `conda run -n kd_mm_beam python scripts/inspect_dataset.py --help` | package preprocessing exposes equivalent read-only audit |
| `scripts/launch_bprr_reliability_router_v1.py` | `local/manual` | BPRR | current follow-up launcher | not public; research launcher | ignored `outputs/bprr_reliability_router_v1/` | `conda run -n kd_mm_beam python scripts/launch_bprr_reliability_router_v1.py --help` | BPRR capability is retired and claims are migrated |
| `scripts/launch_final_c2_ablation_v1.py` | `local/manual` | final C2 | default mainline ablation launcher | not public; research launcher | ignored `outputs/final_c2_ablation_v1/` | `conda run -n kd_mm_beam python scripts/launch_final_c2_ablation_v1.py --help` | final C2 workflow has a replacement owner |
| `scripts/launch_h5_p1_temporal_models_v1.py` | `local/manual` | H5/P1 temporal | group-safe temporal launcher | not public; research launcher | ignored `outputs/h5_p1_temporal_models_v1/` | `conda run -n kd_mm_beam pytest tests/test_h5_p1_temporal_matrix_v1.py -q` | temporal matrix capability is retired |
| `scripts/launch_mmw_all_weather_matrix.py` | `local/manual` | MMW all-weather | current campaign launcher | not public; campaign launcher | ignored `outputs/mmw_all_weather_h5p1_seed1_v2/` | `conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py -q` | all-weather capability is retired and claims are migrated |
| `scripts/launch_mmw_t2_hyperparameter_screening.py` | `local/manual` | MMW T2 development | active hyperparameter screening launcher | not public; active change helper | ignored `outputs/mmw_t2_hyperparameter_screening_v1/` | `conda run -n kd_mm_beam pytest tests/test_mmw_t2_hyperparameter_screening.py -q` | hyperparameter screening change is archived and evidence is migrated |
| `scripts/launch_pcpg_radar_balance_v1.py` | `local/manual` | PCPG | current follow-up launcher | not public; research launcher | ignored `outputs/pcpg_radar_balance_v1/` | `conda run -n kd_mm_beam python scripts/launch_pcpg_radar_balance_v1.py --help` | PCPG capability is retired and claims are migrated |
| `scripts/mmw/visualize_town_label_distribution.py` | `research_diagnostic` | MMW labels | protected label audit | not public; optional diagnostic | ignored `outputs/analysis/mmw/` or explicit output | `conda run -n kd_mm_beam python scripts/mmw/visualize_town_label_distribution.py --help` | package MMW audit provides equivalent evidence |
| `scripts/run_mmw_all_weather_eval_after_training.py` | `local/manual` | MMW all-weather | post-training shard orchestration | not public; campaign orchestrator | ignored MMW status, logs and summaries | `conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py -q` | all-weather capability is retired and claims are migrated |
| `scripts/run_mmw_t2_bpa_cma_ablation_after_training.py` | `local/manual` | MMW T2 evidence | active BPA/CMA orchestration | not public; active change helper | ignored task status, logs and summaries | `conda run -n kd_mm_beam pytest tests/test_mmw_t2_bpa_cma_after_training.py -q` | BPA/CMA change is archived and evidence is migrated |
| `scripts/run_scenes31_34_main.sh` | `local/manual` | Scene31-34 | protected mainline runner | not public; documented runner | ignored `outputs/scenes31_34_main_lmdb/` and `logs/` | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | Scene31-34 workflow has a replacement owner |
| `scripts/run_scenes31_34_tinyvit_ablation.sh` | `local/manual` | Scene31-34 | encoder ablation runner | not public; documented runner | ignored `outputs/scenes31_34_tinyvit_lmdb/` and `logs/` | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | TinyViT ablation is retired |
| `scripts/scene31_eval_resolution.py` | `supporting` | Scene31-34 | shared evaluation resolution | helper only; not direct | stdout or caller-owned ignored output | `conda run -n kd_mm_beam python scripts/verify_compile.py` | no retained caller imports it |
| `scripts/scene31_generator_common.py` | `supporting` | Scene31-34 | shared generator logic | helper only; not direct | caller-owned generated configs | `conda run -n kd_mm_beam python scripts/verify_compile.py` | no retained generator imports it |
| `scripts/scene31_runner_common.py` | `supporting` | Scene31-34 | shared Python runner logic | helper only; not direct | caller-owned ignored outputs | `conda run -n kd_mm_beam python scripts/verify_compile.py` | no retained runner imports it |
| `scripts/scene31_runner_common.sh` | `supporting` | Scene31-34 | shared shell runner logic | helper only; not direct | caller-owned ignored outputs and logs | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | no retained shell runner sources it |
| `scripts/summarize_bprr_reliability_router_v1.py` | `research_diagnostic` | BPRR | claim-gated summary | not public; launcher companion | ignored `outputs/analysis/` | `conda run -n kd_mm_beam python scripts/summarize_bprr_reliability_router_v1.py --help` | BPRR claims are migrated and workflow retired |
| `scripts/summarize_final_c2_ablation_v1.py` | `research_diagnostic` | final C2 | mainline evidence summary | not public; launcher companion | ignored `outputs/analysis/` | `conda run -n kd_mm_beam python scripts/summarize_final_c2_ablation_v1.py --help` | final C2 has a replacement evidence owner |
| `scripts/summarize_h5_p1_temporal_matrix_v1.py` | `research_diagnostic` | H5/P1 temporal | group-safe matrix summary | not public; evaluator companion | ignored temporal final summary | `conda run -n kd_mm_beam pytest tests/test_h5_p1_temporal_matrix_v1.py -q` | temporal matrix capability is retired |
| `scripts/summarize_mmw_all_weather_matrix.py` | `research_diagnostic` | MMW all-weather | current campaign summary | not public; evaluator companion | ignored MMW final summary | `conda run -n kd_mm_beam pytest tests/test_mmw_all_weather_runtime.py -q` | all-weather claims are migrated and workflow retired |
| `scripts/summarize_mmw_multiseed_baselines.py` | `research_diagnostic` | MMW baselines | current multiseed evidence gate | not public; campaign summary | explicit ignored analysis output | `conda run -n kd_mm_beam pytest tests/test_summarize_mmw_multiseed_baselines.py -q` | multiseed capability is retired and claims are migrated |
| `scripts/summarize_mmw_t2_bpa_cma_ablation.py` | `research_diagnostic` | MMW T2 evidence | active paired ablation summary | not public; active change helper | explicit ignored analysis output | `conda run -n kd_mm_beam pytest tests/test_summarize_mmw_t2_bpa_cma_ablation.py -q` | BPA/CMA change is archived and evidence is migrated |
| `scripts/summarize_mmw_task_output_robustness.py` | `research_diagnostic` | MMW T2 evidence | paired task-output robustness | not public; active change helper | explicit ignored analysis output | `conda run -n kd_mm_beam pytest tests/test_mmw_task_output_robustness.py -q` | no active MMW evidence workflow consumes it |
| `scripts/summarize_overnight_branch_router_v2.py` | `research_diagnostic` | overnight router | retained historical summary parser | not public; historical helper | ignored `outputs/analysis/` | `conda run -n kd_mm_beam pytest tests/test_overnight_branch_router_v2.py -q` | historical claim no longer needs reproduction |
| `scripts/summarize_pcpg_radar_balance_v1.py` | `research_diagnostic` | PCPG | claim-gated summary | not public; launcher companion | ignored `outputs/analysis/` | `conda run -n kd_mm_beam python scripts/summarize_pcpg_radar_balance_v1.py --help` | PCPG claims are migrated and workflow retired |
| `scripts/verify_compile.py` | `governance` | project health | on-disk syntax guard | recommended through `make verify-compile` | stdout only | `conda run -n kd_mm_beam python scripts/verify_compile.py` | an equivalent repository-native guard replaces it |

## Document Lifecycle

| Document | Lifecycle | Owner | Purpose |
| --- | --- | --- | --- |
| `AGENTS.md` | `current` | repository governance | agent rules and command boundary |
| `CLAUDE.md` | `adapter` | repository governance | thin cross-tool navigation adapter |
| `ENVIRONMENT.md` | `historical` | repository governance | dated, non-authoritative environment capture |
| `README.md` | `current` | repository documentation | install, quickstart, boundaries and short index |
| `docs/agent_navigation.md` | `current` | repository governance | task navigation and authority routing |
| `docs/maintainer_context_index.yaml` | `current` | repository governance | minimal machine-readable task routing |
| `docs/agent_project_knowledge.md` | `adapter` | repository governance | thin cross-tool project context adapter |
| `docs/agentic_collaboration_guardrails.md` | `current` | repository governance | agent collaboration and review boundary |
| `docs/project_surface_inventory.md` | `current` | repository governance | lifecycle and retained-surface decisions |
| `docs/current_research_brief.md` | `current` | research documentation | short current research orientation |
| `docs/extension_guide.md` | `current` | repository documentation | supported extension workflow |
| `docs/mainline_model_catalog.md` | `current` | research documentation | model-line catalog and claim references |
| `docs/experiment_protocols.md` | `current` | research documentation | current protocol and evidence gates |
| `docs/experiment_matrix.md` | `current` | research documentation | current experiment entry map |
| `docs/literature_matrix.md` | `current` | research documentation | literature scope and evidence classification |
| `docs/model_architecture_inventory.md` | `current` | model governance | model owner and architecture inventory |
| `docs/result_claims_registry.md` | `current` | claim governance | structured claim ledger |
| `docs/mainline_experiment_history.md` | `historical` | claim governance | migrated caveats and retired evidence |
| `docs/p3_v7_multisource_crossroad_analysis.md` | `historical` | research documentation | dated P3/V7 local analysis; not a current claim |
| `docs/progress_report_ppt_revision_and_t2_plan.md` | `historical` | research documentation | archived T2 report feedback and cancelled execution boundary |
| `docs/research_notes.md` | `historical` | research documentation | retained historical research judgments |
| `docs/server_migration_github_codex.md` | `current` | repository operations | source and artifact-safe server migration guide |
| `docs/training_throughput.md` | `current` | runtime documentation | supported throughput controls and measurement boundary |
| `docs/u_mask_beam_jepa_current_scope.md` | `current` | research documentation | final C2 method scope |
| `docs/u_mask_beam_jepa_eval_matrix.md` | `current` | research documentation | U-Mask evaluation matrix contract and usage |
| `docs/readonly_agent_roles.md` | `current` | repository governance | read-only analysis role boundary |
| `docs/agent_memory_ledger.md` | `current` | repository governance | manually reviewed recurring-error ledger |
| `paper/references.bib` | `current` | research documentation | source-controlled literature references only |
| `docs/agent_context/README.md` | `scoped-context` | repository governance | context routing index |
| `docs/agent_context/atlas.md` | `scoped-context` | repository governance | cross-surface atlas |
| `docs/agent_context/claims.md` | `scoped-context` | repository governance | claim task context |
| `docs/agent_context/cli.md` | `scoped-context` | repository governance | CLI task context |
| `docs/agent_context/configs.md` | `scoped-context` | repository governance | config task context |
| `docs/agent_context/data.md` | `scoped-context` | repository governance | data task context |
| `docs/agent_context/diagnostics.md` | `scoped-context` | repository governance | diagnostics task context |
| `docs/agent_context/documentation.md` | `scoped-context` | repository governance | documentation task context |
| `docs/agent_context/models.md` | `scoped-context` | repository governance | model task context |
| `docs/agent_context/openspec.md` | `scoped-context` | repository governance | OpenSpec task context |

已删除的 `README_REPRODUCE.md`、`BASELINE_REPORT.md`、`DATASET_STRUCTURE.md`、`PATCH_NOTES.md`、`TODO_FOR_ATTENTION_MODULE.md` 和 `results/reproduce_baseline.md` 属于退役 BeamBench 表面；唯一有效 caveat 已迁入 `docs/mainline_experiment_history.md`，不得恢复为 current 命令文档。

## Config Lifecycle

`configs/fusion/` 根目录保留分类如下：

| Config | Lifecycle |
| --- | --- |
| `all_modalities_lidar_supervised.yaml` | current canonical fusion |
| `all_modalities_supervised.yaml` | current canonical fusion |
| `amber_full_architecture.yaml` | current missing-modality baseline |
| `amber_lite_missing_modality.yaml` | current missing-modality baseline |
| `amr_net_supervised.yaml` | current missing-modality baseline |
| `image_gps_resnet18_modular_supervised.yaml` | current supervised control |
| `image_gps_supervised.yaml` | current supervised control |
| `mmwave_csi_medium_degraded_supervised.yaml` | protected CSI/MMW |
| `mmwave_csi_supervised.yaml` | protected CSI/MMW |
| `physics_informed_mmw_csi_only.yaml` | protected MMW |
| `physics_informed_mmw_debug.yaml` | protected MMW |
| `physics_informed_mmw_full_multimodal.yaml` | protected MMW |
| `physics_informed_mmw_history_csi_multimodal.yaml` | protected MMW |
| `physics_informed_mmw_hybrid.yaml` | protected MMW |
| `physics_informed_mmw_image_csi.yaml` | protected MMW |
| `physics_informed_mmw_image_only.yaml` | protected MMW |
| `physics_informed_mmw_no_array_consistency.yaml` | protected MMW |
| `physics_informed_mmw_no_csi_reconstruction.yaml` | protected MMW |
| `physics_informed_mmw_no_path_loss.yaml` | protected MMW |
| `physics_informed_mmw_no_physics.yaml` | protected MMW |
| `physics_informed_mmw_no_physics_head.yaml` | protected MMW |
| `physics_informed_mmw_oracle_full_csi.yaml` | protected MMW |
| `physics_informed_mmw_paper_debug.yaml` | protected MMW |
| `physics_informed_mmw_partial_csi_multimodal.yaml` | protected MMW |
| `physics_informed_mmw_sparse_pilot_multimodal.yaml` | protected MMW |
| `physics_informed_mmw_vision_only.yaml` | protected MMW |
| `radar_gps_supervised.yaml` | current canonical fusion |
| `radar_lidar_supervised.yaml` | current canonical fusion |
| `token_transformer_all_modalities_multitask_supervised.yaml` | current token transformer |
| `token_transformer_all_modalities_supervised.yaml` | current token transformer |
| `token_transformer_image_radar_supervised.yaml` | current token transformer |
| `u_mask_beam_jepa_concat_mlp.yaml` | protected U-Mask branch |
| `u_mask_beam_jepa_no_jepa.yaml` | protected U-Mask ablation |
| `u_mask_beam_jepa_no_uncertainty.yaml` | protected U-Mask ablation |
| `u_mask_beam_jepa_s32.yaml` | protected U-Mask |
| `u_mask_beam_jepa_smoke.yaml` | protected U-Mask smoke |
| `u_mask_beam_jepa_weighted_sum.yaml` | protected U-Mask branch |

已迁移到 post-C2 退役墓碑的配置族包括历史 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、RBMA/KD/BTAPA/weakKD、WCL/TII source-audit 和旧 Scene31 generated YAML。保留的 Scene31 tracked YAML 只有 `configs/scene31/templates/main_v3_proto_es20_base.yaml`；其它 Scene31 实体 YAML 应由 generator 在 ignored local root 生成。

## OpenSpec Capability Lifecycle

截至 2026-07-15，`openspec/specs/` 保留 86 个物理 current/supporting capability；无独立 guard 价值的 retired capability 已折叠到 `retired-route-summary` 和 dated archive，不再保留独立墓碑行。

| Capability | Lifecycle | Note |
| --- | --- | --- |
| `adaptive-pattern-balanced-sampler` | `current` | post-C2 保留/主线契约。 |
| `agent-context-portability` | `current` | post-C2 保留/主线契约。 |
| `agentic-collaboration-guardrails` | `current` | post-C2 保留/主线契约。 |
| `ai-maintainer-navigation` | `current` | post-C2 保留/主线契约。 |
| `amber-full-architecture-reproduction` | `current` | post-C2 保留/主线契约。 |
| `amber-lite-missing-modality-reproduction` | `current` | post-C2 保留/主线契约。 |
| `amr-net-architecture` | `current` | post-C2 保留/主线契约。 |
| `automated-cache-policy` | `current` | post-C2 保留/主线契约。 |
| `beam-topology-prototype-alignment` | `current` | post-C2 保留/主线契约。 |
| `beamspace-physical-labels` | `current` | post-C2 保留/主线契约。 |
| `bprr-reliability-router` | `current` | post-C2 保留/主线契约。 |
| `canonical-config-resolution` | `current` | post-C2 保留/主线契约。 |
| `cls-token-transformer-fusion` | `current` | post-C2 保留/主线契约。 |
| `component-registry` | `current` | post-C2 保留/主线契约。 |
| `csi-channel-data` | `current` | post-C2 保留/主线契约。 |
| `csi-channel-degradation` | `current` | post-C2 保留/主线契约。 |
| `csi-hardening-debug-validation` | `current` | post-C2 保留/主线契约。 |
| `csi-hardening-experiment-matrix` | `current` | post-C2 保留/主线契约。 |
| `csi-modality-model` | `current` | post-C2 保留/主线契约。 |
| `dataset-directory-layout` | `current` | post-C2 保留/主线契约。 |
| `dataset-loader-behavior` | `current` | post-C2 保留/主线契约。 |
| `dataset-runtime-contracts` | `current` | post-C2 保留/主线契约。 |
| `deepsense6g-scene-selection` | `current` | post-C2 保留/主线契约。 |
| `distillation-free-project-surface` | `current` | post-C2 保留/主线契约。 |
| `experiment-artifact-registry` | `current` | post-C2 保留/主线契约。 |
| `experiment-run-index` | `current` | post-C2 保留/主线契约。 |
| `experiment-workflow` | `current` | post-C2 保留/主线契约。 |
| `final-c2-ablation-v1` | `current` | post-C2 保留/主线契约。 |
| `first-class-prediction-tasks` | `current` | post-C2 保留/主线契约。 |
| `gps-conditioned-jepa-pretraining` | `current` | post-C2 保留/主线契约。 |
| `gps-modality-model` | `current` | post-C2 保留/主线契约。 |
| `gps-preprocessing` | `current` | post-C2 保留/主线契约。 |
| `image-preprocessing-profiles` | `current` | post-C2 保留/主线契约。 |
| `lidar-modality-model` | `current` | post-C2 保留/主线契约。 |
| `lidar-preprocessing` | `current` | post-C2 保留/主线契约。 |
| `local-missing-modality-baselines` | `supporting` | 只保留 Scene31-34 generator/runner 消费的 AMR-lite mask/gate contract；不恢复 FeatureMod 或旧 baseline pack。 |
| `mainline-experiment-documentation` | `current` | post-C2 保留/主线契约。 |
| `maintainer-context-index` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `missing-modality-statistical-evidence` | `current` | post-C2 保留/主线契约。 |
| `missing-modality-stress-suite` | `current` | post-C2 保留/主线契约。 |
| `mmw-all-weather-missing-modality-matrix` | `current` | MMW current supporting dataset campaign。 |
| `mmw-baseline-multiseed-robustness-evidence` | `current` | MMW current supporting dataset campaign。 |
| `mmw-beam-label-calibration` | `current` | post-C2 保留/主线契约。 |
| `mmw-cross-scene-adaptation-protocol` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `mmw-sensor-assisted-beam-prediction` | `current` | post-C2 保留/主线契约。 |
| `mmw-town-gps-adapter-v2` | `current` | post-C2 保留/主线契约。 |
| `mmw-town10-dataset-preparation` | `current` | post-C2 保留/主线契约。 |
| `mmwave-modality-model` | `current` | post-C2 保留/主线契约。 |
| `mmwave-preprocessing` | `current` | post-C2 保留/主线契约。 |
| `modality-contracts` | `current` | post-C2 保留/主线契约。 |
| `modality-difficulty-pipeline` | `current` | post-C2 保留/主线契约。 |
| `model-architecture-extension-contract` | `current` | post-C2 保留/主线契约。 |
| `model-architecture-summary` | `supporting` | 只保留实例参数、组件、可训练性与 startup artifact 内嵌 schema；独立 CLI 和 renderer 已退役。 |
| `modular-sequence-model` | `current` | post-C2 保留/主线契约。 |
| `multi-task-occlusion-position-learning` | `current` | post-C2 保留/主线契约。 |
| `observability-aware-fusion` | `current` | post-C2 保留/主线契约。 |
| `openspec-document-health` | `current` | post-C2 保留/主线契约。 |
| `original-code-compatibility` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `overnight-branch-router-v2` | `current` | post-C2 保留/主线契约。 |
| `paper-artifact-export` | `current` | post-C2 保留/主线契约。 |
| `pcpg-radar-balance-robustness` | `current` | post-C2 保留/主线契约。 |
| `physics-informed-mmw-beam-baseline` | `current` | post-C2 保留/主线契约。 |
| `project-architecture` | `current` | post-C2 保留/主线契约。 |
| `project-entrypoint-lifecycle` | `current` | post-C2 保留/主线契约。 |
| `project-health-guardrails` | `current` | post-C2 保留/主线契约。 |
| `project-hotspot-governance` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `project-import-surface-consolidation` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `project-surface-cleanup` | `current` | post-C2 保留/主线契约。 |
| `radar-student-model` | `current` | post-C2 保留/主线契约。 |
| `radar-teacher-model` | `current` | post-C2 保留/主线契约。 |
| `research-literature-matrix` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `resnet18-image-encoder` | `current` | post-C2 保留/主线契约。 |
| `retired-route-summary` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `runtime-artifact-cleanup` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `s1-lightweight-temporal-robustness` | `current` | S1/T2 temporal supporting evidence contract。 |
| `scenes31-34-main-missing-modality-workflow` | `current` | post-C2 保留/主线契约。 |
| `snapshot-next-frame-baselines` | `current` | post-C2 保留/主线契约。 |
| `soft-beam-label-training` | `current` | post-C2 保留/主线契约。 |
| `spec-lifecycle-boundaries` | `supporting` | 支撑治理或只读辅助，不是独立主线入口。 |
| `target-shot-domain-splitting` | `supporting` | 由 `data/target_shot_splits.py` 为 MMW protocol 提供 split provenance 与防泄漏支撑；无独立 CLI。 |
| `t2-beam-geometry-head-validation` | `supporting` | 已取消的 DeepSense geometry/head 实验契约与防误报证据，不是默认主线。 |
| `temporal-window-missing` | `current` | 显式窗口、temporal missing mask 和 H5/P1 matrix 契约。 |
| `tinyvit-image-encoder` | `current` | post-C2 保留/主线契约。 |
| `training-evaluation-runtime` | `current` | post-C2 保留/主线契约。 |
| `u-mask-beam-jepa` | `current` | post-C2 保留/主线契约。 |
| `u-mask-beam-jepa-eval-matrix` | `current` | post-C2 保留/主线契约。 |

## Agent Context Registration

Scoped context 文件：`docs/agent_context/README.md`、`docs/agent_context/models.md`、`docs/agent_context/data.md`、`docs/agent_context/configs.md`、`docs/agent_context/cli.md`、`docs/agent_context/diagnostics.md`、`docs/agent_context/openspec.md`、`docs/agent_context/documentation.md`、`docs/agent_context/claims.md`、`docs/agent_context/atlas.md`。

项目技能：`.codex/skills/kd-add-model/SKILL.md`、`.codex/skills/kd-add-config/SKILL.md`、`.codex/skills/kd-update-claim/SKILL.md`、`.codex/skills/kd-diagnose-run/SKILL.md`、`.codex/skills/kd-archive-change/SKILL.md`。

Portable docs：`CLAUDE.md`、`.github/copilot-instructions.md`、`.cursor/rules/kd-sensing-context.mdc`、`.kiro/steering/agent-context.md`、`docs/current_research_brief.md`、`docs/readonly_agent_roles.md`、`docs/agent_memory_ledger.md`、`docs/agent_project_knowledge.md`。

退役文本标记如 HiST-Beam、Top8 selector、GPS residual、camera residual、Raymobtime s008、BGAM、viewer manifest、Gradio viewer、CRAF、MARF、Multimodal-NF 只能在退役、历史、拒绝、防回流或 tombstone 语境出现。
