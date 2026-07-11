## 1. 基线、保护清单与 active change 收口

- [x] 1.1 记录 `git status --short`、`git diff -- scripts/launch_h5_p1_temporal_models_v1.py`、13 个 console scripts、tracked `src/tests/scripts/specs` 文件数与行数；在本 change 的一次性 `deletion-ledger.md` 中登记基线，不读取或修改 `outputs/`、`logs/`、dataset、cache 或 checkpoint。
- [x] 1.2 在 deletion ledger 中为每个删除组记录 tracked source consumer、config/CLI/spec/claim consumer、替代 owner、focused validation 和 rollback wave；记录 protected paths，至少包含 U-Mask/final C2（含 `reliability_biased_missing_attention`、prototype alignment、full-to-partial teacher）、MMW/CSI/physics、Scene31-34 final analysis、runtime cleanup、H5/P1、RMBP core、target-shot、run-index resources 与 instance/startup architecture summary。
- [x] 1.3 收缩 completed `align-paper-baselines-window2` 的 proposal/design/spec/tasks，使其只描述真实存在的 AMBER/AMR + window 2/1 内容，或明确 abandon；不得恢复 WCL CLI/config/spec、旧 local-baseline surface 或删除 H5/P1 正在使用的 `rmbp_mm`。
- [x] 1.4 收缩 `add-temporal-window-missing` 的 proposal/design/spec/tasks：保留 history/prediction aliases、temporal difficulty 和 H5/P1 matrix，移除旧 temporal check/launch/summary 与未完成 S1-S4 scope，并把未来 S1-S4 触发条件记录为新 change 而非本 change 延期任务。
- [x] 1.5 运行 `openspec validate align-paper-baselines-window2 --strict`、`openspec validate add-temporal-window-missing --strict` 和 `openspec validate consolidate-post-c2-current-surface --strict`；任一 active artifact 与实际保留面冲突时停止后续删除。
- [x] 1.6 保存用户 dirty H5/P1 launcher 的 diff 摘要，并在每个删除 wave 后确认 `scripts/launch_h5_p1_temporal_models_v1.py` 的 `--auto-resume` 改动仍存在且未被格式化或覆盖。

## 2. Temporal 与历史 launcher 收敛

- [x] 2.1 删除 `check_temporal_window_missing.py`、`launch_temporal_missing_v1.py` 和 `summarize_temporal_missing_v1.py`，同步 inventory/docs/tests 引用；保留 `tests/test_temporal_window_missing.py` 作为核心行为验证。
- [x] 2.2 删除 S1-S4 三个 scripts、`tests/test_temporal_router_s1_s4_v1.py` 以及只服务 `temporal_router_type`/S1-S4 gate/oracle 的 U-Mask model、loss、config 和 diagnostics branches；不得改变 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 或 temporal mean 默认语义。
- [x] 2.3 删除 `scripts/launch_overnight_branch_router_v2.py` 与测试中的 launcher dry-run/GPU scheduling assertions；保留 `summarize_overnight_branch_router_v2.py` 及 final C2 对它的直接 import 和 parser tests。
- [x] 2.4 更新 script inventory、mainline catalog 和 overnight lifecycle，使 retained H5/P1 与 overnight summary 标记为 local/manual 或 supporting，删除路径不再作为 current command。
- [x] 2.5 运行 `conda run -n kd_mm_beam pytest tests/test_temporal_window_missing.py tests/test_h5_p1_temporal_matrix_v1.py tests/test_u_mask_beam_jepa.py tests/test_prediction_objectives.py tests/test_overnight_branch_router_v2.py tests/test_final_c2_ablation_v1.py -q`。
- [x] 2.6 运行 `conda run -n kd_mm_beam python scripts/verify_compile.py` 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`；确认不再有 S1-S4 wrapper failure 后才进入下一 wave。

## 3. Research dashboard、harvester 与旧 Scene31 owner 退出

- [x] 3.1 从 `pyproject.toml` 删除 `kd-sensing-research-dashboard` 与 `kd-sensing-research-preview`，删除对应 CLI、HTML renderer、preview owner 和专属 tests/docs；不得新增替代 dashboard、HTML product、Web service 或 wrapper。
- [x] 3.2 删除 `research_claim_harvester.py`、`research_claim_harvester_base.py`、`research_claim_harvester_collectors.py`、`research_claim_harvester_dashboard.py`、`research_claim_harvester_gate.py`、`research_claim_harvester_writers.py` 和 `tests/test_research_claim_harvester.py`；保留人工 claim registry、formal protocols、run index/run card 与 paper export。
- [x] 3.3 从 run index 移除 harvester/dashboard 专属命名或 downstream assumptions，但完整保留 `run_index_resources.py` 的 PID discovery/matching、memory/GPU snapshot、稳定 `resources` schema 与 cleanup active-run protection。
- [x] 3.4 删除 `src/kd_sensing/diagnostics/scene31_summary/` 整个旧 owner 和只服务 baseline-pack/next-round 的 tests/docs/config/script references；保留 `scene31_34_final_analysis/`、`scene31_eval_resolution.py`、Scene31-34 generator/runner、AMR-lite/AMBER-lite external rows 和 current paper evidence。
- [x] 3.5 将 `local-missing-modality-baselines` inventory 改为 supporting，只保留 Scene31-34 `amr_lite` build/mask/gate contract；AMBER-lite、AMBER full 和 AMR-Net 继续由各自 current specs 管理，不恢复 FeatureMod 或旧 baseline pack。
- [x] 3.6 运行 `conda run -n kd_mm_beam pytest tests/test_run_index.py tests/test_runtime_artifact_cleanup.py tests/test_paper_artifact_export.py tests/test_scene31_34_final_analysis.py tests/test_scene31_34_encoder_ablation.py tests/test_missing_modality_statistics.py tests/test_missing_modality_stress.py tests/test_amr_net.py tests/test_amber_lite_missing_modality.py tests/test_amber_full_architecture.py -q`。
- [x] 3.7 增加并运行 active-run cleanup 回归：`conda run -n kd_mm_beam pytest tests/test_run_index.py tests/test_runtime_artifact_cleanup.py -q` 必须证明有匹配 live PID 的超时长 run 仍为 `running` 且具有 `run_state_running` protection。

## 4. Training-I/O、LOSO 与窄 orphan owner 删除

- [x] 4.1 删除 `engine/training_io_profile.py` 与 `engine/throughput_recommendations.py` 的 standalone profiler/recommendation 产品面；保留 `engine.run_metadata.throughput_run_metadata()` 及 trainer/evaluator 调用。
- [x] 4.2 在五个 `tests/test_training_io_*` 文件中只删除 profiler/recommendation imports 和专属 recommendation/profile assertions，保留 dataset、cache policy、trainer/evaluator、label 与 run-metadata coverage；不得按 glob 删除这些测试文件或 `tests/training_io_helpers.py`。
- [x] 4.3 删除零 consumer 的 `data/loso.py` 与专属 tests/spec wording；保留 `data/mmw/protocol.py`、`data/target_shot_splits.py`、MMW preparation/support-selection、split provenance、leakage guards 与 target-shot artifact contract。
- [x] 4.4 删除确认零 consumer 的 `utils/geometry.py`、`utils/checkpoint_resolver.py`、duplicate `models/mmw_town_gps_v2.py`、test-only AMBER-lite facade 和 `evaluation/physics_metrics.py`，逐项同步 import/tests；`eval/missing_patterns.py` 因 current consumer 已记录 `retained-with-evidence` 并从删除候选移除。
- [x] 4.5 收缩 `models/architecture_summary.py` 到 instance parameter/component/trainability 与 startup artifact 所需函数，删除 standalone CLI、candidate/sweep/config preflight、Markdown/CSV renderer 和独立 report branches；保留 U-Mask/AMR/AMBER tests 与 Scene31-34 profile 消费的 schema。
- [x] 4.6 将 `target-shot-domain-splitting` 与 `model-architecture-summary` lifecycle 从 `retired-tombstone` 改为 `supporting`，更新 inventory/mainline/maintainer context，且不恢复对应 standalone CLI。
- [x] 4.7 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_cache_workflow.py tests/test_training_io_dataset_workflow.py tests/test_training_io_label_workflow.py tests/test_training_io_run_metadata.py tests/test_training_io_workflow.py -q`。
- [x] 4.8 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmw_town_gps_adapter_v2.py tests/test_u_mask_beam_jepa.py tests/test_amr_net.py tests/test_amber_full_architecture.py tests/test_component_registry.py tests/test_prediction_objectives.py tests/test_physics_informed_mmw.py -q`；已删除专属 test 不得用空 stub 替代。

## 5. GPS-query、Scenario-D 与 JEPA downstream 收缩

- [x] 5.1 将 `jepa_downstream.py` 和 `jepa_downstream_helpers.py` 收缩为 current MMW config 所需的 JEPA context mean pooling/checkpoint extraction；删除 GPS/content/hybrid/predictive/query-weighted/K-token/per-head pooler、adapter registry 和专属 metadata，不删除 JEPA pretraining、visual token encoder、mask sampler、EMA、latent loss或 checkpoint loading。
- [x] 5.2 删除 `diagnostics/gps_query_evidence.py` 及 query/attention/advantage 专属 tests、docs 和 registry entries；retired pooler/config 必须 unknown/fail-fast，不能静默映射到 mean。
- [x] 5.3 从 batch、modular forward、optimizer、prediction objectives 和 run metadata 删除只服务 predictive GPS-query latent/gate/attention 的 branches，保持普通 supervised、U-Mask、AMR/AMBER 和 MMW mean-context输出兼容。
- [x] 5.4 从 difficulty presets/schema/operators 删除 Scenario-D D0-D7、P0-P5 predictive、visual hard-negative、beam-offset wrong-GPS、CxD advantage 和 shortcut benchmark/cache compatibility；保留通用 GPS jitter/drift/delay/dropout、image degradation/observability、missing-stress、determinism、digest 和 no-future-leak。
- [x] 5.5 裁剪 `tests/test_gps_conditioned_jepa.py` 与 `tests/test_modality_difficulty.py` 中 retired query/predictive/D-level tests，保留 pretraining、mean pooling、checkpoint、visual-token、generic operator、missing-stress 和 target-preservation tests。
- [x] 5.6 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py tests/test_modality_difficulty.py tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_optimizer_param_groups.py tests/test_prediction_objectives.py tests/test_physics_informed_mmw.py -q`。
- [x] 5.7 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_mmw_town10_preparation.py tests/test_architecture_boundaries.py -q` 证明 MMW `jepa_context_image pooling: mean`、MMW geometry/preparation 与 import boundaries 未被误删。

## 6. Geometry prior 与 safe reranker 原子删除

- [x] 6.1 删除 `models/geometry_prior.py`、`diagnostics/geometry_prior_beam_fusion.py`、默认 registry imports 及 `tests/test_geometry_prior_beam_fusion.py`；普通 unknown-name 行为承接旧 component 拒绝，不新增 removed-name framework。
- [x] 6.2 从 `models/modular.py`、`models/modular_forward.py`、`engine/batch.py`、`engine/prediction_objectives.py`、`engine/run_metadata.py` 和 architecture summary 中删除 geometry prior/logit fusion/safe reranker construction、forward、loss、diagnostics 与 metadata attachments。
- [x] 6.3 从 `tests/test_modular_sequence_next_query_transformer.py`、`tests/test_prediction_objectives.py`、`tests/test_modality_difficulty.py` 和 fixtures 删除 geometry/rerank 专属 cases，保留 next-beam-query Transformer、generic modular stages、MMW geometry、physics beam scoring 和 current supervised losses。
- [x] 6.4 更新 component registry、model inventory、mainline docs 和 retired guard，使 geometry/safe-rerank names 不再 current；不得删除 `data/datasets/mmw_geometry.py`、`data/mmw/`、`models/physics/` 或 MMW cross-scene geometry metadata。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_component_registry.py tests/test_prediction_objectives.py tests/test_modality_difficulty.py tests/test_physics_informed_mmw.py tests/test_mmw_town10_preparation.py tests/test_architecture_boundaries.py -q`。

## 7. Current specs、文档与治理镜像收敛

- [x] 7.1 验证 design 固定清单的 24 个 capability 均有 all-requirements `REMOVED` delta，逐项确认 deletion ledger 无 current consumer/独立 guard；apply 阶段不得直接删除 `openspec/specs/` 目录，物理折叠留给独立 archive。
- [x] 7.2 将 target lifecycle 中的 `target-shot-domain-splitting`、`model-architecture-summary`、`local-missing-modality-baselines` 明确改为 `supporting`；任何新例外必须有调用路径、owner 和 focused test，post-archive current specs 预期 81 个且不得超过 84 个。
- [x] 7.3 更新 README、`docs/current_research_brief.md`、mainline catalog、experiment matrix/protocols、result claims、agent context 与 `docs/project_surface_inventory.md`，统一为 final C2/U-Mask 主线、MMW/CSI supporting、AMR/AMBER controls 和十个 retained CLI；删除 dashboard/preview/query/geometry/old Scene31 current commands。
- [x] 7.4 收缩 `docs/maintainer_context_index.yaml`，只保留不能从 pyproject、source 和 inventory 推导的 route id、scoped context、authority、protected paths 与最小 validation；删除 CLI/script/file-count/hotspot/source-tree 镜像。
- [x] 7.5 将 architecture boundary tests 中逐文件 deleted-path、固定文案、完整 lifecycle 和 maintainer-index mirrors 改成 source-of-truth 驱动的结构断言；保留 retired token/path、tracked runtime artifact、light import、CLI/config 和 protected owner guards。
- [x] 7.6 运行 `openspec validate consolidate-post-c2-current-surface --strict` 和 `openspec validate --all --strict`；再运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py tests/test_retired_routes.py -q`。

## 8. Surface doctor 最后退出与十个 CLI 定稿

- [x] 8.1 在删除 doctor 前，用现有 doctor 对前述 waves 做最后一次只读 stale-reference 检查；只把 secret/system-config/dangerous-shell-runner 这一独立安全语义迁入现有 architecture/focused test，使用标准库且不新增 doctor/report framework。
- [x] 8.2 为安全 guard 增加真实危险 fixture，证明训练命令写入 `/root/.container_env` credential 字段、system profile 或认证配置会失败；测试不得修改真实系统文件。
- [x] 8.3 删除 `diagnostics/project_surface_doctor.py`、对应 CLI、`tests/test_project_surface_doctor.py`、Make/README/docs/inventory/current spec references；不得保留 alias、stub、JSON report schema 或替代命令。
- [x] 8.4 将 `pyproject.toml` 和 CLI help characterization 定稿为十个 console scripts：train、evaluate、preprocess、runs、clean、organize、paper export、U-Mask matrix、MMW GPS v2、MMW physics inspect。
- [x] 8.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py tests/test_config_load_characterization.py tests/test_run_index.py tests/test_runtime_artifact_cleanup.py -q` 和 `conda run -n kd_mm_beam python scripts/verify_compile.py`。

## 9. 最终验收、计数与独立 archive 准备

- [x] 9.1 运行 `make verify-quick`、`make verify-cli-config` 和 `make verify-compile`，失败时回到对应 wave 修复，不新增永久 allowlist 或 compatibility wrapper。
- [x] 9.2 运行 `openspec validate --all --strict`、`openspec status --change consolidate-post-c2-current-surface`，确认本 change artifacts 完整且另外两个 active changes 的收口状态与实现一致。
- [x] 9.3 运行 `conda run -n kd_mm_beam pytest -q` 完成全量回归；记录 skipped/failed 原因，任何 protected workflow regression 未解决时不得把本 change 标记完成。
- [x] 9.4 重新统计 tracked `src/kd_sensing`、tests、scripts、effective post-archive specs 与 console scripts；`src/kd_sensing` 净减目标不少于 9,000 行、post-archive spec 预期 81 且最多 84、public CLI 必须恰为 10，低于净删目标时逐项记录 retained-with-evidence 理由。
- [x] 9.5 检查 `git status --short` 和 `git diff --check`，确认没有 dataset、outputs、logs、cache、checkpoint、TensorBoard、`All_models/` 或 generated runtime artifact，且用户 H5/P1 `--auto-resume` diff 原样保留。
- [x] 9.6 输出最终 implementation summary：逐 wave 删除量、保留例外、验证命令、known caveat 和 rollback 点；只将三个 changes 标记为 ready-to-archive，不在本 change 中执行 archive，归档必须走独立 archive workflow。
