## 1. 保护清单与删除证据

- [x] 1.1 运行 `openspec list --json`、`openspec status --change final-c2-ablation-v1 --json` 和 `openspec status --change overnight-branch-router-v2 --json`，记录 complete/active 状态，不在本 change 中改写已完成成果。
- [x] 1.2 生成 post-C2 protected inventory，覆盖 final C2 / U-MaskBeamJEPA 主线、MMW/CSI workflow、主线 YAML/manifest、claim/evidence 输入和 U-MaskBeamJEPA fusion 分支实现。
- [x] 1.3 运行只读表面积扫描：`conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope scripts --scope configs --scope hotspots --format markdown --fail-on none`，将候选分为 protected、delete-candidate、historical、supporting、pending-confirmation。
- [x] 1.4 为每个 delete-candidate 记录当前引用、公开入口风险、替代入口、验证命令和回滚方式，形成 deletion candidate ledger。
- [x] 1.5 标记不确定是否主线会用到的 YAML/manifest 为 `pending-confirmation` 或 `protected-until-next-audit`，本 change 不删除。

## 2. 文档与入口生命周期收口

- [x] 2.1 更新 README、`docs/current_research_brief.md`、`docs/mainline_model_catalog.md`、`docs/experiment_matrix.md` 和 `docs/experiment_protocols.md`，将当前默认主线收敛为 final C2 / U-MaskBeamJEPA 缺失模态，并明确 MMW/CSI 保留。
- [x] 2.2 将非主线 Image+GPS JEPA、BeamBench、BEV-Fusion 2604、Vision-Position、旧 RBMA/KD/BTAPA/weakKD sweep 从 current recommended workflow 降级为 historical、supporting 或删除候选说明。
- [x] 2.3 审计 `pyproject.toml` console scripts，删除或降级非主线 CLI；保留核心训练/评估/预处理、MMW/CSI、final C2 或缺失模态评估、必要治理/claim 入口。
- [x] 2.4 更新 `src/kd_sensing/diagnostics/cli_surface.py`、README/docs 和 CLI help smoke，使保留 CLI 都有 owner、输出边界和 focused validation。
- [x] 2.5 运行 `conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope cli-surface --format markdown --fail-on error`，确认 current docs 不引用已删除 public CLI。

## 3. 一次性 scripts 与 runbook 删除

- [x] 3.1 删除已确认只服务历史 sweep 或人工复盘的 `scripts/analyze_*`、`scripts/summarize_*`、`scripts/diagnose_*` 和旧 Scene31 shell runbook。
- [x] 3.2 保留或登记 final C2、当前缺失模态主线、MMW 数据准备/诊断和 protected YAML/manifest 消费的脚本。
- [x] 3.3 删除脚本前把仍有价值的结论迁移到 `docs/mainline_experiment_history.md`、`docs/result_claims_registry.md`、inventory historical note 或等价文档。
- [x] 3.4 更新 `docs/project_surface_inventory.md` 和架构边界测试，使删除后的 `scripts/` surface 不再依赖完整旧 allowlist。
- [x] 3.5 运行 `make verify-compile`，确认剩余 tracked scripts/package CLI 可编译且不引用已删除脚本。

## 4. 非主线源码与测试收缩

- [x] 4.1 删除已确认非主线且无 protected 引用的 BeamBench、BEV-Fusion 2604、Vision-Position、Image+GPS JEPA 诊断源码和对应 tests。
- [x] 4.2 保留 MMW/CSI/physics-informed MMW 源码、configs、CLI 和 tests；若修改其 docs 或 guardrail，追加 MMW focused tests。
- [x] 4.3 保留 `src/kd_sensing/models/u_mask_beam_jepa.py` 与 U-Mask loss/config 中既有 fusion/router/loss 分支；只在 inventory 中记录后续单独审计触发条件。
- [x] 4.4 删除源码后更新 imports、registry default imports、tests、docs 和 OpenSpec current specs，确保没有 stale module reference。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认退役模块不回流且 protected paths 仍存在。

## 5. 配置与 manifest 收缩

- [x] 5.1 审计 `configs/` 中 Scene31、RBMA、KD/BTAPA/weakKD/tau/seed、historical sweep 和非主线复现配置，按 protected inventory 决定保留、删除、生成化或 pending-confirmation。
- [x] 5.2 删除不被 current claim/evidence、final C2、MMW/CSI、focused tests、OpenSpec current spec 或用户主线标记消费的 historical YAML/manifest。
- [x] 5.3 对可由 generator/template 无损重建的实体 YAML，保留 generator/template/manifest 输入或把复跑路径迁移到 docs。
- [x] 5.4 更新 `docs/result_claims_registry.md`、`docs/experiment_protocols.md` 和 `docs/mainline_model_catalog.md`，确保 claim provenance 不指向已删除配置。
- [x] 5.5 运行 `make verify-cli-config`，确认 config load、migration guard 和 CLI/config smoke 仍通过。

## 6. Guardrail 与最终验证

- [x] 6.1 更新 `tests/test_architecture_boundaries.py` 和 project surface doctor 规则，检查 protected MMW、protected YAML/manifest、U-Mask fusion 分支暂不删除、stale references 和退役入口回流。
- [x] 6.2 运行 `openspec validate prune-post-c2-nonmainline-surface --strict`。
- [x] 6.3 运行 `openspec validate --all --strict`。
- [x] 6.4 运行 `make verify-quick`、`make verify-cli-config` 和 `make verify-compile`。
- [x] 6.5 若本 change 触碰 MMW/CSI 相关 docs、configs、CLI lifecycle 或 guardrail，运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_mmw_town_gps_adapter_v2.py tests/test_csi_modality.py tests/test_physics_informed_mmw.py -q` 或记录未运行原因。
- [x] 6.6 最终确认 `git status --short` 中没有 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或训练产物被纳入源码变更。
