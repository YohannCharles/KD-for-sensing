## 1. Baseline 与候选分类

- [x] 1.1 运行 `conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope configs --scope hotspots --format markdown --fail-on none`，保存本 change 范围内的未分类 scripts/configs/hotspots 清单到实现说明或 inventory 草稿。
- [x] 1.2 对未分类脚本逐项标记为 current、secondary/supporting、local/manual、historical、retired、delete-candidate 或 merge-candidate，并记录 owner、输入输出边界、公开入口风险、替代入口和删除触发条件。
- [x] 1.3 对未分类配置逐项标记为 current/evidence、generated、local/manual、historical 或 delete-candidate，并检查 README、docs、OpenSpec、tests 和 scripts 的当前引用。
- [x] 1.4 对未登记热点逐项标记为 accepted、monitor、split-next、merge-candidate 或 deleted，并记录预算理由与 focused 验证命令。

## 2. Scene31-34 Encoder Ablation 收敛

- [x] 2.1 比较 `scripts/generate_scenes31_34_tinyvit_ablation.py` 与 `scripts/generate_scenes31_34_patchvit_ablation.py`，提取真正不同的 encoder family 参数。
- [x] 2.2 实现一个统一的 Scene31-34 encoder ablation generator owner，覆盖 TinyViT 与 PatchViT 的 YAML 生成或 dry-run 输出。
- [x] 2.3 删除或降级旧重复 generator，确保 docs、tests 和 inventory 不再同时推荐两个同构入口。
- [x] 2.4 将 `scripts/run_scenes31_34_tinyvit_ablation.sh` 收敛为 family/manifest 驱动 runner，或登记为唯一 local/manual owner；不得新增 PatchViT 专用 shell runner。
- [x] 2.5 更新 encoder ablation focused tests，至少覆盖 TinyViT 与 PatchViT 的最小 manifest、生成路径和无固定 GPU 假设。

## 3. Final/Presentation/Historical Helper 生命周期

- [x] 3.1 审计 `scripts/export_scene31_34_presentation_artifacts.py` 和 `scripts/run_final_scene31_34_polish.sh` 的 current 引用；若仍服务当前交付则登记 local/manual analysis helper，否则删除脚本和专属测试。
- [x] 3.2 审计 Scene31/Scene31-34 paper table、per-scene summary、final conclusion helper，指定一个 current final export owner 或将其余 helper 标记为 historical/local/manual。
- [x] 3.3 删除已经沉淀结论且无 current 引用的一次性报告脚本，并把仍有价值的结论、caveat 或替代入口写入 docs 或 provenance 文档。
- [x] 3.4 更新 `docs/project_surface_inventory.md`、`docs/mainline_model_catalog.md`、相关 scoped context 和引用文档，确保 final/presentation/historical helper 不冒充 current workflow。

## 4. Scene31 与 RBMA/KD/BTAPA 配置收缩

- [x] 4.1 审计 `configs/scene31/`、`configs/fusion/experiments/rbma_missing_workflow/` 和 `configs/fusion/experiments/rbma_missing_workflow_strong_encoders/`，标记 current/evidence、generated、local/manual、historical 和 delete-candidate。
- [x] 4.2 对可由 generator、template 和 manifest 无损重建的 Scene31/Scene31-34 实体 YAML，删除 tracked YAML 或将其登记为生成产物边界。
- [x] 4.3 对不再被 current docs、OpenSpec、tests、claim provenance 或实验矩阵引用的 RBMA/KD/BTAPA/weakKD/tau/seed/PatternFiLM overlay，删除或降级为 historical/local manual。
- [x] 4.4 对仍被 claim 或复现表格引用的配置，保留实体 YAML 或提供等价 manifest/generator 输入，并更新 provenance 说明。
- [x] 4.5 更新配置加载和路径引用测试，确保删除后的 current docs/OpenSpec/scripts 不指向不存在配置。

## 5. Hotspot Governance 与防回流

- [x] 5.1 在 hotspot inventory 或等价文档中登记 `src/kd_sensing/diagnostics/gps_query_evidence.py`、`src/kd_sensing/engine/run_metadata.py`、`src/kd_sensing/models/u_mask_beam_jepa.py` 的状态、职责边界、headroom 和后续 split-next 条件。
- [x] 5.2 若实现阶段低风险拆分热点，只拆分职责清晰且不改变公开语义的 helper，并用 focused tests 覆盖正常路径、错误路径和 current config/dry-run 场景。
- [x] 5.3 更新 doctor 或 architecture boundary 检查，使未登记大 owner、重复 encoder generator、PatchViT 专用 runner 和 fixed-GPU shell 回流会被发现。
- [x] 5.4 保留 U-MaskBeamJEPA 当前模型/loss/run metadata 公共语义；任何 branch 删除必须先证明对应配置、spec、tests 和 claim provenance 不再消费该 branch。

## 6. 文档与测试同步

- [x] 6.1 更新 `docs/project_surface_inventory.md`，同步 scripts、configs、hotspots 的分类、删除项、保留理由和替代入口。
- [x] 6.2 更新 `docs/current_research_brief.md`、`docs/experiment_matrix.md`、`docs/mainline_model_catalog.md` 或 scoped context 中受影响的主线/secondary/local/manual 描述。
- [x] 6.3 更新 `tests/test_architecture_boundaries.py`、surface doctor 相关测试和 encoder ablation focused tests，避免维护大 allowlist，同时保留旧入口回流、tracked runtime artifact、current path/config 引用失效检查。
- [x] 6.4 删除或更新只服务被删除入口的测试；不得为旧入口新增兼容 wrapper 或 virtual alias 来让旧测试通过。

## 7. 验证与收口

- [x] 7.1 运行 `openspec validate prune-missing-modality-mainline-surface --strict`。
- [x] 7.2 运行 `openspec validate --all --strict`。
- [x] 7.3 运行 `conda run -n kd_mm_beam python -m kd_sensing.cli.project_surface_doctor --scope scripts --scope configs --scope hotspots --format markdown --fail-on warning`。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.5 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 7.6 运行 encoder ablation 相关 focused tests，使用实现后的统一 test 文件名；若重命名测试，最终说明必须列出实际命令。
- [x] 7.7 运行 `conda run -n kd_mm_beam python scripts/verify_compile.py`。
- [x] 7.8 在最终实现说明中列出删除/合并/保留清单、未运行验证原因、剩余风险和后续可选 cleanup change。
