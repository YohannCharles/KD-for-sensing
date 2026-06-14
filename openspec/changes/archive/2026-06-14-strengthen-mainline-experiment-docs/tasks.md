## 1. OpenSpec 规格收束

- [x] 1.1 清理 `openspec/specs/experiment-workflow/spec.md` 中仍把 KD、teacher/student、`kd_mode`、Fusion KD 或旧 no-KD 路径描述为当前 active workflow 的 requirement/scenario。
- [x] 1.2 保留并核对旧 KD、Hist、Top8 standalone、GPS residual、camera residual、Raymobtime s008 等路线的拒绝、退役、supporting 或 migration guard 语义。
- [x] 1.3 将 `mainline-experiment-documentation` 新能力归档到当前 specs，并在 `docs/project_surface_inventory.md` 的 lifecycle inventory 中标记为 `current`。
- [x] 1.4 更新 `spec-lifecycle-boundaries` 或相关 current specs，使 current spec 内部冲突被描述为规格漂移并要求通过 OpenSpec change 收敛。

## 2. 当前主线文档

- [x] 2.1 新增 `docs/mainline_model_catalog.md`，列出 Image+GPS JEPA、paired controls、Vision-Position baselines、Camera AE+GPS Direct、BEV-Fusion 2604、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、viewer/diagnostic/benchmark 的模型目录。
- [x] 2.2 新增 `docs/experiment_protocols.md`，为主要配置族记录 formal、lowmem、smoke、debug、evaluation-only、upper-bound、historical ablation 或 mock 状态，以及 seed、epoch、batch、lr、split、target、metric 和输出边界。
- [x] 2.3 新增 `docs/result_claims_registry.md`，记录当前可引用结果、blocked official 状态、本地 substitute、strict-validation、upper-bound、mock/smoke 和 historical ablation 的 claim provenance。
- [x] 2.4 更新 README 和 `docs/experiment_matrix.md`，将详细表格职责转交给三份 current 文档，只保留 quickstart、推荐顺序和关键 caveat。
- [x] 2.5 更新 `docs/project_surface_inventory.md` 的文档生命周期分类，登记新增文档职责，避免 README、inventory、experiment matrix 和新文档边界重叠。

## 3. BeamBench/Arnold22 报告治理

- [x] 3.1 重排 `BASELINE_REPORT.md`，在开头提供 current summary，明确 Table III 本地 substitute 的当前口径为 `beam_target_source=current`、`seq_len=1`、`num_pred=1`、`paper_distance_angle` 和 linear DBA。
- [x] 3.2 将 `BASELINE_REPORT.md` 中旧 `future` target、旧 GPS 公式、旧 AE 维度、`test_as_validation`、scene31-only、dry-run 和 mock 记录标记为 historical ablation、upper-bound、smoke 或 mock。
- [x] 3.3 更新 `README_REPRODUCE.md`，确保当前推荐命令使用 current target，并指向 current summary 和结果账本。
- [x] 3.4 更新 `results/reproduce_baseline.md` 开头说明其为历史流水账，不能覆盖 current summary，并为高风险旧命令补充不可作为当前正式结果的 caveat。

## 4. 配置族 README 和协议补齐

- [x] 4.1 更新 `configs/fusion/experiments/jepa_image_gps/README.md`，补充主线/对照/2604/BeamBench-fair/GPS-query 的协议状态和结果账本引用。
- [x] 4.2 为 `configs/fusion/experiments/bev_fusion_2604/` 增加或更新 README，区分 `paper_full`、`low_memory`、`smoke` 和 ablation 的口径、参数、metric 和 caveat。
- [x] 4.3 补充 MMW GPS v2、MMW BGAM、DeepSense6G BGAM、CSI hardening、difficulty/benchmark profile 的协议表引用，明确哪些配置是 formal、quick validation、debug 或 diagnostic-only。
- [x] 4.4 检查主要 YAML 注释与新协议表一致，避免 YAML 注释把 historical、smoke 或 upper-bound 口径写成正式结果。

## 5. 健康护栏和测试

- [x] 5.1 扩展 `tests/test_architecture_boundaries.py` 或新增轻量文档健康测试，验证新增文档存在、README 索引存在、lifecycle inventory 包含 `mainline-experiment-documentation`。
- [x] 5.2 增加 high-risk wording 检查，覆盖未加限定的 `target-beam-source future`、`target_beam_source: future`、`test_as_validation`、mock/smoke/upper-bound 结果和旧 KD active wording。
- [x] 5.3 确保文档健康检查只读取已跟踪源码、配置、文档和 OpenSpec artifact，不读取 `dataset/`、`outputs/`、checkpoint、cache、metrics 或 logs。
- [x] 5.4 如检查使用 Python，验证命令必须写成 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 或对应 focused pytest 命令。

## 6. 验证和收尾

- [x] 6.1 运行 `openspec validate strengthen-mainline-experiment-docs --strict` 并修复所有 OpenSpec 格式或 requirement 问题。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证文档/架构边界。
- [x] 6.3 如修改 README、CLI 索引或配置加载说明，追加 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 6.4 检查 `git status --short`，确认没有把 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、metrics、figures 或 TensorBoard 产物纳入源码变更。
- [x] 6.5 在最终说明中列出新增文档、已清理的规格漂移、运行过的验证命令和未运行项原因。
