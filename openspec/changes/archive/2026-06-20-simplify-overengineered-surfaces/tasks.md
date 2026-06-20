## 1. 契约和治理表同步

- [x] 1.1 更新 README、AGENTS、`docs/agent_navigation.md` 和相关 workflow docs，将当前训练、评估、预处理和 BeamBench 命令改为 `kd-sensing-*` console scripts。
- [x] 1.2 更新 `docs/maintainer_context_index.yaml` 的 package CLI、script allowlist、hotspot、merge-candidate、dependency audit 和 remediation wave metadata。
- [x] 1.3 更新 `docs/project_surface_inventory.md`，记录 deleted、merged、base+overlay、remove-internal-only 和 no-current-surface 分类。
- [x] 1.4 更新当前 OpenSpec specs 或引用说明，确保不再把 `scripts/*.py` thin alias、孤立诊断模块、未接入 LiDAR pillar 原型或重复 CSI YAML 写成 current required surface。

## 2. 入口和依赖删减

- [x] 2.1 删除 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、`scripts/check_dataset.py`、`scripts/eval_baseline.py`、`scripts/train_baseline.py`、`scripts/train_beambench_image_ae_gps.py` 和 `scripts/run_beambench_image_ae_gps_tableiii.py` thin alias。
- [x] 2.2 更新 CLI help tests 和架构入口 allowlist，验证 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 和保留的 BeamBench console scripts 可用。
- [x] 2.3 从 `pyproject.toml` 的 dev extra 删除未使用的 `thop` 和 `pytorch-model-summary`。
- [x] 2.4 确认 runtime dependencies 不变，并更新任何提到已删除 dev deps 的文档或测试。

## 3. 源码表面收敛

- [x] 3.1 删除 `src/kd_sensing/diagnostics/communication_state_features.py` 和只服务它的测试。
- [x] 3.2 删除未接入的 `src/kd_sensing/models/lidar_pillar_encoder.py`，并清理相关测试、导出或索引引用。
- [x] 3.3 收缩 `src/kd_sensing/data/dataset_runtime.py`：保留或迁移仍被消费的轻量 row 类型，删除未消费的 `RuntimeDataset`、adapter framework 和 index writer。
- [x] 3.4 合并两份重复 `OutputRegistry`，保留一个 owner helper 或改为局部 `list_outputs(root, skipped)` 函数。
- [x] 3.5 运行结构扫描或 CodeGraph 检查，确认删减对象没有 current CLI、registry、docs、OpenSpec 或内部调用残留。

## 4. JEPA benchmark facade 收窄

- [x] 4.1 调整 `jepa_gps_shortcut_benchmark.py`，只保留公开 runner/API、manifest loading、公开常量和 CLI 所需导入。
- [x] 4.2 将测试中对 facade `_private_helper` 的引用迁移到 helper 所在窄模块。
- [x] 4.3 更新架构边界测试，防止 benchmark facade 重新导出 private helper 或超过维护索引声明预算。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest` 的 JEPA GPS shortcut benchmark focused tests。

## 5. CSI hardening 配置矩阵去重

- [x] 5.1 选择最小实现：base config + overlay YAML、recipe table 或复用现有 virtual config resolver。
- [x] 5.2 将 `configs/csi/hardening_matrix/` 和 `configs/fusion/csi_hardening_matrix/` 中重复完整 YAML 收敛为 base+overlay/recipe 表达。
- [x] 5.3 保持 A0/A1/A2/B3/B4/B5/B6/C1/C2/D1/D2/D3/D4/E0/E1/E2/E3 配置 ID 可加载。
- [x] 5.4 添加或更新 focused tests，验证 A2 destructive negative control、D 组非 destructive、E 组 GPS+CSI 语义和 resolved config metadata。
- [x] 5.5 更新 README/docs/OpenSpec/维护索引中的 CSI hardening 配置路径引用。

## 6. 架构健康护栏收缩

- [x] 6.1 精简 `tests/test_architecture_boundaries.py` 中逐字断言 README/docs/OpenSpec prose 的测试。
- [x] 6.2 保留并加强机器可读检查：维护索引 schema、entrypoint 双向同步、路径存在性、lifecycle、retired wording、AST/import 边界和本地产物边界。
- [x] 6.3 更新 `tests/helpers/maintainer_context.py`，删除只为 prose mirror 服务的 helper。
- [x] 6.4 确认健康检查不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event。

## 7. 验证

- [x] 7.1 运行 `openspec validate simplify-overengineered-surfaces --strict`。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 7.4 运行 JEPA benchmark、CSI hardening 和 dataset runtime 相关 focused tests。
- [x] 7.5 高风险删减完成后运行 `conda run -n kd_mm_beam pytest -q`，并在最终说明中记录无法运行的命令和原因。
