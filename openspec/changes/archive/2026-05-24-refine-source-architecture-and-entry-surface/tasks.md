## 1. Inventory 与 Guardrails

- [x] 1.1 更新 `docs/project_surface_inventory.md`，记录当前大文件拆分候选、实体 YAML 数量、脚本入口 allowlist 和本 change 不触碰数据/产物的范围。
- [x] 1.2 更新 `tests/test_architecture_boundaries.py` 或等价快速检查，覆盖入口生命周期分类、可生成 YAML 防回流和大模块职责回流检查。
- [x] 1.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认 guardrail 基线可运行。

## 2. Viewer 源码拆分

- [x] 2.1 将 `tools/visualization/viewer_utils.py` 中 manifest 读取、路径解析、scene/split/show mode 过滤逻辑迁移到窄模块，并保留现有 import 兼容。
- [x] 2.2 将 viewer 图表构造、prediction summary 表格和 legacy prediction adapter 拆到独立 helper，确保 Gradio 布局编排只调用公开 helper。
- [x] 2.3 更新 viewer 相关测试，运行 `conda run -n kd_mm_beam pytest tests/test_gradio_complementarity_explorer.py tests/test_modality_visual_diagnostics.py -q`。

## 3. Raymobtime s008 预处理拆分

- [x] 3.1 为 Raymobtime s008 预处理建立职责模块，覆盖 paths/audit、index、beam labels、ray features 和 cache writer。
- [x] 3.2 迁移 `src/kd_sensing/preprocessing/raymobtime_s008.py` 中 index split、official split 检测和 metadata 写出逻辑，保持 preprocessor registry 入口兼容。
- [x] 3.3 迁移 beam label normalization、ray feature 提取和 cache 写出逻辑，保持已声明配置和输出文件语义兼容。
- [x] 3.4 运行 `conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py -q`。

## 4. Diagnostics 与 CSI 模型职责收敛

- [x] 4.1 将 `src/kd_sensing/diagnostics/complementarity.py` 拆分为 schema adapter、case mining、summary 和 writers，并保持公开函数兼容。
- [x] 4.2 将 `src/kd_sensing/models/csi.py` 中 pilot estimation、CSI hardening、view tokenizer/fusion 和 encoder registry glue 分离到窄模块。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_complementarity_analysis.py tests/test_csi_modality.py -q`。

## 5. 配置矩阵与入口收敛

- [x] 5.1 生成高级配置二次瘦身候选清单，标记可无损 recipe 化、存在显式差异和需要人工保留的实体 YAML。
- [x] 5.2 为首批可删除实体 YAML 增加关键字段等价测试，覆盖 experiment、task、dataset、modalities、model、loss/distillation、training、run name 和 checkpoint 来源。
- [x] 5.3 将首批无损候选迁移到 canonical/advanced overlay recipe，并删除对应实体 YAML。
- [x] 5.4 复核 `scripts/`、`tools/analysis/`、`tools/visualization/` 入口分类，删除已有包内 CLI 覆盖的重复 wrapper 或在 inventory 中记录短期保留原因。
- [x] 5.5 更新 README、`docs/project_surface_inventory.md` 和相关工具文档中的推荐入口，确保不把研究脚本描述为核心 workflow 唯一入口。

## 6. 验证与 OpenSpec

- [x] 6.1 运行 `openspec validate refine-source-architecture-and-entry-surface --strict`。
- [x] 6.2 运行 `openspec status --change refine-source-architecture-and-entry-surface`，确认 artifacts 和 tasks 状态可读。
- [x] 6.3 运行核心 CLI smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-preprocess --help`、`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`。
- [x] 6.4 运行快速回归：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_student_configs.py -q`。
- [x] 6.5 运行最终回归：`conda run -n kd_mm_beam pytest -q`。
