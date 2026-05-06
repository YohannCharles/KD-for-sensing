## 1. 轻量导入边界

- [x] 1.1 增加轻量导入回归测试，覆盖 `import kd_sensing.config`、`from kd_sensing.utils.paths import resolve_path`、`import kd_sensing.registries` 不触发 dataset/model/diagnostics 默认导入。
- [x] 1.2 调整 `kd_sensing.utils.__init__`，避免包级导入 checkpoint registry 传播 dataset 场景和重依赖；将内部调用点迁移到窄模块导入。
- [x] 1.3 调整 `kd_sensing.data.__init__` 和 `kd_sensing.data.datasets.__init__` 的默认导出策略，避免轻量场景/config 导入时 eager import dataset 类。
- [x] 1.4 确保 `kd_sensing.config.io` 只依赖轻量模块，并用 `conda run -n kd_mm_beam python -c "import kd_sensing.config"` 验证配置导入边界。

## 2. 模态契约

- [x] 2.1 新增 `src/kd_sensing/modalities.py`，定义固定模态顺序、模态元数据结构和查询/标准化 helper。
- [x] 2.2 将 `engine/builders.py`、`models/fusion/networks.py`、`config/canonical.py` 和 diagnostics 中重复的合法模态列表迁移到模态契约。
- [x] 2.3 将 dataset flag、fusion input key、sample key 和默认字段推导改为使用模态契约，保持现有配置解析结果兼容。
- [x] 2.4 增加模态契约单元测试，覆盖乱序标准化、未知模态、重复模态、dataset flag 生成和 batch input key 查询。

## 3. Engine Builder 拆分

- [x] 3.1 新增 `engine/modality_resolution.py`，迁移启用模态推导、teacher/student 一致性校验和 dataset flag 冲突校验。
- [x] 3.2 新增 `engine/cache_policy.py`，迁移 cache policy 解析、校验和 dataset knob 注入。
- [x] 3.3 新增 `engine/data_factory.py`，迁移 dataset/dataloader 构建，继续在构建前显式调用默认组件注册导入。
- [x] 3.4 新增 `engine/run_metadata.py` 和 `engine/normalization_artifacts.py`，迁移 split/cache/throughput metadata 与 GPS/LiDAR/mmWave artifact 保存加载。
- [x] 3.5 新增 `engine/optim.py`，迁移 optimizer、scheduler 和 device 构建。
- [x] 3.6 保留 `kd_sensing.engine.builders` 兼容 facade，并更新训练、评估、诊断内部调用点优先使用窄模块。

## 4. Data Transforms 拆分

- [x] 4.1 建立 data transform 内部模块结构，按 image、radar、lidar、gps、mmwave、通用 IO/cache/normalization 切分实现。
- [x] 4.2 迁移 image motion mask 与 image cache 相关函数，并保持旧 `kd_sensing.data.transforms` 导入兼容。
- [x] 4.3 迁移 radar map 读取、LiDAR 点云/BEV、GPS feature、mmWave feature 和 scaler/normalizer 实现。
- [x] 4.4 更新 dataset、preprocessing 和 tests 的主要调用点，减少对聚合 transforms 入口的新增依赖。
- [x] 4.5 增加兼容导入测试，确保旧公开转换函数和 scaler 仍可从原路径导入。

## 5. Diagnostics Visualization 拆分

- [x] 5.1 新增 `diagnostics/visualization/` 内部包，拆出配置解析、dataset 构建、样本选择、统计、渲染和写出模块。
- [x] 5.2 保留 `kd_sensing.diagnostics.modality_visualization.visualize_modalities`、`visualize_modality_scene_comparison` 和 CLI 行为兼容。
- [x] 5.3 迁移 samples、summary、split_stats、final_config、PNG 输出逻辑，确保 preserve-existing metadata 后缀规则不变。
- [x] 5.4 增加或更新诊断测试，覆盖公开入口、输出路径、metadata 字段、按 seq/label 抽样和多 scene 输出隔离。

## 6. 文档与 Spec 整理

- [x] 6.1 更新 README 和 `docs/extension_guide.md`，说明模态契约、新模块边界、registry 默认组件导入方式和本地产物目录边界。
- [x] 6.2 补充相关 OpenSpec specs 的 Purpose，至少覆盖 `project-architecture`、`component-registry`、`modality-aware-data-loading` 和 `modality-visual-diagnostics`。
- [x] 6.3 记录兼容 facade 和推荐窄导入路径，避免后续新增代码继续依赖聚合文件。

## 7. 验证

- [x] 7.1 运行 `conda run -n kd_mm_beam python -c "import kd_sensing; import kd_sensing.config; import kd_sensing.registries"` 验证轻量导入。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest -q tests/test_student_configs.py tests/test_training_io_workflow.py` 验证配置、模态和训练 IO 兼容。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest -q tests/test_modality_visual_diagnostics.py` 验证诊断输出兼容。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest -q` 做全量回归；若环境缺数据依赖或 GPU，不可运行项必须在实现总结中明确记录。
