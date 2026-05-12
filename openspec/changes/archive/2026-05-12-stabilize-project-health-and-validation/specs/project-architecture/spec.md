## ADDED Requirements

### Requirement: 包级导入不得牵出重依赖
项目 MUST 保持包级公共 API 兼容，同时避免 `__init__.py` eager import 触发重依赖运行模块。导入某个具体子模块时，系统 MUST 不因为父包初始化而额外导入训练器、dataset、诊断渲染或大型第三方依赖。

#### Scenario: 导入 engine 轻量子模块
- **WHEN** 开发者执行 `import kd_sensing.engine.model_output`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.engine._builders_impl`
- **AND** 系统 MUST 不导入 `kd_sensing.data.transform_ops._legacy`
- **AND** 系统 MUST 不导入 `pandas` 或 `scipy`

#### Scenario: 导入 diagnostics 轻量子模块
- **WHEN** 开发者执行 `import kd_sensing.diagnostics.g2d_diagnostics`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.diagnostics.visualization.core`
- **AND** 系统 MUST 不导入 `matplotlib`

#### Scenario: 导入 distillation 工具子模块
- **WHEN** 开发者执行 `import kd_sensing.distillation.g2d_smp`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不因为 `kd_sensing.distillation.__init__` 导入 distiller registry、engine builder 或 dataset 转换模块

#### Scenario: 旧包级公共符号仍可访问
- **WHEN** 现有代码执行 `from kd_sensing.engine import train` 或 `from kd_sensing.diagnostics import export_viewer_manifest`
- **THEN** 导入 MUST 继续成功
- **AND** 对应重依赖模块 MUST 仅在访问该公共符号时按需加载

### Requirement: builder 实现不得集中在私有聚合模块
训练引擎 MUST 将 builder 实现放在对应职责模块中。`kd_sensing.engine._builders_impl` MAY 作为临时兼容层存在，但新实现和测试 MUST 以 `cache_policy`、`modality_resolution`、`data_factory`、`normalization_artifacts`、`run_metadata` 和 `optim` 等窄模块为主。

#### Scenario: cache policy 实现在 cache 模块
- **WHEN** 开发者查看或修改 cache policy 解析逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.cache_policy`
- **AND** 不需要编辑 optimizer、run metadata 或 dataset 构建实现

#### Scenario: optimizer 和 device 构建实现在 optim 模块
- **WHEN** 开发者查看或修改 optimizer、scheduler、device 或 distiller 参数组构建逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.optim`
- **AND** 不需要编辑 dataset/dataloader 构建实现

#### Scenario: 兼容 builders facade
- **WHEN** 现有代码从 `kd_sensing.engine.builders` 导入公开构建函数
- **THEN** 导入 MUST 继续成功
- **AND** 函数行为 MUST 与拆分前保持兼容

### Requirement: 模态转换实现不得集中在 legacy 聚合模块
数据转换模块 MUST 将仍在使用的 image、GPS、LiDAR、mmWave、radar、IO、cache 和 normalization 实现放入对应模块。`kd_sensing.data.transform_ops._legacy` MAY 作为兼容过渡层存在，但新增实现和主要维护点 MUST 不再集中于 `_legacy.py`。

#### Scenario: 修改 image motion cache 不触碰 LiDAR 实现
- **WHEN** 开发者修改 image motion mask 或 image motion cache key
- **THEN** 主要变更 MUST 限定在 image 或通用 cache/IO 模块
- **AND** 不需要编辑 LiDAR、GPS、mmWave 或 radar 转换实现

#### Scenario: 修改 GPS scaler 不触碰 image 实现
- **WHEN** 开发者修改 GPS feature 或 scaler 加载保存逻辑
- **THEN** 主要变更 MUST 限定在 GPS 或通用 normalization 模块
- **AND** 不需要编辑 image motion、LiDAR BEV 或 radar map 转换实现

#### Scenario: 旧 transforms facade 兼容
- **WHEN** 现有代码从 `kd_sensing.data.transforms` 导入已公开的转换函数或 scaler
- **THEN** 导入 MUST 继续成功
- **AND** 函数行为 MUST 与拆分前保持兼容

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。

#### Scenario: 可视化 manifest 导出入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: 可视化兼容入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest 或指向 Gradio viewer 工作流

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-visualize-modalities` 和 `kd-sensing-export-viewer-manifest`

### Requirement: 内置权重与本地产物边界明确
项目 MUST 明确区分内置复现权重和本地生成 checkpoint。已跟踪的 `All_models` 权重如果继续保留，MUST 被文档标记为内置复现输入；新训练、评估或诊断产生的 checkpoint 和缓存 MUST 继续被忽略。

#### Scenario: README 说明 All_models 策略
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `All_models` 中已跟踪权重的用途、加载路径和是否属于源码仓库的可复现输入
- **AND** 文档 MUST 说明新生成的 `.pth` checkpoint 不应进入源码变更

#### Scenario: 新生成 checkpoint 不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令并生成 `.pth`、cache 或输出文件
- **THEN** 这些文件 MUST 位于 `.gitignore` 覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

### Requirement: 项目健康检查可分层运行
项目 MUST 提供或记录一组快速健康检查，用于在不启动真实训练的情况下验证导入边界、CLI 入口、Phase 1.5 决策语义和核心诊断逻辑。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 轻量导入 smoke
- **WHEN** 开发者运行项目健康检查中的轻量导入 smoke
- **THEN** 检查 MUST 验证配置、路径、模态契约、engine 轻量子模块、diagnostics 轻量子模块和 distillation 工具子模块可导入
- **AND** 检查 MUST 验证这些导入不触发指定重依赖模块

#### Scenario: 快速回归命令覆盖当前红点
- **WHEN** 开发者运行项目健康检查中的快速回归命令
- **THEN** 检查 MUST 覆盖 Phase 1.5 pending gate、架构导入边界、console script help 和互补分析核心测试
- **AND** 命令 MUST 能在全量 pytest 之前快速暴露项目结构回归

#### Scenario: 全量测试仍作为最终验收
- **WHEN** 变更实现完成
- **THEN** 开发者 MUST 使用 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 全量测试 MUST 通过
