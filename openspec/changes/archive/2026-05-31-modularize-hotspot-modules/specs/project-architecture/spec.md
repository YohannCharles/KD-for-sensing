## ADDED Requirements

### Requirement: 第一批源码热点必须收敛为薄 facade
项目 MUST 优先拆分 `src/kd_sensing/engine/hist_beam_loso_execution.py` 和 `src/kd_sensing/data/mmw/preparation.py` 这两个 2000 行级源码热点。拆分后，这两个文件 MAY 保留现有公开 import 和公开入口，但 MUST 不再作为主要实现聚合文件；主要实现 MUST 位于同包内按职责命名的窄模块中。架构边界测试 MUST 对这些 facade 设置降低后的行数上限和禁止实现片段断言，单个第一批 facade 的上限 MUST 不超过 1000 行。

#### Scenario: HiST-Beam LOSO executor facade 收敛
- **WHEN** 开发者查看或修改 HiST-Beam LOSO preflight、stage orchestration、run record、progress event、summary/conclusion 或 matrix metadata 逻辑
- **THEN** 主要实现 MUST 位于 `hist_beam_loso_preflight.py`、`hist_beam_loso_stages.py`、`hist_beam_loso_records.py`、`hist_beam_loso_artifacts.py`、`hist_beam_loso_summary.py`、`hist_beam_loso_matrix.py`、`hist_beam_loso_config.py` 或等价窄模块
- **AND** `hist_beam_loso_execution.py` MUST 只承担公开入口编排、常量声明和兼容导出
- **AND** 架构边界测试 MUST 拒绝把上述窄职责的大段 helper 重新实现到 `hist_beam_loso_execution.py`

#### Scenario: MMW preparation facade 收敛
- **WHEN** 开发者查看或修改 MMW Town10 preparation 的配置解析、zip/input 审计、sensor/channel indexing、sequence split、beam power 派生、manifest 写出、report 或 proxy geometry 逻辑
- **THEN** 主要实现 MUST 位于 `preparation_config.py`、`preparation_audit.py`、`preparation_index.py`、`preparation_splits.py`、`preparation_beam_power.py`、`preparation_writers.py`、`preparation_geometry.py` 或等价窄模块
- **AND** `data/mmw/preparation.py` MUST 只承担公开 orchestration、现有公开 helper 的兼容导出和顶层参数编排
- **AND** 架构边界测试 MUST 拒绝把上述窄职责的大段 helper 重新实现到 `data/mmw/preparation.py`

### Requirement: MMW preparation 拆分后职责不得交叉回流
MMW preparation 拆分后的窄模块 MUST 按配置、输入审计、索引、split、beam power、写出和几何/proxy 特征职责组织。新增或修改某一职责时，变更 MUST 不要求编辑其它无关职责模块，除非公开 orchestration 需要连接新的参数或结果字段。

#### Scenario: 修改 sequence split 不触碰 beam power
- **WHEN** 开发者调整 MMW sequence row 构造、group-safe split、guard band 或 leakage diagnostics
- **THEN** 主要变更 MUST 位于 `preparation_splits.py` 或等价 split 模块
- **AND** 不需要修改 channel payload 读取、DFT/codebook beam power 计算或 power vector 校验实现

#### Scenario: 修改 zip/input 审计不触碰 manifest 写出
- **WHEN** 开发者调整 MMW zip 输入校验、extract marker、source hash 或 availability audit
- **THEN** 主要变更 MUST 位于 `preparation_audit.py` 或等价审计模块
- **AND** 不需要修改 manifest CSV、split CSV、report JSON 或 artifact path 写出实现

#### Scenario: 修改 proxy geometry 不触碰 index
- **WHEN** 开发者调整 relative geometry、pose 解析、vehicle proxy feature 或 azimuth bin 逻辑
- **THEN** 主要变更 MUST 位于 `preparation_geometry.py` 或等价几何模块
- **AND** 不需要修改 sensor frame indexing、channel file indexing 或 scenario root 搜索实现

### Requirement: 热点拆分 inventory 必须覆盖优先级和禁止回流路径
项目 MUST 在表面积 inventory 或等价文档中记录热点模块拆分优先级、兼容 facade、推荐窄模块和禁止内部回流路径。架构边界测试 MUST 与 inventory 保持一致，并 MUST 对第一批 facade 执行行数上限、禁止片段和 helper 所属模块断言。

#### Scenario: inventory 记录第一批与第二梯队热点
- **WHEN** 开发者运行架构边界测试或审阅 `docs/project_surface_inventory.md`
- **THEN** inventory MUST 记录 `hist_beam_loso_execution.py` 和 `data/mmw/preparation.py` 的 facade 到窄模块映射
- **AND** inventory MUST 记录第二梯队热点清单，至少包含 `models/fusion/hist_beam.py`、`diagnostics/run_index.py`、`tools/visualization/gradio_multimodal_viewer.py`、`data/transform_ops/csi.py`、`engine/batch.py` 和 `engine/evaluation_pass.py`
- **AND** inventory MUST 说明第二梯队热点的后续拆分方向或暂缓原因

#### Scenario: 内部代码不得从第一批 facade 回流导入 helper
- **WHEN** 内部源码新增对第一批 facade 中已迁移 helper 的 import 或调用
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应窄模块作为修复路径

### Requirement: 热点拆分必须保持公开行为兼容
热点模块拆分 MUST 保持现有公开 CLI、公开 import、manifest schema、run metadata、summary CSV/JSON、preparation artifact 命名、样本契约和默认路径策略兼容。拆分只允许改变内部模块组织，不得改变模型数值语义、数据 split 语义、beam label 语义或本地产物边界。

#### Scenario: HiST-Beam LOSO 产物兼容
- **WHEN** HiST-Beam LOSO executor 拆分完成后运行 focused characterization tests
- **THEN** 现有 run metadata、execution progress JSONL、summary JSON、summary CSV、quick validation conclusion、checkpoint reuse metadata 和公开 CLI 参数 MUST 保持兼容
- **AND** 测试 MUST 覆盖关键公开字段，避免拆分时丢失或重命名字段

#### Scenario: MMW preparation 产物兼容
- **WHEN** MMW preparation 拆分完成后运行 focused characterization tests
- **THEN** 现有 frame manifest、sequence split CSV、split metadata、beam power artifact、data availability report 和 report JSON 的关键字段 MUST 保持兼容
- **AND** 测试 MUST 覆盖公开 `prepare_town10_skybridge` 工作流和仍保留的公开 helper import

#### Scenario: 本地产物边界不随拆分改变
- **WHEN** 开发者实施热点拆分、运行 focused tests 或执行 CLI smoke
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或真实本地运行产物的删除、移动、压缩或重写
- **AND** 生成的临时验证产物 MUST 位于忽略规则覆盖范围内或测试临时目录中

### Requirement: 分层验证必须覆盖热点拆分
热点拆分实现完成后，项目 MUST 分层验证 OpenSpec、架构边界、focused 行为兼容和公开入口。所有项目相关 Python 验证 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 快速架构与 OpenSpec 校验
- **WHEN** 热点拆分任务完成
- **THEN** 开发者 MUST 运行 `openspec validate modularize-hotspot-modules --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 领域 focused tests 校验
- **WHEN** HiST-Beam executor 或 MMW preparation 拆分完成
- **THEN** 开发者 MUST 运行对应 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_mmw_town10_preparation.py -q`
- **AND** 若拆分触碰公开 CLI 或 viewer 入口，开发者 MUST 运行对应 help smoke 或 viewer/import smoke

#### Scenario: 全量回归作为最终验收
- **WHEN** 第一批热点拆分和架构防护全部完成
- **THEN** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 若全量测试因环境或本地数据缺失无法完成，最终说明 MUST 明确列出未运行原因和已完成的替代 focused 验证
