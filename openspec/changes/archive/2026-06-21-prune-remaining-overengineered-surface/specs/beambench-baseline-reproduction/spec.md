## ADDED Requirements

### Requirement: BeamBench Image AE+GPS owner 必须避免大聚合 re-export
BeamBench Image AE+GPS Direct workflow MUST 保持 package CLI、训练 runner、paper-split runner、evaluation、config、dataset、model 和 report 行为兼容，但不要求保留一个导出所有符号的大聚合 module。内部代码和测试 MUST 直接导入具体 owner 模块；聚合 module 如保留，MUST 只作为薄 public shim。

#### Scenario: package CLI 继续可用
- **WHEN** 用户运行 `kd-sensing-train-beambench-image-ae-gps` 或 `kd-sensing-run-beambench-image-ae-gps-tableiii`
- **THEN** 命令 MUST 继续调用 Image AE+GPS Direct 的训练、paper-split 或 eval-only workflow
- **AND** 输出 checkpoint、history、predictions、CSV/Markdown/JSON summary 和 BeamBench metrics 的边界 MUST 保持兼容

#### Scenario: 内部代码使用具体 owner
- **WHEN** CLI 或 tests 需要 `run_image_ae_gps_training`、`run_image_ae_gps_paper_split_training`、config resolver、dataset 或 model class
- **THEN** 代码 MUST 从 `image_ae_gps_training.py`、`image_ae_gps_paper_split.py`、`image_ae_gps_config.py`、`image_ae_gps_datasets.py` 或 `image_ae_gps_models.py` 等具体 owner 导入
- **AND** 内部代码 MUST 不依赖 `image_ae_gps.py` 大聚合转发表

#### Scenario: 聚合 shim 可删除
- **WHEN** README、docs、OpenSpec current specs、CLI 和 tests 都不再把 `kd_sensing.baselines.beambench.image_ae_gps` 声明为 current public import owner
- **THEN** 本 change MAY 删除该 module
- **AND** 删除 MUST 不改变 package console scripts 和当前 BeamBench focused tests 的行为

### Requirement: BeamBench baseline package export 必须最小化
`kd_sensing.baselines.beambench` package-level exports MUST 不急切导入训练、dataset、Pillow/torch 或 pandas-heavy owner 模块，除非用户显式调用对应 package CLI 或具体 owner。保留 package `__init__` 时，MUST 避免成为内部代码依赖的便利聚合层。

#### Scenario: package import 轻量
- **WHEN** 开发者执行 `import kd_sensing.baselines.beambench`
- **THEN** 导入 SHOULD 不触发真实 dataset 读取、训练构建或 checkpoint 加载
- **AND** 需要重依赖的训练和评估功能 MUST 通过具体 owner 或 CLI 触发

#### Scenario: 不新增 package-level wrapper
- **WHEN** 新增 BeamBench helper 或 report writer
- **THEN** 实现 MUST 放入现有具体 owner 模块或新窄模块
- **AND** `__init__.py` MUST 不新增只为便利导入的长期 re-export
