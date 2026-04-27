## 1. 项目骨架与依赖

- [x] 1.1 创建 `src/kd_sensing/` 包结构及 `cli/`、`config/`、`data/`、`distillation/`、`engine/`、`evaluation/`、`models/`、`preprocessing/`、`utils/` 子目录
- [x] 1.2 增加基础项目文件，至少包含 `pyproject.toml` 或等价依赖声明、包发现配置和最小开发安装说明
- [x] 1.3 实现项目根路径与资源路径解析工具，覆盖数据目录、权重目录、配置目录和输出目录
- [x] 1.4 确保 `python -c "import kd_sensing"` 不读取数据、不加载权重、不启动训练

## 2. 配置系统

- [x] 2.1 实现 YAML 配置加载、默认值合并和命令行覆盖解析
- [x] 2.2 新增 image-only no-KD、logits KD、RKD 配置示例
- [x] 2.3 新增 image+radar no-KD、logits KD、RKD 配置示例
- [x] 2.4 将旧脚本中的默认超参数迁移到新配置文件，不保留旧 argparse 兼容入口

## 3. 组件注册表

- [x] 3.1 实现轻量 `Registry` 类，支持注册、查询、构建、列出可用名称和重复注册检测
- [x] 3.2 定义 `MODELS`、`DATASETS`、`LOSSES`、`METRICS`、`DISTILLERS`、`PREPROCESSORS` 注册表
- [x] 3.3 为未知组件、重复注册和缺失参数提供包含注册表名称与可用名称的错误信息
- [x] 3.4 为注册表提供可被临时验证脚本调用的最小 self-check，覆盖成功构建、未知名称和重复注册

## 4. 迁移模型、损失和指标

- [x] 4.1 将 `model_image.py` 中的 image teacher/student 相关类迁移到 `kd_sensing.models`
- [x] 4.2 将 `model_both.py` 中的 radar extractor、image extractor、fusion teacher/student 相关类迁移到 `kd_sensing.models`
- [x] 4.3 将 `DistillationLoss.py` 和 `MyFunc.py` 中的 focal loss、KD loss、Top-K、DBA、FLOPs、checkpoint、plot 等逻辑拆入对应模块
- [x] 4.4 为迁移后的模型、loss、metric、distiller 注册默认名称
- [x] 4.5 使用 synthetic tensor 验证 image-only 和 fusion 模型 forward 输出形状与旧实现语义一致

## 5. 迁移数据与预处理

- [x] 5.1 将 CSV 样本解析和 sequence sample 构建拆入 `kd_sensing.data.samples`
- [x] 5.2 将 Scenario 9 dataset 迁移为可注册 dataset，并标准化 batch 字段名称
- [x] 5.3 将图像 resize、灰度、motion mask 和雷达 RA/DA 读取逻辑拆入 transform 或 dataset helper
- [x] 5.4 将 `Radar_KPI.py`、`CSV_process.py` 和 `gen_data_seq.py` 的核心逻辑迁移到 `kd_sensing.preprocessing`
- [x] 5.5 增加仅用于验证的 synthetic dataset 构建工具，支撑训练/评估 smoke 验证

## 6. 训练、评估与 CLI

- [x] 6.1 实现构建器，根据配置构建 dataset、dataloader、model、optimizer、scheduler、loss、distiller 和 metric
- [x] 6.2 抽离统一训练循环，覆盖 teacher 冻结、KD 模式、alpha warmup、gradient clipping、early stopping、scheduler 和 checkpoint
- [x] 6.3 抽离验证与测试流程，覆盖 loss、Top-K、DBA、latency 和可用的 FLOPs 统计
- [x] 6.4 实现 `scripts/train.py`，内部调用 `kd_sensing.cli.train`，支持配置文件和命令行覆盖
- [x] 6.5 实现 `scripts/evaluate.py`，内部调用 `kd_sensing.cli.evaluate`，支持权重路径、配置文件和输出目录
- [x] 6.6 实现 `scripts/preprocess.py`，内部调用 `kd_sensing.cli.preprocess`，支持 CSV 处理和序列生成
- [x] 6.7 统一输出目录，保存最终配置、checkpoint、metrics、日志和曲线

## 7. 删除旧入口

- [x] 7.1 删除 `train_image.py` 和 `train_both.py`，训练统一改用 `scripts/train.py`
- [x] 7.2 删除 `test_model_image.py` 和 `test_model_both.py`，评估统一改用 `scripts/evaluate.py`
- [x] 7.3 删除 `CSV_process.py` 和 `gen_data_seq.py`，预处理统一改用 `scripts/preprocess.py`
- [x] 7.4 更新所有文档和示例命令，移除旧脚本调用方式
- [x] 7.5 检查仓库根目录，确认旧脚本入口文件不存在

## 8. 临时验证脚本与文档

- [x] 8.1 创建临时验证脚本，覆盖 `kd_sensing.models`、`kd_sensing.data`、`kd_sensing.distillation`、`kd_sensing.engine` 和 `kd_sensing.preprocessing` 的导入
- [x] 8.2 创建临时验证脚本，覆盖注册表构建、未知名称、重复注册、默认配置加载和 CLI 覆盖
- [x] 8.3 创建临时验证脚本，执行 image-only 与 image+radar synthetic dry-run，覆盖 forward、loss、backward、validation 和 checkpoint 保存
- [x] 8.4 运行 `python scripts/train.py --help`、`python scripts/evaluate.py --help` 和 `python scripts/preprocess.py --help`
- [x] 8.5 运行所有临时验证脚本并记录关键输出，然后删除临时验证脚本和临时验证目录
- [x] 8.6 更新 README，说明新目录结构、安装方式、训练/评估/预处理命令、旧命令迁移表和 breaking change
- [x] 8.7 增加扩展指南，说明如何新增模型、dataset、loss、metric、distiller 和 preprocessor
- [x] 8.8 运行 OpenSpec 状态检查和最终文件清理检查，确认 tasks 前置 artifact 已完成、旧脚本入口不存在、临时验证脚本不存在
