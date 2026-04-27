## ADDED Requirements

### Requirement: 可导入包结构
项目 MUST 提供 `src/kd_sensing/` Python 包，并将数据、模型、蒸馏、训练引擎、评估、预处理和通用工具放入职责清晰的子模块。包内模块 MUST 使用包内绝对导入或明确相对导入，不得依赖仓库根目录脚本名作为运行时导入条件。

#### Scenario: 从项目根目录导入包
- **WHEN** 开发者在项目根目录安装或设置本地包路径后执行 `import kd_sensing`
- **THEN** 导入 MUST 成功，并且不触发数据集读取、模型权重加载或训练逻辑

#### Scenario: 导入核心子模块
- **WHEN** 开发者导入 `kd_sensing.models`、`kd_sensing.data`、`kd_sensing.distillation`、`kd_sensing.engine` 和 `kd_sensing.preprocessing`
- **THEN** 每个子模块 MUST 成功导入，并暴露对应领域的公共构建入口或注册入口

### Requirement: 模块边界清晰
项目 MUST 按职责拆分当前根目录代码：模型定义进入 `models/`，数据集与样本解析进入 `data/`，KD 与 loss 进入 `distillation/` 或 `losses/`，训练/验证/测试循环进入 `engine/`，雷达和 CSV 预处理进入 `preprocessing/`，指标与 checkpoint 等通用逻辑进入 `utils/` 或 `evaluation/`。

#### Scenario: 新增模型时不修改数据模块
- **WHEN** 开发者新增一个 student 或 teacher 模型实现
- **THEN** 变更 MUST 限定在模型相关模块和注册代码内，且不需要修改 dataset、preprocess 或训练循环中的数据解析逻辑

#### Scenario: 新增预处理流程时不修改模型模块
- **WHEN** 开发者新增一种雷达或 CSV 预处理流程
- **THEN** 变更 MUST 限定在 `preprocessing/` 与配置/注册代码内，且不需要修改模型定义文件

### Requirement: 统一新脚本入口并移除旧入口
项目 MUST 使用 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 或包内 CLI 作为唯一运行入口。项目 MUST 删除现有顶层旧脚本入口，包括 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 和 `gen_data_seq.py`，不得保留兼容包装脚本。

#### Scenario: 运行新训练脚本帮助信息
- **WHEN** 开发者执行 `python scripts/train.py --help`
- **THEN** 命令 MUST 正常退出，并展示配置文件、训练任务和命令行覆盖相关的参数说明

#### Scenario: 运行新评估和预处理脚本帮助信息
- **WHEN** 开发者执行 `python scripts/evaluate.py --help` 或 `python scripts/preprocess.py --help`
- **THEN** 命令 MUST 正常退出，并展示对应任务的参数说明

#### Scenario: 旧脚本入口已删除
- **WHEN** 结构重构完成后检查仓库根目录
- **THEN** 根目录 MUST 不存在 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 或 `gen_data_seq.py`

### Requirement: 项目根路径与资源路径统一
项目 MUST 提供统一路径解析工具，用于解析项目根目录、数据目录、权重目录、配置目录和输出目录。运行入口 MUST 通过该工具解析相对路径，避免依赖当前工作目录偶然匹配。

#### Scenario: 从子目录运行命令
- **WHEN** 开发者从仓库子目录调用新脚本或新 CLI，并传入相对数据路径
- **THEN** 系统 MUST 根据项目根路径解析资源位置，而不是错误地相对于当前子目录查找

#### Scenario: 读取默认资源目录
- **WHEN** 用户未显式传入数据目录或权重目录
- **THEN** 系统 MUST 使用配置中的默认路径，并能定位当前仓库的 `dataset/` 和 `All_models/` 目录
