## Why

当前项目以顶层脚本为主，训练、评估、数据集、模型、损失函数、雷达预处理和工具函数混在根目录，后续添加新模态、新蒸馏方法、新数据集或新实验配置时容易复制代码、改错路径，并且难以复现实验。现在需要把研究原型整理成接近常见开源深度学习论文代码库的结构，例如采用 `src` 包、配置文件、注册表和统一新脚本入口，让扩展成本更低。

## What Changes

- 引入可安装的 Python 包结构，将模型、数据集、损失函数、指标、训练引擎、评估逻辑、预处理脚本和通用工具拆分到清晰模块。
- 新增配置驱动的实验入口，支持 image-only、image+radar、多种 KD 模式、teacher/student 参数、数据路径和输出目录通过配置或命令行覆盖。
- 新增轻量组件注册机制，支持后续按名称添加模型、数据集、loss、KD 方法、metric、preprocess pipeline。
- **BREAKING** 移除现有 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py`、`gen_data_seq.py` 等顶层旧脚本入口，不提供兼容包装；统一使用 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 或包内 CLI。
- 增加基础项目文件和文档，包括依赖声明、配置示例、目录说明、运行命令、扩展指南和临时 smoke 验证流程。
- 实现时允许编写临时测试/验证脚本；验证完成后必须删除这些临时脚本，不在最终项目结构中保留额外测试脚本。
- 不迁移 `dataset/` 和 `All_models/` 中的大型数据与权重文件，只规范其引用方式和默认路径。

## Capabilities

### New Capabilities
- `project-architecture`: 定义可扩展的包结构、模块边界、新脚本入口、旧脚本移除和导入约定。
- `experiment-workflow`: 定义配置驱动的训练、评估、预处理和实验输出工作流。
- `component-registry`: 定义可按名称注册和构建模型、数据集、损失、指标与蒸馏组件的扩展机制。

### Modified Capabilities

无。

## Impact

- 影响根目录训练、测试和预处理脚本的组织方式；旧脚本入口将被删除，用户需要切换到新脚本或新 CLI。
- 影响模型、数据加载、蒸馏损失、指标和工具函数的导入路径，需要统一到新包命名空间。
- 新增 `configs/`、`scripts/`、`src/`、依赖声明和文档内容；临时验证脚本只用于实现期间验证，最终必须清理。
- 训练结果、checkpoint、日志、曲线和评估报告应输出到统一 `outputs/` 或配置指定目录。
- 算法行为默认保持不变；结构调整不应改变现有 image-only 和 image+radar 训练/评估的默认参数语义。
