## Context

当前项目是典型论文原型代码：`train_image.py`、`train_both.py`、测试脚本、数据处理脚本、模型定义、损失函数和工具函数都位于仓库根目录。它可以支撑当前 Scenario 9 的 image-only 与 image+radar 实验，但后续扩展会遇到几个问题：

- 训练脚本之间存在大量重复逻辑，例如参数解析、数据构建、模型构建、KD loss、验证、测试、checkpoint 与曲线保存。
- `model_image.py` 与 `model_both.py` 内部同时包含特征提取器、teacher、student 和融合逻辑，添加新模型时缺少统一入口。
- `DataFeed.py` 将 CSV 解析、样本构建、图像处理、雷达读取和张量组装集中在一个类中，添加新数据集或新模态时容易牵连训练代码。
- 当前导入依赖根目录文件名，难以作为包被测试、复用或被命令行工具调用。

设计参考常见开源深度学习论文仓库的可扩展结构：OpenMMLab/MMEngine 的 registry/config 思路、BasicSR 的 `archs/data/losses/models/options` 分层、以及 PyTorch Lightning/Hydra 类项目中配置驱动实验的组织方式。但本项目规模较小，不引入重量级训练框架，优先采用轻量包结构和简单注册表。

## Goals / Non-Goals

**Goals:**

- 将代码整理为 `src/kd_sensing/` 下的可导入包，目录边界能表达数据、模型、蒸馏、训练、评估、预处理、工具等职责。
- 提供统一配置和 CLI 入口，让 image-only、image+radar、no-KD、logits KD、RKD 实验通过配置组合运行。
- 删除现有顶层旧脚本入口，统一使用新 `scripts/` 命令或包内 CLI，避免长期维护两套入口。
- 让新增模型、数据集、loss、metric、KD 方法时只需新增模块并注册，训练入口无需复制大段逻辑。
- 将实验输出统一到可配置目录，保存 config、checkpoint、metrics、曲线和测试报告，便于复现和比较。
- 在实现期间增加临时 smoke 验证脚本，覆盖包导入、注册表构建、配置加载和新 CLI 入口；验证完成后删除这些临时脚本。

**Non-Goals:**

- 不改变现有网络结构、KD 算法、默认超参数和评估指标的语义。
- 不迁移或重命名 `dataset/`、`All_models/` 中的大型数据和权重文件。
- 不引入复杂分布式训练、实验追踪平台或完整训练框架。
- 不在本次重构中重新调参、重新训练或重新发布权重。

## Decisions

### Decision: 使用 `src/kd_sensing` 包结构

目标结构：

```text
configs/
  image/
  fusion/
  preprocess/
scripts/
  train.py
  evaluate.py
  preprocess.py
src/
  kd_sensing/
    __init__.py
    cli/
    config/
    data/
    distillation/
    engine/
    evaluation/
    models/
      backbones/
      heads/
      fusion/
    preprocessing/
    registries.py
    utils/
```

不保留根目录旧脚本入口。迁移完成后删除 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 和 `gen_data_seq.py`，统一通过新脚本调用：

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/preprocess.py`

也可以在 `pyproject.toml` 中提供 console scripts，但最终用户文档以新脚本为主。

替代方案：保留旧脚本作为薄包装。该方案迁移阻力小，但会留下两套入口和额外文档负担，不符合本次“直接使用新脚本”的目标。

### Decision: 采用轻量配置系统，不强依赖 Hydra/MMEngine

配置文件使用 YAML，默认配置拆为数据、模型、训练、KD、输出几类。CLI 支持 `--config configs/image/rkd.yaml` 和少量点式覆盖，例如 `training.epochs=20`。内部将配置规范化为 dataclass 或简单字典，训练代码只消费规范化后的配置。旧 argparse 参数不再作为兼容接口保留，只在迁移时参考其默认值。

替代方案：直接引入 Hydra 或 MMEngine Config。它们功能更完整，但会改变运行习惯并增加依赖复杂度；当前项目更适合先落地轻量实现，后续可替换为成熟配置库。

### Decision: 引入统一 registry，但保持实现简单

新增 `registries.py`，提供 `MODELS`、`DATASETS`、`LOSSES`、`METRICS`、`DISTILLERS`、`PREPROCESSORS`。组件模块通过装饰器注册，构建函数接收配置中的 `type` 和参数。

替代方案：在训练脚本中使用 `if/elif` 分支选择组件。它对当前两个训练脚本足够，但新增 student、teacher、雷达编码器或 KD 变体时会持续膨胀。

### Decision: 抽离训练/验证/测试公共流程

将公共逻辑放入 `engine/`：

- `builders.py`: 根据配置构建 dataset、dataloader、model、optimizer、scheduler、loss、distiller。
- `trainer.py`: 负责训练循环、KD teacher 冻结、梯度裁剪、early stopping、scheduler、checkpoint。
- `validator.py` 和 `evaluator.py`: 负责验证、测试、latency/FLOPs/Top-K/DBA 等指标。
- `checkpoint.py` 和 `logging.py`: 统一保存和恢复。

image-only 与 fusion 的差异通过 batch adapter 或 task runner 处理，避免复制整段 epoch loop。

替代方案：分别保留 `train_image` 和 `train_both` 两套实现，只共享工具函数。该方案迁移容易，但后续每加一个模态组合都会新增脚本副本。

### Decision: 数据与预处理拆分

将 `DataFeed.py` 拆为：

- `data/samples.py`: CSV 行解析和 sequence sample 构建。
- `data/datasets/scenario9.py`: Scenario 9 Dataset。
- `data/transforms.py`: 图像 resize、灰度、motion mask、雷达 map 组装。
- `preprocessing/radar.py` 与 `preprocessing/csv.py`: 雷达 FFT/KPI 与 CSV 预处理。

训练代码只依赖 dataset 返回的标准 batch 字段，如 `image`、`radar_ra`、`radar_da`、`input_beam`、`target_beam`。

替代方案：保留当前 Dataset 单类。该方案代码少，但不利于新增 dataset 或替换 preprocessing pipeline。

### Decision: 分阶段迁移并保持行为等价

迁移先复制/移动代码到新包，再实现新脚本入口，验证通过后删除旧脚本和临时验证脚本。每个阶段都运行最小导入和 dry-run 检查，避免结构重构同时改变算法行为。

替代方案：一次性重写训练脚本。速度看似更快，但训练代码包含 KD、early stopping、checkpoint、评估等细节，一次重写容易引入不可见回归。

## Risks / Trade-offs

- 路径默认值在包化后可能变化 -> 使用项目根路径解析工具，支持新配置默认 `dataset/scenario9`，并在 README 明确迁移后的路径写法。
- 迁移训练循环可能改变数值行为 -> 先保留原有函数语义，增加临时 smoke 验证和小批量 dry-run，对关键默认参数做快照验证，验证后删除临时脚本。
- registry 增加间接层，初期阅读成本上升 -> registry API 保持极小，只提供 register/build/list，并在 README 写扩展示例。
- 直接删除旧入口会破坏旧命令 -> 在 README 给出旧命令到新命令的迁移表，并在变更中明确这是 breaking change。
- 大型权重和数据不适合放入测试 -> 实现期间使用临时 synthetic smoke 脚本和小批量 dry-run 验证，验证完成后删除临时脚本。

## Migration Plan

1. 创建新目录、依赖声明和包初始化，确保 `python -c "import kd_sensing"` 可运行。
2. 移入通用工具、指标、checkpoint、plot、loss、KD 逻辑，并统一包内导入路径。
3. 移入模型定义并注册 image teacher/student 与 fusion teacher/student。
4. 拆分 Scenario 9 dataset 和雷达/CSV 预处理模块。
5. 抽离训练、验证、测试 engine，并让新 CLI 支持 image-only 与 fusion 配置。
6. 创建临时 smoke 验证脚本，验证导入、注册表、配置加载、新脚本入口和 synthetic dry-run。
7. 删除旧顶层脚本入口和已完成验证的临时 smoke 脚本。
8. 更新 README，补充目录结构、新脚本命令、旧命令迁移表、配置示例和扩展新模型/KD 方法的步骤。
9. 增加 lint/format 建议，并记录已执行的验证命令。

回滚策略：每次迁移模块时保持小步提交，删除旧脚本前完成新脚本 smoke 验证；若新 engine 行为异常，回滚对应迁移步骤，而不是长期保留旧脚本入口。

## Open Questions

- 项目是否计划发布为 pip 包。如果需要，`pyproject.toml` 应补充包元数据、console scripts 和可选依赖分组。
- 是否需要后续接入 TensorBoard、Weights & Biases 或 CSV logger。本次只预留 logger 接口。
- 是否需要支持多数据集。当前规范以 Scenario 9 为默认实现，但目录会为多数据集注册预留空间。
