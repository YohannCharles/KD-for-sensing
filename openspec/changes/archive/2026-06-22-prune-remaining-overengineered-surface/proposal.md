## Why

上一轮项目表面收敛后，当前训练、评估、预处理和诊断主流程已经集中到 `src/kd_sensing` 包与 `kd-sensing-*` 入口，但仓库仍保留一批只为历史兼容、重复治理或单调用抽象存在的代码面。它们增加配置、测试和文档同步成本，却不提升当前实验能力；现在需要把 ponytail 审计结论转成可验证的 OpenSpec change，按风险删除或合并剩余过度工程。

## What Changes

- **BREAKING** 删除已从 registry 退役但仍可直接导入和 forward 的整模型类、旧类名 alias 及只服务它们的测试；保留仍被 `modular_sequence`、fusion 和当前 baseline 使用的特征提取器。
- **BREAKING** 收缩历史兼容 guard：只保留高频、仍有迁移价值且能防止静默误跑的拒绝项；完全退役且已有 tombstone、inventory 或 registry unknown-name 覆盖的路线不再维护专用 runtime 分支。
- **BREAKING** 删除或最小化无当前公开价值的 facade 和薄转发入口，包括内部 `__all__` 镜像、package-level re-export、3 行 CLI wrapper 和只为旧 import path 存在的模块。
- 用 `pyyaml` 作为唯一 YAML 解析实现，删除手写 `parse_simple_yaml` fallback 和 optional-yaml 分支；项目依赖已声明 `pyyaml`，无需维护第二套解析器。
- 折叠小型 recipe/dataclass 层：将只有一两个调用方、只包装常量表的 canonical recipe、objective metadata、dataset descriptor 等结构合并到真实 owner 或改为普通字典/helper。
- 收缩 runtime cleanup/run-index 的 legacy-only 分支，只保留当前输出分区、dry-run manifest、安全删除和必要 legacy archive 识别；历史输出考古不继续扩张为当前支持面。
- 简化 TinyViT 等重复注册代码为表驱动循环；不新增注册抽象或 factory 层。
- 保持核心行为不变：不改变训练数学语义、模型主路径、dataset split、beam label、metric 口径、checkpoint schema、当前 package CLI、canonical config 语义和本地产物保护边界。

## Capabilities

### New Capabilities

- 无。本变更只收缩现有项目表面和维护承诺。

### Modified Capabilities

- `project-surface-cleanup`: 增加 ponytail 审计候选分类、删除/保留证据、源码瘦身不得触碰本地产物的要求。
- `project-architecture`: 收缩 facade、thin CLI、internal `__all__`、轻量导入和 owner module 边界。
- `component-registry`: 明确 registry 只保留 canonical 构建面；退役整模型类和旧 alias 不再作为可导入兼容对象保活。
- `canonical-config-resolution`: 允许折叠 recipe/dataclass 小层和删除 YAML fallback，同时要求配置加载语义、实体 YAML 优先和覆盖顺序不变。
- `dataset-runtime-contracts`: 收敛重复 dataset descriptor/runtime row 层，优先复用 `modalities.py` 合约或局部 Mapping。
- `first-class-prediction-tasks`: 收缩 objective metadata/history 常量拆分，要求 objective 行为、metric alias 和 tensorboard 标量语义保持不变。
- `experiment-run-index`: 缩减 legacy-only 发现分支，保留当前 run index 的状态、过滤和渲染契约。
- `runtime-artifact-cleanup`: 缩减历史输出考古规则，保留 dry-run manifest、保护路径和安全删除边界。
- `tinyvit-image-encoder`: 允许表驱动注册 TinyViT preset，要求注册名和构建行为保持兼容。

## Impact

- 代码范围：`src/kd_sensing/models/{gps,image,radar,lidar,mmwave}.py`、`src/kd_sensing/models/fusion/`、`src/kd_sensing/registries.py`、`src/kd_sensing/config/`、`src/kd_sensing/data/dataset_descriptors.py`、`src/kd_sensing/engine/objectives/`、`src/kd_sensing/diagnostics/{run_index,runtime_artifact_cleanup}.py`、`src/kd_sensing/models/tinyvit.py` 和薄 CLI/facade 文件。
- 测试范围：registry、student/model config、GPS/LiDAR/mmWave/fusion focused tests、config load characterization、architecture boundaries、runtime artifact cleanup、run index、TinyViT encoder 和 objective metadata tests。
- 文档/OpenSpec 范围：本 change 的 delta specs、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml` 和必要 README/current docs 引用。
- API 影响：当前 `kd-sensing-*` console scripts、canonical configs、registry build 路径和主要 owner modules 保持可用；旧整模型类直接导入、旧 alias、内部 facade 和手写 YAML fallback 属于 breaking 收缩。
- 产物边界：不得删除或修改 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 历史权重或本地训练产物。
