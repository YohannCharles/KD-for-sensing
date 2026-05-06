## Context

当前代码已经形成 `config -> engine -> data/models/distillation -> outputs` 的主路径，但几个横切关注点仍混在一起：

- `kd_sensing.config` 需要配置加载和校验，却会通过 `utils/__init__.py`、`artifact_registry` 或 `data/__init__.py` 间接导入 dataset 及 pandas/scipy/skimage 等重依赖。
- 模态集合、顺序、dataset flag、batch key、model input key 和默认字段在 `engine/builders.py`、`data/datasets/scenario9.py`、`models/fusion/networks.py`、`diagnostics/modality_visualization.py` 和 `config/canonical.py` 中重复定义。
- `engine/builders.py`、`data/transforms.py`、`diagnostics/modality_visualization.py` 已成为聚合文件，修改一个模态时容易牵连训练构建、数据读取、诊断渲染和元数据写出。

这个 change 是架构整理，不改变训练算法和现有 CLI 语义。

## Goals / Non-Goals

**Goals:**

- 让轻量配置、路径和场景元数据模块可独立导入，不依赖 dataset/model 组件及其重依赖。
- 提供统一模态契约，减少新增模态或编码方式时的多处重复修改。
- 拆分 builders、transforms、visual diagnostics 的职责边界，并保留兼容公开入口。
- 用 focused tests 约束导入边界、模态解析、兼容导入路径和诊断输出稳定性。
- 更新扩展文档，明确新模型、新模态、新数据转换和诊断逻辑的落点。

**Non-Goals:**

- 不调整模型结构、KD loss、训练循环数学逻辑或默认超参数。
- 不移除现有配置文件、CLI、legacy 入口或已记录的输出格式。
- 不引入新的第三方运行时依赖。
- 不重命名用户可见的 registry type，除非同时保留兼容别名。

## Decisions

### 1. 新增中心化模态契约模块

新增 `src/kd_sensing/modalities.py`，定义稳定的 `MODALITY_ORDER` 和 `ModalitySpec`。每个 spec 至少包含：

- `name`
- `dataset_flag`，如 `use_gps`
- `sample_keys`，如 `("radar_ra", "radar_da")`
- `fusion_input_key`，如 `radar_batch`
- `model_field_defaults`，如 `gps_input_size: 3`
- `dataset_field_defaults`，如 `gps_feature_mode: relative_polar`
- `supports_cache` 与 `normalizer_artifact_key`

现有模块通过 helper 使用这些元数据：`normalize_modalities()`、`resolve_enabled_modalities()`、`dataset_flags_for_modalities()`、`batch_input_keys_for_modalities()`。这样新增模态时先更新 contract，再补具体 dataset/model/diagnostics 实现。

备选方案是继续在每个模块维护本地 tuple 和 if 分支。它改动小，但会继续让新增模态产生散点修改和语义漂移。

### 2. 轻量模块禁止通过包级 `__init__` 导入重依赖

`kd_sensing.config`、`kd_sensing.utils.paths`、`kd_sensing.data.scenes` 必须能在缺少 pandas/scipy/skimage/matplotlib 的环境中导入。做法是：

- `utils/__init__.py` 只导出轻量工具，checkpoint registry 相关函数改为从 `kd_sensing.utils.artifact_registry` 显式导入，或使用延迟 `__getattr__`。
- `data/__init__.py` 不 eager import datasets；默认组件注册仍由 `import_default_components()` 明确触发。
- `config/io.py` 只依赖轻量配置、路径和场景模块。

备选方案是在 pyproject 中把 pandas/scipy/skimage/matplotlib 作为强制依赖并要求所有环境安装。这样不能解决导入边界问题，也不利于配置/文档工具的轻量运行。

### 3. `engine/builders.py` 按职责拆分但保持 facade

保留 `kd_sensing.engine.builders` 作为兼容 facade，把实现迁移到更窄模块：

- `engine/modality_resolution.py`：启用模态推导与冲突校验。
- `engine/cache_policy.py`：cache policy 解析和 dataset knob 注入。
- `engine/data_factory.py`：dataset 和 dataloader 构建。
- `engine/run_metadata.py`：dataset、cache、throughput 和 split metadata。
- `engine/normalization_artifacts.py`：GPS/LiDAR/mmWave scaler 保存和加载。
- `engine/optim.py`：optimizer、scheduler、device 构建。

训练、评估和诊断可逐步切到新模块；旧 import 路径继续 re-export，降低一次性迁移风险。

### 4. `data/transforms.py` 先拆实现，后收窄兼容入口

创建 `data/transforms/` 包或等价子模块布局，按模态拆分实现：

- `image.py`
- `radar.py`
- `lidar.py`
- `gps.py`
- `mmwave.py`
- `cache.py` 或 `io.py`
- `normalization.py`

为了避免大范围调用点一次性改动，第一阶段可让现有 `data/transforms.py` re-export 新模块符号；第二阶段再把调用点改为窄 import。若 Python 文件和同名包不能同时存在，则先使用 `data/transform_ops/` 作为内部包，再在后续清理中迁移命名。

### 5. 诊断可视化作为子应用拆分

保持 `kd_sensing.diagnostics.modality_visualization.visualize_modalities` 和 CLI 不变，把内部逻辑拆到：

- `diagnostics/visualization/config.py`
- `diagnostics/visualization/datasets.py`
- `diagnostics/visualization/sampling.py`
- `diagnostics/visualization/stats.py`
- `diagnostics/visualization/render.py`
- `diagnostics/visualization/writers.py`

`modality_visualization.py` 可变成 facade。拆分必须保持 `summary.json`、`split_stats.json`、`samples.jsonl`、`samples.csv`、PNG 命名和 preserve-existing 行为兼容。

## Risks / Trade-offs

- 兼容 re-export 可能短期保留旧聚合文件，导致“已拆分但仍看起来很大”。缓解：tasks 中要求迁移主要调用点，并用文档标记旧入口为兼容 facade。
- 延迟导入可能影响 registry 中已注册组件列表。缓解：保留 `import_default_components()` 作为明确注册边界，并增加测试验证 builders 仍会在构建前导入默认组件。
- 拆分 transforms 容易引入循环导入。缓解：先提取无状态 helper 和常量，再迁移模态函数；scaler 和 normalizer 单独成模块。
- 诊断拆分可能改变输出顺序或路径。缓解：以现有诊断测试为基线，增加路径/metadata 兼容断言。

## Migration Plan

1. 新增模态契约和轻量导入测试，先不改行为。
2. 调整 `utils/__init__.py`、`data/__init__.py` 和相关 imports，解除配置导入对重依赖的传播。
3. 迁移 engine builders 内部实现，保持 facade re-export。
4. 拆分 transforms 并迁移 dataset/preprocessing 调用点。
5. 拆分 diagnostics visualization，保持公开函数和 CLI 不变。
6. 更新 README、extension guide 和 spec Purpose。

回滚策略：由于外部入口保持兼容，若某个拆分阶段失败，可回退该阶段内部模块迁移，同时保留已通过的轻量导入和模态契约测试。

## Open Questions

- `data/transforms.py` 是否立即改成包目录，还是先使用 `data/transform_ops/` 避免文件/目录同名迁移风险？
- 是否需要为 `ModalitySpec` 使用 frozen dataclass，还是先用 `TypedDict`/普通 dict 降低序列化和测试成本？
- 旧的 `from kd_sensing.utils import resolve_teacher_checkpoint` 是否需要长期兼容，还是只在本 change 内迁移项目内部调用点？
