## Context

当前代码把 DeepSense6G 的可训练数据集实现命名为 `Scenario9Dataset`，并以 registry 名称 `scenario9` 构建。大多数 YAML 显式设置 `data.dataset.type: scenario9`、`data.dataset.data_root: dataset/scenario9`，训练输出由 `create_run_dir()` 写入 `outputs/<run_name>/`，最佳 checkpoint registry 默认写入 `outputs/best_checkpoints/`。这使 Scenario 9 和未来场景共享同一输出命名空间，也使 KD 配置的默认 teacher 权重路径与场景无关。

这个变更要在不破坏现有 Scenario 9 配置和训练入口的前提下，加入 DeepSense6G 场景选择，并把现有训练产物归入 `scene9`。

## Goals / Non-Goals

**Goals:**

- 支持通过配置和命令行覆盖选择 DeepSense6G 场景，首批支持 9 和 32。
- 新默认场景为 Scenario 32；旧 `scenario9` 配置仍明确选择 Scenario 9。
- 将数据集实现从“只有 Scenario 9”的命名约束中解耦，形成可扩展的场景注册/解析层。
- 训练、评估、TensorBoard、checkpoint、registry 和 KD teacher 默认路径按场景隔离。
- 保留旧配置 `type: scenario9`、旧 CSV 名和显式绝对路径覆盖语义。
- 将当前 `outputs/<run_name>/` 和 `outputs/best_checkpoints/` 迁移到 `outputs/scene9/` 下。

**Non-Goals:**

- 不为 Scenario 32 新增预处理算法或补齐缺失原始数据。
- 不改变模型结构、损失函数、指标计算或现有多模态 batch 契约。
- 不引入外部配置系统或新依赖。

## Decisions

### 1. 使用 `data.dataset.scene` 作为场景选择字段

`data.dataset.scene` 接受整数或字符串别名，例如 `9`、`"9"`、`"scene9"`、`"scenario9"`、`32`、`"scene32"`、`"scenario32"`。新默认值为 `32`。当旧配置显式使用 `data.dataset.type: scenario9` 且未提供 `scene` 时，系统将其作为兼容入口解析为 Scenario 9，而不是套用全局默认值。

原因：场景属于 DeepSense6G dataset 的数据选择，而不是模型或训练超参数。放在 `data.dataset` 下可以继续使用现有 dotted override：`data.dataset.scene=32`。

替代方案是新增顶层 `experiment.scene`。该方案更短，但会让数据路径解析和输出路径解析依赖不同配置分支，后续扩展多数据集时更容易混淆。

### 2. 新增 DeepSense6G 场景描述符和兼容 registry 名

实现上新增轻量场景描述符，至少包含：

- `scene_id`: 规范整数，如 `9`
- `scene_slug`: 输出目录使用的稳定短名，如 `scene9`
- `dataset_aliases`: 可解析别名，如 `scenario9`
- `default_data_root`: 如 `dataset/scenario9`
- `default_train_csv_name` / `default_test_csv_name`

数据集 registry 应新增通用 `deepsense6g` 类型，并继续注册 `scenario9`。`scenario9` 构建时隐式选择 scene 9；`scenario32` 可以作为 scene 32 的兼容别名。现有 `Scenario9Dataset` 类可以先保留为导出别名，内部委托或继承通用 `DeepSense6GDataset`，减少测试和导入变更。

### 3. 默认数据路径由场景解析，但显式配置优先

当用户未显式提供 `data.dataset.data_root`、`train_csv_name` 或 `test_csv_name` 时，系统使用场景描述符的默认值。显式提供的路径和 CSV 文件名必须保持最高优先级，便于用户测试自定义 split 或临时数据目录。

因为当前 `DEFAULT_CONFIG` 和 YAML 已显式写了 `dataset/scenario9`，实现时需要区分“旧默认值”和“用户显式覆盖”。保守做法是先批量更新 canonical YAML 和默认配置：改为通用 `deepsense6g` 类型和 `scene: 32`，删除或同步旧的硬编码路径；保留 `type: scenario9` 作为显式兼容配置时解析为 Scenario 9。配置加载后统一调用场景 resolver 补齐缺失默认值。

### 4. 输出目录按 `scene_slug` 分组

配置中的 `output.dir` 解释为输出根目录，默认仍是 `outputs`。当构建 DeepSense6G 场景化运行时，训练和默认评估输出目录解析为：

```text
<output.dir>/<scene_slug>/<run_name>
```

因此新默认训练写入 `outputs/scene32/<run_name>/`；显式选择 Scenario 9 时写入 `outputs/scene9/<run_name>/`。显式传给评估入口的 `--output-dir` 仍作为完整目录使用，不再自动追加场景层；这是为了保留脚本化评估的确定性路径。

### 5. Registry 和 KD teacher 解析按场景隔离

默认 checkpoint registry 从 `outputs/best_checkpoints/` 调整为按场景解析的 `outputs/<scene_slug>/best_checkpoints/`。KD teacher 默认解析先查当前场景 registry，再查当前场景下的 `paths.weights_dir`。为了迁移期间更稳妥，可以保留旧路径 fallback，但 canonical YAML 必须指向当前默认场景的 `outputs/scene32/...`；历史 Scenario 9 训练结果迁移到 `outputs/scene9/...`。

metadata sidecar 需要记录 `scene_id` 和 `scene_slug`，防止同名 run 在不同场景中误匹配。

## Risks / Trade-offs

- [Risk] 历史 sidecar 中保存的 `run_dir` 和 `path` 在移动目录后失效。→ 迁移时同步重写 JSON sidecar 中的路径字段，或在最终说明中明确哪些历史 metadata 仅供参考。
- [Risk] 旧配置显式写死 `output.dir: outputs` 后行为改变为多一层场景目录，且新默认场景变为 Scenario 32。→ 这是本变更的目标行为；resume 文档和测试必须更新，旧目录由迁移步骤移动到 `scene9`。
- [Risk] 用户已有自定义输出目录不希望按场景分组。→ 增加 `output.group_by_scene: true` 默认值，允许用户显式关闭；评估 `--output-dir` 不自动追加场景。
- [Risk] Scenario 32 的 CSV 命名可能与 Scenario 9 不完全一致。→ 场景描述符只提供默认值，用户仍可覆盖 `train_csv_name` 和 `test_csv_name`。

## Migration Plan

1. 实现并测试场景解析、输出目录解析和 registry 场景隔离。
2. 批量更新默认配置和 canonical YAML，使默认 DeepSense6G 配置写入 `scene: 32` 并默认输出到 `outputs/scene32/`。
3. 将当前 `outputs/<run_name>/` 目录移动到 `outputs/scene9/<run_name>/`，将 `outputs/best_checkpoints/` 移动到 `outputs/scene9/best_checkpoints/`。
4. 更新 KD 配置的 `paths.weights_dir` 或解析逻辑，使新默认 teacher 权重指向 scene32 下的训练结果；历史 Scenario 9 配置继续指向 scene9 下的迁移结果。
5. 运行 `conda run -n kd_mm_beam pytest ...` 覆盖配置、训练目录、registry 和数据集构建测试。

## Open Questions

- Scenario 32 的默认 split CSV 是否沿用 `train_seqs_RA_GPS_LIDAR.csv` / `test_seqs_RA_GPS_LIDAR.csv`。当前设计先沿用，并允许配置覆盖。
- 是否需要为 `outputs/scene9` 下的历史 sidecar 做完全路径重写。实现阶段应先检查现有 sidecar 内容后决定。
