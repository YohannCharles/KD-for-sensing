## Context

当前配置加载流程是 `DEFAULT_CONFIG + 单个 YAML + CLI override`，只有磁盘上存在的 YAML 才能作为 `--config` 输入。这个模型在单模态阶段足够简单，但 fusion canonical 矩阵已经展开为 26 个多模态 slug × 4 个模式，再加 legacy 入口，导致 `configs/fusion/` 出现大量重复内容。

现有 OpenSpec 要求完整 canonical 矩阵存在，但用户真正依赖的是稳定路径和语义，例如：

```text
configs/fusion/gps_mmwave_logits_kd.yaml
        │      └──────────── mode = logits_kd
        └────────────────── slug = gps_mmwave
```

这个路径本身已经包含足够信息来生成大部分配置。保留路径语义、减少实体文件，是比继续维护 YAML 展开矩阵更合适的方向。

## Goals / Non-Goals

**Goals:**

- 让 canonical fusion 配置可以由 loader 从路径生成，不要求每个路径都有实体 YAML。
- 保持现有命令形态兼容：`--config configs/fusion/<slug>_<mode>.yaml` 继续可用。
- 固定并校验 slug 命名顺序：`image > radar > gps > lidar > mmwave`。
- 保留 legacy 入口和特殊配置文件，不强制删除用户自定义或 ablation 配置。
- 将测试契约从 `Path.exists()` 调整为 `load_config(path)` 能解析且语义正确。
- 保存训练产物时仍写出完整 `final_config.yaml`，避免虚拟配置影响复现实验。

**Non-Goals:**

- 不引入 Hydra、OmegaConf 或其它外部配置框架。
- 不改变模型、dataset、KD loss、checkpoint registry 的运行语义。
- 不在本次变更中生成单模态虚拟配置；单模态 YAML 数量有限，先保持实体文件。
- 不删除 legacy alias，例如 `configs/fusion/no_kd.yaml`、`configs/fusion/all_modalities_lidar_no_kd.yaml`。

## Decisions

### 1. 在配置加载器内实现虚拟 canonical resolver

`load_config(config_path, overrides)` 会先解析 `config_path`。如果路径存在，继续按实体 YAML 加载；如果路径不存在且符合 `configs/fusion/<stem>.yaml`，尝试按 canonical fusion 规则生成 override dict，再与 `DEFAULT_CONFIG` 合并。

这样做的理由：

- CLI、测试和 Python API 不需要增加新的参数。
- 旧的实体配置、legacy 配置和自定义 YAML 保持优先。
- 用户仍然可以通过创建实体 YAML 覆盖虚拟生成逻辑。

备选方案是新增 `--preset fusion:gps_mmwave:logits_kd`，但这会引入第二套入口，和当前 README 推荐路径不一致。

### 2. 由文件 stem 解析 slug 和 mode

resolver 支持四个 mode 后缀：

```text
_teacher_no_kd
_student_no_kd
_logits_kd
_rkd
```

解析步骤：

1. 从 stem 末尾匹配 mode 后缀。
2. 去掉后缀得到 slug。
3. 将 slug 按 `_` 拆分为模态列表。
4. 校验模态必须来自 `image, radar, gps, lidar, mmwave`。
5. 校验至少两个模态，避免和单模态配置重复。
6. 校验列表必须严格按 canonical 顺序排列，乱序时抛出错误并提示正确 slug。

例如 `mmwave_gps_logits_kd.yaml` 应拒绝，并提示使用 `gps_mmwave_logits_kd.yaml`。

### 3. 用小型规则表生成配置 override

生成器只构造相对 `DEFAULT_CONFIG` 的 override，而不是复制完整配置。核心规则：

- `experiment.name` 和 `output.run_name` 等于文件 stem。
- `experiment.task` 为 `fusion`。
- 非 image+radar fusion 延续当前扩展配置默认：`seed: 0`、teacher/student GRU 均为 `[64, 64, 2]`、训练参数 `lr: 0.00075`、`weight_decay: 0.0001`、`temperature: 3.0`、`alpha: 0.4`。
- image+radar fusion 保持上游兼容语义：teacher GRU `[64, 64, 2]`，student GRU `[64, 64, 1]`，并保留既有训练/KD 参数和 `All_models` teacher checkpoint 默认来源。
- KD 模式默认 teacher checkpoint：
  - `image_radar_{logits_kd,rkd}` 使用 `paths.weights_dir: All_models` 和 `BothTeacher_best.pth`。
  - 其它 fusion KD 使用 `outputs/scene32/<slug>_teacher_no_kd/checkpoints/best.pth`。
- 根据模态补充字段：
  - GPS：`data.dataset.use_gps: true`、`gps_feature_mode: relative_polar`、`gps_normalize: true`、`gps_input_size: 3`。
  - LiDAR：`data.dataset.use_lidar: true`、BEV size/ROI/normalize 默认、`lidar_channels: 3`。
  - mmWave：`data.dataset.use_mmwave: true`、`mmwave_normalize: true`、`mmwave_input_size: 64`。
  - image/radar 不额外设置 dataset flag，保持现有默认行为。

### 4. 清理实体 YAML 但保留兼容入口

实现生成器并验证等价后，可删除 `configs/fusion/` 下所有可生成 canonical 文件：

```text
<slug>_teacher_no_kd.yaml
<slug>_student_no_kd.yaml
<slug>_logits_kd.yaml
<slug>_rkd.yaml
```

保留：

- `configs/fusion/no_kd.yaml`
- `configs/fusion/logits_kd.yaml`
- `configs/fusion/rkd.yaml`
- `configs/fusion/image_gps_no_kd.yaml`
- `configs/fusion/radar_gps_no_kd.yaml`
- `configs/fusion/radar_lidar_no_kd.yaml`
- `configs/fusion/all_modalities_no_kd.yaml`
- `configs/fusion/all_modalities_lidar_no_kd.yaml`

这些是 legacy alias，继续由实体 YAML 表达历史语义。未来如果需要，也可以再把 legacy alias 映射到 canonical resolver，但本次不做。

### 5. 测试验证加载语义而不是文件存在

测试中的 canonical 矩阵应来自共享常量或 helper，而不是扫描实体文件。断言重点改为：

- `load_config(ROOT / "configs/fusion/<slug>_<mode>.yaml")` 成功。
- `experiment.name`、`output.run_name`、`modalities`、KD mode、checkpoint 来源和模态字段正确。
- 乱序 slug、未知模态、单模态 fusion virtual path 抛出清晰错误。
- 删除重复 YAML 后，canonical 矩阵测试仍能覆盖 26 × 4 个路径。

## Risks / Trade-offs

- [Risk] 生成规则与旧 YAML 不完全等价，导致训练参数静默漂移。 → Mitigation：实现阶段先在删除文件前对代表性旧 YAML 与生成配置做等价测试，覆盖 image+radar、GPS、LiDAR、mmWave、三/四/五模态组合。
- [Risk] 用户误以为所有不存在的 YAML 都能自动生成。 → Mitigation：resolver 仅接受 `configs/fusion/<canonical>.yaml`，其它路径继续报缺失文件。
- [Risk] 物理文件优先会让同名自定义 YAML 覆盖虚拟规则，行为来源不明显。 → Mitigation：文档说明“实体 YAML 优先，缺失时才生成 canonical fusion 配置”，训练输出保存完整最终配置。
- [Risk] `outputs/scene32/...` teacher path 仍是静态默认，不随 `data.dataset.scene` override 自动变化。 → Mitigation：保持现有 YAML 语义不变；需要跨场景时继续使用 checkpoint registry 或显式 override。
- [Risk] README 中“查看 configs/fusion/”无法再通过文件列表展示完整矩阵。 → Mitigation：文档改为列出命名规则和受支持模态组合，测试作为契约来源。
