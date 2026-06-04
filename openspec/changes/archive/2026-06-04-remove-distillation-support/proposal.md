## Why

历史 KD 已经从当前主线隔离，但源码、配置 schema、测试和文档仍保留 `logits_kd`、`rkd`、`distillation.type: no_kd`、teacher checkpoint 解析和 distiller registry 等表面。继续维护这些入口会扩大配置矩阵、增加运行时分支，并让项目叙述停留在 KD-first，而当前研究主线已经转向 supervised/adaptation、HiST-Beam、MMW 和 history-anchored workflow。

## What Changes

- **BREAKING** 删除 teacher-student 蒸馏能力：移除 `logits_kd`、`rkd`、distiller/loss 模块、teacher checkpoint 训练解析、distillation optimizer 参数组、KD loss 日志和 legacy KD summary eligibility 逻辑。
- **BREAKING** 删除源码配置中的 `distillation` 配置块；当前训练默认变为 supervised/adaptation base loss，soft beam label 继续作为 supervised label smoothing，不再经过 distillation 命名或分支。
- **BREAKING** 删除 `configs/**/logits_kd.yaml`、`configs/**/rkd.yaml` 和 fusion KD virtual/overlay 入口；缺失的 KD 路径必须失败，不得回退为 supervised 配置。
- **BREAKING** 将推荐配置和文档从 `*_no_kd`/KD lifecycle 命名迁移到 distillation-free 的 strong/lightweight/supervised 命名；不新增旧路径兼容包装。
- 保留强模型与轻量模型能力本身，但把它们作为 supervised baseline 或 adaptation backbone，而不是 teacher/student KD runtime。
- 更新 README、docs、OpenSpec、pyproject 描述、架构边界测试、配置加载测试、训练/评估 smoke 和 summary metadata，使项目表面不再声明 KD baseline 可运行。
- 不自动删除 `outputs/`、`logs/`、`All_models/` 或历史 checkpoint；如需清理本地产物，应走已有 runtime cleanup manifest 流程。

## Capabilities

### New Capabilities

- `distillation-free-project-surface`: 定义项目删除 KD 后的配置、运行时、metadata、文档和拒绝旧入口的全局契约。

### Modified Capabilities

- `project-architecture`: 包结构不再要求 `kd_sensing.distillation`，轻量导入和健康检查不再覆盖 distillation 工具子模块。
- `component-registry`: 组件注册表不再暴露 `DISTILLERS` 或内置 distiller 注册；`logits_kd`、`rkd`、`g2d` 作为已删除组件被拒绝。
- `canonical-config-resolution`: canonical/virtual 配置模式收敛到 supervised strong/lightweight 与当前 overlay；`logits_kd`、`rkd` 和 `distillation` 字段不再生成。
- `configurable-multimodal-fusion`: fusion canonical 矩阵不再要求 KD 配置路径或 RKD 兼容输出，只保留 supervised/adaptation fusion 配置。
- `experiment-workflow`: 训练、评估、quickstart、运行日志、TensorBoard 和 metadata 不再包含 KD 模式、teacher checkpoint 或蒸馏 loss。
- `experiment-artifact-registry`: checkpoint registry 不再服务 KD teacher 加载；保留普通评估权重和归一化 artifact 解析。
- `legacy-kd-isolation`: 从“隔离保留 legacy KD”改为“删除并拒绝 legacy KD”。
- `hist-beam-cross-scene-adaptation`: HiST-Beam 不再允许 KD baseline 或增强作为显式 profile。
- `soft-beam-label-training`: soft label 只作为 supervised beam smoothing，不再定义与 KD 共存或历史 KD 命名兼容。
- `radar-teacher-model`: radar 强模型不再承担 frozen teacher 蒸馏角色。
- `radar-student-model`: radar 轻量模型不再描述为 KD student。
- `mmwave-modality-model`: mmWave 配置不再提供 logits KD/RKD 兼容要求。
- `gps-modality-model`: GPS 强/轻量模型与配置不再保留 KD 命名或 checkpoint 依赖。
- `lidar-modality-model`: LiDAR 强/轻量模型与配置不再保留 KD 命名或 checkpoint 依赖。
- `cls-token-transformer-fusion`: 输出特征契约不再以 KD 兼容为目的。
- `resnet18-image-encoder`: image encoder 契约不再声明 distillation workflow 兼容性。
- `multi-task-occlusion-position-learning`: 多任务基础 loss 不再引用 beam/KD 基础 loss。
- `snapshot-next-frame-baselines`: snapshot baseline 不再校验 `distillation.type: no_kd`。
- `modality-aware-data-loading`: batch/label 对齐描述不再包含 KD 相关 loss。
- `original-code-compatibility`: 原始兼容叙述不再保留 KD 训练模式。
- `csi-channel-degradation`: CSI hardening 配置命名和场景描述去除 no-KD/KD 语义。
- `csi-modality-model`: CSI-only 配置不再通过 no-KD 命名表达 supervised baseline。
- `lidar-preprocessing`: 预处理或示例配置不再依赖 no-KD 命名。

## Impact

- 源码：`src/kd_sensing/distillation/`、`src/kd_sensing/registries.py`、`src/kd_sensing/config/*`、`src/kd_sensing/engine/*`、`src/kd_sensing/utils/artifact_registry.py`、模型注册名/配置 normalization、summary metadata 和 CLI help。
- 配置：删除 KD 实体 YAML；迁移单模态、fusion、snapshot、CSI、Raymobtime、HiST-Beam 和 MMW 配置中的 `distillation` block 与 `*_no_kd` 命名。
- 测试：重写配置矩阵、架构边界、训练 IO、registry、modality、prediction objective、HiST-Beam、snapshot 和 docs surface 检查，新增旧 KD 入口拒绝测试。
- 文档/OpenSpec：README、experiment matrix、extension guide、research notes、project surface inventory 和相关 specs 需要同步去 KD-first 化。
- 运行产物：历史 `outputs/`、`logs/`、checkpoint 和 archive 不作为本 change 自动删除对象；历史 artifact 中已有 `distillation_*` 字段只作为旧结果读取，不作为新训练输出契约。
