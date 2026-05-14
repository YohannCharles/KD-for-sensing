## Context

当前项目已经具备 `resnet18_imagenet_rgb` encoder 和 LiDAR BEV 数据路径，但默认 image teacher 仍使用从头训练的小 CNN；这会让 camera-only baseline 与论文中使用 ImageNet 预训练 ResNet-18 的 baseline 不可比。LiDAR 侧已经支持 BEV、ROI、cache 和 streaming stats，但默认 LiDAR teacher 配置未显式启用训练集统计归一化，且缺少输入质量诊断，导致 Scenario 31 上的 LiDAR-only 结果可能退化到多数类猜测而不易定位。

这次变更跨配置、模型构建、数据预处理、训练 metadata 和测试，因此需要先明确默认行为与验证边界。

## Goals / Non-Goals

**Goals:**

- 将默认 camera/image baseline 改为 `rgb_imagenet` + ImageNet 预训练 ResNet-18，并让 image-only 与包含 image 的 canonical fusion 配置使用一致的 camera encoder。
- 不再发货旧从头训练小 CNN image 配置入口，避免默认或 ablation 名称继续误用旧 camera encoder。
- 修复 LiDAR baseline 的默认训练剖面：显式启用训练集 streaming stats normalizer、可复用 BEV cache、可追踪 ROI/FoV 参数和安全增强。
- 增加 LiDAR 输入质量和退化检测，使训练报告能看出 BEV 是否为空、通道是否近似常量，以及 LiDAR 模型是否只达到多数类 baseline。
- 增加测试覆盖，防止后续默认配置回退到从头训练 camera encoder 或未归一化 LiDAR baseline。

**Non-Goals:**

- 不重写全部 fusion 架构，也不在本变更中引入新的多模态路由策略。
- 不保证所有场景上的 LiDAR Top-1 达到固定绝对值；验证目标是消除静默退化，并用统一报告衡量是否优于多数类/退化 baseline。
- 不移除现有 LiDAR BEV 数据实现；旧模型注册类如仍存在，仅作为内部兼容实现，不作为默认/canonical 配置入口。
- 不把 ResNet-18 预训练权重 vendoring 到仓库中；依赖仍由 `torchvision` 在运行环境中提供。

## Decisions

### 1. 默认 camera baseline 使用 modular ResNet-18，而不是改造旧 `image_teacher`

默认 image teacher/no-KD 配置将使用 `modular_sequence` + `resnet18_imagenet_rgb` encoder，训练策略采用保守微调：`pretrained: true`、`weights: DEFAULT`、`freeze_backbone: true`、解冻 `layer4` 或等价最后 stage、训练 projection/GRU/head。

理由：项目已经存在 ResNet-18 encoder 契约和 modular sequence 路径，直接复用能避免把预训练 backbone 硬塞进旧 `image_teacher` 类。旧小 CNN 不再作为配置入口发货，避免默认矩阵继续静默选择旧 encoder。

备选方案：在 `ImageModalityNet` 内部替换 CNN 为 ResNet-18。该方案会让类名与历史语义不一致，也更容易破坏旧 checkpoint 和 tests，因此不采用。

### 2. 包含 image 的 canonical fusion 配置复用同一 image encoder 规格

包含 image 或 LiDAR 的 fusion teacher/student canonical 配置必须显式声明 modular encoder profile。对于论文式强 camera baseline 和 teacher 初始化路径，默认使用 ResNet-18；LiDAR 默认使用 `lidar_cnn` encoder，不再让默认/canonical 配置隐式落回旧 image/LiDAR 模型类。

理由：当前比较里 image-only 和 fusion image branch 容易使用不同 encoder，导致单模态性能与融合结果难以解释。

### 3. LiDAR 修复优先从输入质量和默认剖面入手

LiDAR 默认 baseline 配置将显式启用：

- `lidar_normalization.enabled: true`
- `lidar_normalization.mode: streaming_stats`
- 参数 hash 隔离的 `lidar_cache_dir`
- 可追踪 `lidar_roi`、`lidar_fov_degrees`、ground/background 参数
- 安全增强仅限 point dropout 和小幅 jitter

理由：当前 BEV 构造已经有 height/intensity/density 三通道，但未归一化、ROI/FoV 不可见或 cache 参数混用都会让模型看到低质量或近常量输入。先把输入管线稳定下来，比立即换复杂 PointNet/稀疏卷积风险更低。

备选方案：直接新增 PointNet-style raw point encoder。该方案可能更接近论文描述，但实现和训练成本更高；本变更可以把它列为后续 ablation，而不是第一阶段默认修复。

### 4. 训练报告增加退化基线，而不是只看 TensorBoard `accuracy/val`

训练/评估报告需要记录 per-horizon Top-K、跨 horizon 平均 Top-K、majority-class baseline、last-beam baseline，以及 LiDAR 输入质量摘要。LiDAR 模型若未超过多数类 baseline，报告必须显式标记为退化风险。

理由：TensorBoard 当前 `accuracy/val` 只取第一个未来 horizon，容易被误读。LiDAR 修复需要能回答“模型是否学到了 LiDAR 输入”，而不仅是“曲线有没有动”。

## Risks / Trade-offs

- [Risk] ResNet-18 预训练权重下载或 `torchvision` 版本不可用 → Mitigation: 构建阶段保留清晰错误；测试覆盖 `pretrained: false` 和配置结构，训练环境使用 `conda run -n kd_mm_beam` 验证。
- [Risk] 解冻过多 ResNet stage 导致小样本过拟合 → Mitigation: 默认冻结 backbone，仅解冻最后 stage；提供配置覆盖做 ablation。
- [Risk] LiDAR streaming stats 预扫描增加训练启动时间 → Mitigation: cache/stats 文件可复用，并在进度日志中明确当前处于 stats fit 阶段。
- [Risk] LiDAR 仍可能受 ROI/FoV 或传感器同步影响而低于预期 → Mitigation: 输出 BEV 非空率、通道统计和退化 baseline，先定位是输入问题还是模型容量问题。
- [Risk] 默认配置变化影响历史实验可比性 → Mitigation: 运行输出 metadata 记录 encoder/preprocessing profile；历史结果通过既有输出目录追溯，不再通过旧配置入口继续扩散。

## Migration Plan

1. 将默认 image teacher/no-KD 和相关 canonical image 配置切到 modular ResNet-18 路径。
2. 移除 legacy image CNN 配置入口，并将 image student/KD 配置同步到 modular ResNet-18。
3. 更新包含 image 的 fusion canonical 配置生成逻辑，使 image branch 默认使用 ResNet-18 profile 或显式轻量 profile。
4. 更新 LiDAR teacher/no-KD 默认配置，显式启用 streaming stats normalizer、cache 和质量诊断。
5. 在训练/评估 metadata 中记录 image encoder、ResNet 训练策略、LiDAR normalizer/stats、cache 参数、BEV 质量摘要和退化 baseline。
6. 跑 unit tests、配置加载 tests、短训练 smoke tests，再跑 Scenario 31 image/LiDAR 单模态 sanity 实验。

Rollback 策略：如确需复现实验，可从历史输出目录或 git 历史恢复旧配置；当前默认/canonical 配置不保留旧 image/LiDAR encoder 入口。

## Open Questions

- 默认 ResNet-18 是否只解冻 `layer4`，还是使用 `freeze_backbone: true` 完全冻结 backbone 作为最稳 baseline，需要通过短训练对比确认。
- LiDAR 默认 FoV 是否应按 Scenario 31 专门收窄，还是保持通用 ROI 并仅通过诊断提示异常，需要先检查 BEV 可视化和非空率。
- 是否在后续变更中新增 PointNet-style raw point LiDAR encoder，用于更严格对齐论文实现。
