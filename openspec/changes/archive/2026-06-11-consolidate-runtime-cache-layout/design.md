## Context

当前本地布局的混乱主要来自三类 cache：

1. 根目录 `cache/physical_labels`，由 MMW physical label 默认配置生成。
2. 数据集目录内的可再生成缓存，例如 `dataset/DeepSense6G/scenario31/image_derived_cache`、`dataset/DeepSense6G/scenario31/lidar_bev_cache`、`dataset/MMW/sunny/lidar_bev_cache`。
3. 已经部分存在的 `outputs/cache/...`，例如 DeepSense6G LiDAR BGAM cache。

这些缓存和真实数据的生命周期不同：cache 可以重建，真实数据不应被清理工具误删；训练/诊断输出也有独立的运行审计需求。因此本设计只改变默认路径和低风险小缓存位置，不自动迁移大目录。

## Goals / Non-Goals

**Goals:**

- 新生成的可再生 cache 默认落到 `outputs/cache/`，并按 dataset family、scene/condition 和 cache kind 分层。
- 保留显式配置路径兼容，避免旧实验、旧缓存和外部数据根失效。
- 将物理标签、image-derived 和 LiDAR BEV cache 默认路径纳入同一 layout helper。
- 更新文档、配置和 focused tests，避免未来新增默认又回到根 `cache/` 或 `dataset/*/*cache`。

**Non-Goals:**

- 不自动移动、删除或压缩 `dataset/` 中的大体量真实数据或历史 cache。
- 不改变数据 split、样本字段、checkpoint schema、训练数值语义或模型输出。
- 不把 `outputs/cache/` 中已有历史 cache 强制改名或重写。
- 不移除用户显式配置旧 cache 路径的能力。

## Decisions

### Decision 1: 默认 runtime cache 根为 `outputs/cache`

集中式 helper 提供以下默认路径：

```text
outputs/cache/DeepSense6G/scenario31/image_derived
outputs/cache/DeepSense6G/scenario31/lidar_bev
outputs/cache/MMW/sunny/image_derived
outputs/cache/MMW/sunny/lidar_bev
outputs/cache/physical_labels
```

这样 `dataset/` 中的 family/scene/condition 目录继续表达数据来源，`outputs/cache/` 表达可再生成本地产物。训练输出、评估结果和诊断报告仍继续使用 `outputs/training`、`outputs/analysis`、`outputs/diagnostics` 或既有 run dir。

### Decision 2: 显式旧路径继续可用

dataset 和预处理入口如果收到显式 `cache_dir`、`image_cache_dir` 或 `lidar_cache_dir`，必须继续使用该路径。相对路径的兼容语义保持：历史配置中的 `image_derived_cache` 或 `lidar_bev_cache` 仍解析到对应 data root 下，以免旧命令突然找不到已预热缓存。只有“未显式配置”的默认值切到 `outputs/cache/`。

### Decision 3: 只迁移低风险根 cache

根目录 `cache/physical_labels` 体量小且由代码默认生成，不是原始数据。实现可将它移动到 `outputs/cache/physical_labels`。对 `dataset/DeepSense6G/*/*cache`、`dataset/MMW/*/*cache` 和 `outputs/` 内历史实验产物不做自动迁移；后续如果要清理大目录，必须先使用 cleanup manifest 或用户明确指定路径。

### Decision 4: 清理工具识别 legacy cache 但不误删数据

runtime cleanup 可继续把根 `cache/` 视为 legacy cache 扫描候选，便于用户显式生成 manifest；同时受保护根必须继续包含 `dataset/`、`All_models/`、源码、OpenSpec 和已跟踪文件。默认推荐文档不再引导用户新建根 `cache/`。

## Risks / Trade-offs

- [Risk] 旧预热缓存在 `dataset/` 中，默认切换后导致首次运行重新生成 cache。→ Mitigation：显式旧路径仍兼容，文档说明可用 override 复用旧缓存。
- [Risk] 相对路径语义同时承担兼容和新默认，容易误解。→ Mitigation：只把“未配置默认”迁出；配置中出现的相对路径仍按历史 data root 解析。
- [Risk] 大目录迁移耗时且可能打断实验。→ Mitigation：本 change 不自动移动大目录，只移动小型根 physical label cache。

## Migration Plan

1. 新增 runtime cache layout helper。
2. 修改默认 cache 解析和配置，使新运行默认写入 `outputs/cache/`。
3. 移动 `cache/physical_labels` 到 `outputs/cache/physical_labels` 并保留 `.gitignore` 防回流规则。
4. 更新 README/docs/OpenSpec 和 focused tests。
5. 运行 OpenSpec validate 和相关路径/缓存测试。

Raymobtime s008 cache layout 曾在本 change 初稿中覆盖；后续由 `remove-raymobtime-s008` 退役删除，因此不再作为本 change 的实现目标。
