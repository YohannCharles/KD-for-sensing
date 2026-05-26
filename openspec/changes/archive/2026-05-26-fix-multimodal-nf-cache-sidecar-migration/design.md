## Context

Multimodal-NF 的 image/LiDAR 读取已经从原始 gzip HDF5 优化为本地 `.npy` 派生 cache。上一轮吞吐优化把 cache sidecar schema 升级到 `multimodal_nf_derived_v2`，并在轻量校验中要求 source identity、cache schema、IO layout、bytes、shape、dtype 等字段。当前真实工作区里大量 `.npy` 数据文件已经存在，但 sidecar 仍是 `multimodal_nf_derived_v1`。结果是：

- `read_only` 会在 dataset 构建阶段直接失败，训练没有进入 GPU step；
- `auto` 会尝试重新生成/强校验，可能在训练启动前进行大量磁盘 IO；
- 用户看到的是“GPU 不占用、进程像没动”，但真实瓶颈在 cache validation/migration 阶段；
- 旧 sidecar 通常已有 `source_path`、`source_fingerprint`、shape、dtype、sample_count、seq_len、num_pred 等核心字段，缺的是新版轻量校验字段和 IO layout 字段。

这个 change 要把“旧 sidecar 升级到新 sidecar”从隐式 rebuild 变为显式、可诊断、可测试的迁移路径。

## Goals / Non-Goals

**Goals:**

- 对可安全迁移的 v1 sidecar 执行 metadata-only upgrade，不重写 `.npy` 数据文件。
- 让 `auto` 和预处理优先选择 sidecar-only upgrade；仅在必要时重建 `.npy`。
- 让 `read_only` 保持只读语义，但错误信息能明确说明 cache 是否“可迁移但尚未升级”。
- 在 profile/recommend 输出中报告 sidecar schema 分布、migration pending 数量和推荐命令。
- 保持派生 cache 读取结果、sample keys、target 字段、metric 语义和模型接口不变。
- 为 fixture 和真实路径无关的小样本测试补齐旧 sidecar 迁移覆盖。

**Non-Goals:**

- 不改变 `.npy` 派生 cache 的数据布局，不引入 Zarr/LMDB 等新存储。
- 不自动移动、删除或改写原始 Multimodal-NF HDF5。
- 不把真实 cache、profile 输出、训练日志或 checkpoint 纳入源码。
- 不把多任务模型结构或 loss 权重作为本 change 的优化目标。
- 不让 `read_only` 在训练 dataset 构建时默认写 sidecar；只读语义必须可预期。

## Decisions

### 1. 增加 sidecar-only upgrade helper

新增或扩展 cache helper，提供类似 `upgrade_derived_cache_sidecar(...)` 的内部能力。它读取旧 sidecar 和 `.npy` header，补齐：

- `version: multimodal_nf_derived_v2`
- `cache_schema_version: 2`
- `source_key`
- `source_size_bytes`
- `source_mtime_ns`
- `storage_kind: npy_mmap`
- `layout: source_contiguous_rows`
- `bytes`
- `shape`
- `dtype`
- `recommended_access_pattern`
- `validation` 记录

升级必须通过原子写入更新 JSON sidecar。升级不应读取完整 `.npy` 数据，只允许读取 `.npy` header、文件 stat 和必要的源文件 stat。若请求 strong validation，则可以重新扫描 source fingerprint；默认 lightweight 不扫描原始大文件。

备选方案：直接把旧 sidecar 全部视为 invalid 并要求 rebuild。拒绝，因为会把 metadata 兼容问题变成 100GB 级重复 IO。

### 2. 区分三类状态：valid、migration_pending、invalid

`cache_status` 或等价读取计划需要能表达旧 sidecar 是否可迁移：

- `valid`: 当前 v2 sidecar 满足指定 validation mode。
- `migration_pending`: `.npy` 存在，旧 sidecar 与当前 source/profile/split/window 参数基本一致，但缺 v2 轻量字段。
- `invalid`: 数据缺失、shape/dtype/sample_count 不匹配、source identity 不可确认、强校验不匹配或 JSON 损坏。

`read_only` 遇到 `migration_pending` 时仍失败，但错误必须说明可以运行预处理升级 sidecar，而不是笼统说 cache unavailable。`auto` 遇到 `migration_pending` 时先 sidecar-only upgrade，再重新执行 lightweight status。

备选方案：让 `read_only` 自动升级 sidecar。暂不采用，因为 `read_only` 在训练进程中自动写文件会违背用户对只读缓存的预期，也容易在多进程并发启动时产生 sidecar 写竞争。

### 3. 预处理负责批量升级，训练负责清晰诊断

`multimodal_nf_derived_cache` 预处理入口继续作为用户主动维护 cache 的工具。默认可在 `rebuild=false` 时：

1. 对 valid cache 计为 skipped/valid；
2. 对 migration_pending cache 执行 metadata-only upgrade；
3. 对 missing/invalid cache 按现有 ensure 逻辑生成或按配置失败；
4. 对 `rebuild=true` 强制重写 `.npy`。

训练 dataset 的 `auto` policy 可以作为兜底升级单个需要的 sidecar，但长训练推荐路径仍应提示先运行预处理，避免第一次训练启动时承担大量 cache 维护工作。

备选方案：只在训练时升级。拒绝，因为用户需要一个可提前运行、可看到计数的维护命令。

### 4. profile 和推荐器输出 cache migration 诊断

Multimodal-NF profile/recommend 需要输出机器可读字段，例如：

- `sidecar_schema_versions`
- `migration_pending`
- `metadata_upgrade_supported`
- `metadata_upgraded`
- `would_rebuild`
- `cache_validation_mode`
- `prewarm_command`

推荐器在发现 v1 sidecar 覆盖率高且 `.npy` 文件存在时，应推荐 metadata-only upgrade/prewarm 命令，而不是直接建议 `read_only` 或让训练现场重建。

备选方案：只改错误信息。拒绝，因为用户的直接体验是“不知道在干啥”，需要 profile/recommend 能提前暴露阶段。

### 5. 并发写入沿用原子 sidecar 写入

sidecar upgrade 使用现有原子 JSON 写入模式。若多个进程同时升级同一个 sidecar，最终内容应等价；读取方不得看到半写入 JSON。实现不需要引入锁服务。

备选方案：增加文件锁。暂不采用，除非测试或真实运行证明原子 replace 不足以覆盖并发预热场景。

## Risks / Trade-offs

- [Risk] 旧 sidecar 缺少 source size/mtime，lightweight upgrade 只能从当前 source stat 补齐，无法证明旧 `.npy` 一定来自当前 source。→ Mitigation：仅在旧 sidecar 的 source path、fingerprint 字段、profile、split、seq_len、num_pred、shape/dtype/sample_count 与 `.npy` header 基本一致时允许迁移；需要更强保证时使用 strong validation。
- [Risk] `auto` 在训练启动时升级大量 sidecar，仍可能造成可见等待。→ Mitigation：profile/recommend 明确提示先运行预处理；run metadata 记录 migration 行为和耗时。
- [Risk] 自动重建和 metadata-only upgrade 的边界不清会让用户误解 cache 是否重写。→ Mitigation：预处理和 runtime metadata 分别记录 `metadata_upgraded`、`cache_generated`、`cache_rebuilt`、`cache_fallback`。
- [Risk] 旧测试当前期待 old sidecar read_only error/auto rebuild，需要调整。→ Mitigation：保留 read_only error，调整 auto 预期为优先 metadata-only upgrade，并新增 invalid old sidecar 仍 rebuild/fail 的测试。
- [Risk] 强校验会重新扫描原始 HDF5，可能慢。→ Mitigation：strong validation 只在用户显式请求、rebuild 或审计场景触发；默认训练和 read_only lightweight 不扫描完整 source。

## Migration Plan

1. 实现 sidecar status 分类和 metadata-only upgrade helper。
2. 将预处理入口接入 upgrade 优先路径，并输出 upgraded/rebuilt/skipped/failed 汇总。
3. 将 dataset `auto` policy 接入 migration pending 升级；保持 `read_only` 清晰失败和不写文件。
4. 扩展 runtime metadata、profile 和推荐器输出 sidecar schema/migration 字段。
5. 更新 focused tests，验证迁移不改变样本读取结果且不重写 `.npy`。
6. 用真实本地数据运行一次预处理升级 train/test sidecar，作为本地操作产物，不提交。

回退策略：将 cache policy 改为 `off` 使用原始 HDF5，或将有问题的 cache 以 `rebuild=true` 重建。若 sidecar-only upgrade 发现不匹配，应拒绝升级并保留原 sidecar，避免破坏可审计线索。

## Open Questions

- 是否需要新增显式配置名如 `upgrade_metadata_only`，还是把它作为 `rebuild=false` 下的默认预处理行为？
- 推荐器是否应根据 sidecar 统计自动把 `read_only` 推荐降级为 `auto`，直到 migration pending 为 0？
- 是否需要为大型真实 cache 提供单独的只统计命令，避免 profile 构建完整 dataset 才发现 migration pending？
