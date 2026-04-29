## Context

项目当前已经实现 image motion mask cache、LiDAR BEV cache、beam label cache、参数 hash 目录和预处理入口。但训练配置仍暴露为多个低层开关：`image_motion_use_cache`、`image_motion_write_cache`、`lidar_use_cache`、`lidar_write_cache` 等。用户并行跑大量单模态和 fusion 组合时，需要手动判断哪些组合包含 image/LiDAR、哪些 cache 已存在、是否允许写入缺失 cache，容易出现缓存未复用或多个进程重复写入同一文件。

本设计在现有 cache 能力上增加一层统一策略，不改变模型、loss、指标和已有显式低层开关的含义。所有 Python 验证命令使用 `conda run -n kd_mm_beam ...`。

## Goals / Non-Goals

**Goals:**

- 增加统一 `data.cache.policy` 配置，用于控制训练和评估是否自动读取、写入或重建预处理 cache。
- 在 dataset 构建前根据实际启用模态自动派生 image/LiDAR cache 读写开关。
- 自动策略只访问当前任务启用模态相关的 cache，非 image/LiDAR 组合不得因为这些 cache 缺失而失败。
- 对 cache 写入使用原子写入或等价保护，降低并行实验中半成品文件被读取的风险。
- 在最终配置和运行 metadata 中记录实际生效 cache policy、cache 目录和读写状态。

**Non-Goals:**

- 不引入新的缓存格式；继续复用现有 image pair-level `.npy` 和 LiDAR BEV `.npy`。
- 不强制训练启动前全量预热 cache；`auto` 可在训练取样时按需写入。
- 不为 GPS/mmWave 增加大规模磁盘 cache；它们仍主要依赖轻量 scaler/normalization artifact。
- 不改变已有显式低层开关作为高级用户覆盖入口的能力。

## Decisions

1. 增加统一 policy，而不是让用户继续直接组合多个低层开关。

   `data.cache.policy` 支持 `off`、`read_only`、`auto`、`rebuild`。`off` 禁用 image/LiDAR 磁盘 cache；`read_only` 只读已有 cache、缺失时在线计算但不写；`auto` 读已有 cache、缺失时按需写入；`rebuild` 读写开启并覆盖或重算缺失/已有 cache。保留 `data.cache.image.policy` 和 `data.cache.lidar.policy` 作为可选模态级覆盖。

2. policy 在 dataloader 构建前解析为现有 dataset 字段。

   实现上不把 policy 判断散落在 `__getitem__`。`build_dataset` 先推导启用模态，再把 policy 解析成 `image_motion_use_cache`、`image_motion_write_cache`、`image_motion_overwrite_cache`、`lidar_use_cache`、`lidar_write_cache`、`lidar_overwrite_cache` 等现有或新增字段。这样 Scenario9Dataset 仍只处理具体读写行为。

3. 显式低层覆盖优先于自动派生。

   如果用户直接在命令行或 YAML 中设置了低层字段，系统尊重显式值；未显式设置时由 policy 派生。为避免难以判断“默认值是否显式”，实现可接受一个简单规则：默认配置声明 `data.cache.enabled: true` 且 canonical 配置不再重复声明低层读写开关，policy 负责常规路径；用户仍可通过 `-o data.dataset.<field>=...` 强制覆盖。

4. 写入使用原子 replace。

   image motion mask 和 LiDAR BEV cache 写入先写入同目录临时文件，完成后用 `os.replace` 或等价原子替换目标。读取方只读取最终目标文件；遇到损坏 cache 时应清晰报错或按 policy 重新生成，不能静默返回错误 tensor。

5. 运行 metadata 记录“生效值”而不是只记录原始配置。

   `final_config.yaml` 的 `runtime.cache` 记录全局 policy、每模态 policy、启用模态、实际 cache 目录、实际读写/覆盖开关和参数 hash 目录。profile 或评估输出也应记录同样摘要，方便比较不同实验是否真的复用了 cache。

## Risks / Trade-offs

- [Risk] `auto` 在多进程并发下仍可能重复计算同一缺失 cache → 使用原子 replace 避免半成品读取，允许少量重复计算作为简单可靠的首版策略。
- [Risk] 自动写入会增加磁盘占用 → README 明确 cache 目录和清理方式，`read_only` 可用于只复用已有 cache。
- [Risk] 低层开关与 policy 同时配置时用户难以理解 → 文档说明优先级，并在 runtime metadata 记录最终生效值。
- [Risk] `rebuild` 误覆盖正在复用的 cache → 默认不使用 `rebuild`，并要求该模式显式配置。

## Migration Plan

- 默认 policy 设为 `read_only` 或 `auto`，在不改变训练数值语义的前提下让包含 image/LiDAR 的任务自动复用 cache。
- 将 README 中手动 cache 开关示例更新为 policy 示例，同时保留低层覆盖说明。
- 补充测试覆盖 policy 解析、按模态访问、cache miss 写入、非相关模态不访问 cache 和原子写入。
