## Context

当前 `Scenario9Dataset.__getitem__` 总是读取 motion mask、RA、DA 和 beam，即使配置是 GPS-only 或 LiDAR-only；这会让多个并行训练在 CPU/I/O 上相互放大抖动。`build_dataloaders` 只透传 `num_workers`，缺少 `persistent_workers`、`pin_memory`、`prefetch_factor` 等可调项。训练输出目录在设置固定 `output.run_name` 时会复用同一目录，评估固定写入 `outputs/evaluation`，会造成新旧产物混杂。LiDAR 已有逐样本 cache 能力，但默认配置没有系统性复用。

## Goals / Non-Goals

**Goals:**

- 让 dataset、batch 准备和 DataLoader 都由任务/模态选择驱动，只读取当前实验需要的输入字段。
- 保持现有 image、radar、gps、lidar 和 fusion 模型 forward 契约不变。
- 让训练和评估默认写入互不覆盖的运行目录，同时保留显式 resume 到固定目录的能力。
- 让 LiDAR BEV cache 成为 LiDAR 实验的推荐默认路径，并继续保持逐样本懒加载。
- 用测试覆盖 I/O 选择、输出目录隔离、`num_pred=1` 标签维度和短训练路径。

**Non-Goals:**

- 不改变模型结构、损失函数、蒸馏算法或指标定义。
- 不新增第三方依赖，不引入独立数据服务或异步预取框架。
- 不新增并维护多套冗余 compare 配置；默认实验配置直接使用统一 split。
- 不把 LiDAR BEV 全量预加载到内存中，也不要求所有用户预先生成 cache。

## Decisions

1. **在 builder 层推导 `enabled_modalities`，dataset 只执行选择结果。**  
   `build_dataset` 根据 `experiment.task`、fusion teacher/student `modalities` 以及显式 dataset 开关生成有序模态集合，并传入 `Scenario9Dataset`。这样任务语义集中在 builder，dataset 不需要理解 teacher/student 角色。备选方案是在 dataset 内读取完整 cfg，但会扩大 dataset 对全局配置结构的耦合。

2. **保留样本元数据解析，推迟文件 I/O 到启用模态分支。**  
   `create_samples` 仍解析 CSV 列与路径列表；`__getitem__` 只对启用的 image/radar/gps/lidar 调用对应加载函数，并始终加载 `input_beam` 和 `target_beam`。这样改动局部且保持 CSV 兼容。备选方案是为每种模态建立独立 Dataset 类，但会复制标签、split、normalizer 与缓存逻辑。

3. **统一 DataLoader 参数构造函数。**  
   新增内部 helper 从 `data.dataloader` 提取 batch size、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、`drop_last` 等字段，并在 `num_workers=0` 时避免传入 worker-only 参数。训练和评估共用该 helper，减少配置漂移。

4. **输出目录默认唯一，resume 显式固定。**  
   `create_run_dir` 在没有 `output.run_name` 时继续使用时间戳；在设置 `run_name` 时默认追加 run id 或时间戳，除非配置显式要求 `output.overwrite` 或 `training.resume=true`。评估也使用同样的 run id 策略。备选方案是遇到已有目录直接报错，但会影响当前固定 run_name 的快速迭代习惯。

5. **默认配置统一到同一组 train/test CSV。**  
   所有单模态和 fusion 实验配置指向 `train_seqs_RA_GPS_LIDAR.csv` / `test_seqs_RA_GPS_LIDAR.csv`，训练/评估在 `final_config.yaml` 和报告中记录实际 split 路径与样本数。

6. **LiDAR cache 是逐样本磁盘 cache，不是内存 materialize。**  
   默认 LiDAR 配置启用 `lidar_use_cache`，并为 `lidar_cache_dir` 提供稳定路径；可选 `lidar_write_cache` 用于首次训练或预处理后补齐 cache。内存 LRU cache 保持可选且有上限。

## Risks / Trade-offs

- **[Risk] 模态推导不一致导致某些 fusion KD 配置缺输入** → 在 builder 中对 teacher/student `modalities` 做一致性校验，并用测试覆盖 single、dual、all-modality fusion。
- **[Risk] 唯一目录策略影响 resume 路径** → `training.resume=true` 必须要求固定 `output.run_name` 或显式 checkpoint 路径；resume 时不追加新 run id。
- **[Risk] `persistent_workers` 在 `num_workers=0` 时无效或报错** → helper 根据 worker 数量过滤参数，并测试零 worker 与多 worker 配置。
- **[Risk] LiDAR cache 首次生成仍有 I/O 峰值** → 文档和配置提供预处理入口；训练路径只按样本写入/读取，不做全量扫描。
- **[Risk] 统一 split 改变既有历史指标口径** → 以统一 CSV 和输出报告中的样本数作为新的横向比较基准。

## Migration Plan

1. 新增 `enabled_modalities` 推导与 dataset 懒加载实现，先保持默认 image+radar 行为兼容。
2. 更新 DataLoader 配置解析和默认字段，验证 `num_workers=0` 与当前默认 worker 配置。
3. 更新训练/评估运行目录策略，并确保 resume 语义明确。
4. 更新 image、radar、GPS、LiDAR、fusion 配置中的 split/cache/loader 默认值。
5. 补充测试并使用 `conda run -n kd_mm_beam pytest` 验证；必要时运行短训练 smoke test。

## Open Questions

- 无。
- LiDAR cache 目录是否按数据集根目录共享，还是按 BEV 参数 hash 分目录以避免 ROI/size 改动后的缓存复用风险。
