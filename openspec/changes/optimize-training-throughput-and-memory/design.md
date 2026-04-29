## Context

项目已经完成按模态懒加载、LiDAR 懒加载初始化、LiDAR 参数 hash cache、统一 split、唯一输出目录和最佳 checkpoint registry。当前瓶颈集中在高重复滑动窗口下的 CPU/I/O 数据路径：image motion mask 每次取样重新计算，LiDAR cache 没有可靠预热入口，beam label 每次读取文本并 `argmax`，同时并行训练时 DataLoader worker 和 prefetch 会成倍增加。

本 change 的目标是先让吞吐瓶颈可测，再缓存高重复且参数稳定的预处理结果，最后打开 GPU transfer/AMP 优化。所有 Python 验证命令使用 `conda run -n kd_mm_beam ...`。

## Goals / Non-Goals

**Goals:**

- 为 dataset、DataLoader、GPU transfer 和 training step 提供轻量 benchmark/profile 脚本。
- 实现 image motion mask 长期可复用 cache，并用参数 hash 与 metadata 避免误用旧缓存。
- 让 LiDAR BEV cache 可通过预处理入口一次性预热 train/test split，并让训练配置直接复用。
- 缓存 beam 文本到整数 label 的映射，消除重复 `np.loadtxt`。
- 让 batch transfer 支持 `non_blocking=True`，训练支持可配置 AMP。
- 调整并行实验配置建议，降低默认 worker/prefetch 放大效应。

**Non-Goals:**

- 不改变模型结构、KD loss、指标定义或 checkpoint 格式。
- 不把所有预处理 cache 强制常驻内存；磁盘 cache 仍是默认策略。
- 不重写 image/radar/lidar/fusion backbone 重复代码；模型抽象重构排在吞吐优化之后。
- 不改变 image motion mask 的默认算法输出，除非显式修改缓存版本或预处理参数。

## Decisions

1. image motion mask cache 按“相邻帧 pair”优先设计。

   当前 `load_motion_masks` 对长度为 8 的窗口输出 7 张相邻帧差分 mask。相邻窗口高度重叠，因此 pair-level cache 比 sequence-level cache 更能复用。cache key 使用两个相对 image 路径、image size、灰度化方式、Gaussian sigma、阈值策略、实现版本和原始文件 metadata 生成；默认保存为 `.npy` `uint8`，dataset 读取后转换为 `torch.float32`，保持返回契约不变。若后续需要极致顺序读性能，可追加 sequence-level pack，但首版不必引入更复杂格式。

2. 所有预处理 cache 使用参数 hash 子目录和 `metadata.json`。

   LiDAR 已有 `parameterized_lidar_cache_dir`，本 change 将同样原则推广到 image motion mask。metadata 记录 cache version、关键参数、源 CSV、data root、生成时间、生成数量、跳过数量和可选的源文件 mtime/size。训练参数如 lr、epochs、batch size、num_workers、模型结构和 KD 类型不进入 cache key；原始文件内容或预处理参数变化必须进入新目录或触发 cache miss。

3. Dataset 读取 cache 时仍保持懒加载。

   image motion mask、LiDAR BEV 和 beam label 都不得在 Dataset 初始化阶段全量 materialize 成大 tensor。beam label 是例外中的轻量路径：Dataset 可以在初始化阶段扫描当前 split 中唯一 beam path 并建立 `dict[path, int]`，因为它只保存整数且数量约数千；也可以按需 lazy fill。image/LiDAR 大数组只在当前样本取用时从磁盘读取。

4. LiDAR 预热入口复用现有 BEV 构造逻辑。

   `generate_lidar_bev_cache` 继续调用 `build_lidar_bev` 和 `lidar_cache_path`，但扩展为接受多个 CSV 或一个 train/test 配置列表，跳过已存在 cache，支持 `overwrite`，显示 tqdm 进度，并写出 metadata。由于训练读取路径已经使用同一参数 hash，预热和训练必须共享同一 cache 目录解析函数。

5. profiling 先面向定位，不引入重依赖。

   新增脚本应能在小比例 CSV 上运行，输出 JSON/CSV 摘要：每模态 `__getitem__` 均值/P50/P95、DataLoader batch wait、transfer time、forward/backward/optimizer time、GPU memory 和 samples/s。GPU 计时使用 `torch.cuda.Event`，CPU 计时使用 `time.perf_counter`。脚本不依赖 TensorBoard 或新的第三方 profiler。

6. DataLoader 默认区分“单实验吞吐”和“并行实验稳定”。

   默认配置从 `num_workers: 4`、`prefetch_factor: 2` 保持保守或下调；大量 YAML 中显式 `num_workers: 8` 的 canonical 配置应改为更稳的 2 到 4，并支持命令行覆盖。文档和任务中要求提供 profiling 建议：并行跑 4 个实验时先测 `num_workers=2~4,prefetch_factor=1`，单实验再逐步调高。

7. AMP 以配置开关接入训练循环。

   新增 `training.amp.enabled`、`dtype` 和 `grad_scaler` 相关配置。CUDA 可用且 AMP 启用时，用 `torch.autocast(device_type="cuda", dtype=torch.float16或bfloat16)` 包裹 forward/loss，用 `GradScaler` 执行 backward/step；CPU 或禁用 AMP 时完全回退现有 FP32 路径。验证路径可先支持 autocast，但不改变指标计算 dtype 语义。

## Cache Reuse Rules

可长期复用的缓存包括 LiDAR BEV、image motion mask、radar 预生成 map 和 beam label 映射，前提是原始文件内容和对应预处理参数不变。训练参数、模型结构、KD 类型、optimizer、scheduler、batch size、epochs、seed 和输出目录不应使这些缓存失效。

必须新建 cache 或触发 miss 的情况包括：原始 jpg/LiDAR/radar/GPS/beam 文件内容变化；LiDAR BEV size、ROI、FoV、ground/background filter 参数变化；image size、Gaussian sigma、阈值策略、灰度化方式或 cache version 变化；radar FFT/RA/DA 生成参数变化；GPS 坐标转换或 feature mode 变化。归一化统计不同于原始模态 cache，仍需要和 train split、portion、feature mode、预处理版本绑定。

## Risks / Trade-offs

- [Risk] image mask cache 占用磁盘空间增加 → 使用 `uint8`/bool 存储，metadata 记录数量和总大小，保留清理说明。
- [Risk] 原始文件被覆盖但路径不变导致误用 cache → 首版 metadata 至少记录 mtime/size；若需要更强保证，后续可启用内容 hash。
- [Risk] AMP 引入数值差异 → 默认可关闭，并用 smoke test 确认 loss backward、checkpoint 和指标路径可运行；精度对比留给实验。
- [Risk] 下调 DataLoader worker 可能降低单实验峰值吞吐 → 提供 profile 脚本和配置覆盖，默认优先服务并行实验稳定性。
- [Risk] beam label 初始化扫描会增加 dataset 构建时间 → 只扫描当前 split 唯一路径并保存 int dict，数量远小于样本引用次数，且可配置为 lazy。

## Migration Plan

- 先增加 profile 脚本和 baseline 记录，确认优化前数据路径瓶颈。
- 实现 image motion mask cache 和 beam label cache，并让 Dataset 支持 cache hit/miss/write。
- 扩展 LiDAR BEV cache 预处理入口，预热 train/test cache。
- 接入 non-blocking transfer 和 AMP，调整 DataLoader 默认和 YAML。
- 补充测试、README/配置说明和 smoke benchmark，最后运行 `conda run -n kd_mm_beam pytest -q -p no:cacheprovider` 与 OpenSpec 校验。

## Open Questions

- image motion mask 是否需要同时提供 sequence-level pack 以减少大量小 `.npy` 文件；首版采用 pair-level cache，除非实际 profile 显示文件数量成为新瓶颈。
- 是否要为原始文件内容启用强 hash；首版可先用 mtime/size，强 hash 作为可选慢速校验。
