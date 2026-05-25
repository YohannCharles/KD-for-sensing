## 1. 基线与元数据

- [x] 1.1 用 `conda run -n kd_mm_beam python scripts/profile_training_io.py` 对 Multimodal-NF GPS-only、image-only、LiDAR-only 和 fusion 小样本配置建立基线，并记录模态级 `__getitem__`、DataLoader wait、transfer、forward/backward 和 samples/s。
- [x] 1.2 扩展 Multimodal-NF runtime metadata，使每个 split 和启用模态记录 `source_kind`、缓存策略、缓存路径、是否命中、是否生成和是否回退。
- [x] 1.3 扩展 `scripts/profile_training_io.py`，输出 Multimodal-NF 数据源、派生缓存策略、缓存状态和 train/test DataLoader 参数。

## 2. 派生缓存生成

- [x] 2.1 设计并实现 image/LiDAR 派生缓存 metadata sidecar，包含原始路径或 fingerprint、profile、shape、dtype、样本数、`seq_len`、`num_pred` 和生成时间。
- [x] 2.2 实现 Multimodal-NF image 派生缓存生成 helper，保证 fixture 上生成结果与原始 HDF5 读取路径 shape 和样本顺序等价。
- [x] 2.3 实现 Multimodal-NF LiDAR 派生缓存生成 helper，保持 `point_cloud_xyz_10000` 输入语义不变。
- [x] 2.4 为缓存生成使用原子写入或等价机制，避免多进程读取半成品缓存。
- [x] 2.5 增加或扩展预处理入口，使用户可显式预热 Multimodal-NF image/LiDAR 派生缓存。

## 3. Dataset 读取路径

- [x] 3.1 为 Multimodal-NF dataset 增加 image/LiDAR 派生缓存策略解析，支持 `off`、`auto`、`read_only` 和 `rebuild`。
- [x] 3.2 在 `_MultimodalNFAdapter` 或等价边界中加入派生缓存读取分支，并保持 sample keys、tensor shape、dtype 语义和 target 字段兼容。
- [x] 3.3 实现缓存 metadata 校验：原始 fingerprint、profile、split、`seq_len`、`num_pred` 不匹配时按策略失败、回退或重建。
- [x] 3.4 确保 GPS-only、CSI-only 和不包含 image/LiDAR 的 fusion 配置不解析、不创建、不读取 image/LiDAR 缓存路径。

## 4. 配置与推荐

- [x] 4.1 更新 Multimodal-NF image/LiDAR/fusion 示例配置或注释性配置，提供合理的 DataLoader worker、prefetch、pin memory、test worker、AMP 和 progress 建议。
- [x] 4.2 扩展 `scripts/recommend_parallel_training.py` 或等价推荐入口，为 Multimodal-NF image/LiDAR/fusion 输出缓存预热、AMP、DataLoader 和 progress 覆盖建议。
- [x] 4.3 确保推荐只输出命令行覆盖或说明，不直接修改用户配置文件。

## 5. 测试与验证

- [x] 5.1 增加 Multimodal-NF fixture 测试，验证原始 HDF5 与派生缓存路径的 image/LiDAR tensor shape、target fields、metadata 关键字段和样本顺序等价。
- [x] 5.2 增加测试覆盖 `read_only` 缓存缺失、metadata 不匹配、`auto` 回退或生成、`rebuild` 重建。
- [x] 5.3 增加测试确认未启用 image/LiDAR 时不访问对应缓存或原始大模态路径。
- [x] 5.4 增加 profile 输出字段测试，验证 Multimodal-NF 模态级 getitem、数据源、缓存策略、DataLoader split 参数和 samples/s 字段稳定。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_multimodal_nf_dataset.py -q`。
- [x] 5.6 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`。
- [x] 5.7 运行 `openspec validate optimize-multimodal-nf-throughput --strict`。
