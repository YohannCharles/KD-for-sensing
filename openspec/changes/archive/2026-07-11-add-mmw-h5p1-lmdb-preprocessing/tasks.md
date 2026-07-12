## 1. LMDB 入口与配置

- [x] 1.1 让 sample LMDB 生成器通过 dataset registry 支持 DeepSense6G 和 MMW，并保留旧入口
- [x] 1.2 增加 MMW 三天气 H5/P1 图像、LiDAR 和 LMDB 预处理配置
- [x] 1.3 增加 registry 构建、MMW metadata 和旧入口兼容测试

## 2. 本地数据处理

- [x] 2.1 并行准备 rainy/foggy 十个场景并将 sunny 五个场景重建为 H5/P1
- [x] 2.2 并行生成三种天气的 image-derived、LiDAR BEV 和 split-level LMDB cache

## 3. 验证

- [x] 3.1 使用 `conda run -n kd_mm_beam pytest` 运行聚焦测试并运行 CLI/config 验证
- [x] 3.2 使用 `openspec validate add-mmw-h5p1-lmdb-preprocessing --strict` 校验 change
- [x] 3.3 核验全部场景 split metadata、缓存覆盖率、LMDB metadata 和剩余磁盘空间
