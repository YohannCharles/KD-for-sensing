## 1. Launcher 契约

- [x] 1.1 在 H5/P1 launcher 最终 overlay 中统一 Scene31-34 和 split contract
- [x] 1.2 扩展 dry-run 测试，断言 U-Mask、AMBER、RMBP-MM 生成配置的数据划分一致

## 2. 验证与重训

- [x] 2.1 使用 `conda run -n kd_mm_beam pytest tests/test_h5_p1_temporal_matrix_v1.py -q` 并严格校验 OpenSpec change
- [x] 2.2 生成并检查 AMBER、RMBP-MM seed1 Scene31-34 配置
- [x] 2.3 在独立 GPU 上启动两路重训并检查进程、日志和显存
