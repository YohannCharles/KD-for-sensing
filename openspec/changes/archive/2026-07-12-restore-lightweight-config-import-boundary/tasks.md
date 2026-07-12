## 1. 纯时序配置契约

- [x] 1.1 新增 stdlib-only `kd_sensing.data.temporal_missing_contract`，集中承载两个 mode 常量和两个字符串 normalizer，且不导入 torch、config 或 runtime owner。
- [x] 1.2 在 fresh subprocess 中增加配置轻量导入 characterization，验证 `import kd_sensing.config` 后不加载 torch、模型、dataset runtime、诊断渲染或训练主循环，且不使用时间/RSS 硬阈值。

## 2. Config 与 Runtime 迁移

- [x] 2.1 将 `kd_sensing.config.normalization` 改为直接依赖纯 contract，并确认合法/非法时序模式与聚合值继续使用原规范化和错误语义。
- [x] 2.2 将 `kd_sensing.data.temporal_missing` 改为复用并继续暴露 contract 符号，保持现有 `__all__`、difficulty operator、tensor 聚合、mask 采样、batch 变换和固定 mask cache 行为。
- [x] 2.3 补充或调整最小 focused tests，覆盖 contract 单一来源、原 runtime import 路径兼容和 config normalization 不触发 tensor runtime。
- [x] 2.4 将 configuration validation 使用的 difficulty preset helper 改为按需导入，避免 fresh config import 经 missing-pattern runtime 加载 torch，并保持 preset/operator 错误语义。

## 3. 验证与收尾

- [x] 3.1 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_temporal_window_missing.py tests/test_modality_difficulty.py -q`；若当前 H5P1 launcher 的独立在途 safety guard 仍失败，记录基线且不修改该在途文件。
- [x] 3.2 运行 `conda run -n kd_mm_beam python -c "import sys; import kd_sensing.config; assert 'torch' not in sys.modules"` 和 `make verify-cli-config`，确认轻量导入与公共 CLI/config characterization 通过。
- [x] 3.3 运行 `openspec validate restore-lightweight-config-import-boundary --strict`、`openspec validate --all --strict` 和 `make verify-quick`，确认 change、current specs 与架构护栏状态；任何无关在途失败必须单独报告，不得通过放宽本 change 的轻量导入断言掩盖。
