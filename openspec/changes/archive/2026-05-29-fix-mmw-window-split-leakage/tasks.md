## 1. Characterization 与泄漏诊断

- [x] 1.1 为当前 MMW sequence split 增加 focused characterization test，构造重叠滑窗 fixture，复现 random-window split 会让 train/test 共享相邻 frame 上下文。
- [x] 1.2 实现独立 leakage diagnostics helper 的测试用例，覆盖 frame overlap、最大窗口重叠、相邻窗口跨 split 和未来标签序列复用比例。
- [x] 1.3 使用 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q` 运行现有准备流程测试，记录当前失败或基线行为。

## 2. Group-safe Split 实现

- [x] 2.1 在 MMW preparation/split utility 中引入 split strategy 参数，默认并公开支持 `group_safe_time_block`。
- [x] 2.2 为 frame manifest 到 sequence rows 的流程补充 stable group metadata，包括 scenario、agent、contiguous segment id、frame range、window frame ids 和 future label sequence。
- [x] 2.3 实现 group-safe time-block 分配逻辑，按连续片段内 block 或等价 group 分配 train/test，并保留默认不少于 `seq_len + pred_len - 1` 的 guard band。
- [x] 2.4 移除旧 random-window 生成兼容路径，unsupported strategy 按普通不支持策略处理。
- [x] 2.5 更新 `scripts/mmw/build_sequence_splits_from_manifest.py` 和 MMW Town10 preparation 配置，使未指定 strategy 时默认使用 group-safe split，并允许 CLI/config 显式覆盖。

## 3. Metadata 与运行产物接入

- [x] 3.1 扩展 split metadata，写出 split strategy、protocol version、group key、block size、guard band、train/test group 列表、样本数、标签分布和 leakage diagnostics。
- [x] 3.2 更新训练 runtime metadata 或 run metadata 读取 split metadata，并在 `final_config.yaml`、`metrics.json`、`train_log.json` 或等价产物中记录核心字段。
- [x] 3.3 更新 standalone evaluate 报告，使其记录 test CSV、split metadata 路径、strict eligibility 和 unknown/ineligible split 警告。
- [x] 3.4 更新 MMW quick summary/conclusion 过滤逻辑，默认排除 `strict_validation_eligible=false` 或 unknown 的 MMW Town10 run，并保留 debug/sanity 指标查看能力。

## 4. 脚本与迁移边界

- [x] 4.1 更新 MMW sunny modal15 orchestration 或推荐命令，默认使用新的 strict split tag，避免复用旧 `l5p6` 随机滑窗 CSV。
- [x] 4.2 为已存在但缺少新 metadata 的旧 split 提供清晰警告或修复提示，不静默跳过 split 准备。
- [x] 4.3 更新相关 README、docs 或 inventory 中的 MMW split 说明，只记录协议边界和推荐命令，不纳入本地数据产物。

## 5. 测试与验证

- [x] 5.1 添加或更新 `tests/test_mmw_town10_preparation.py`，覆盖 group-safe split 无 frame overlap、guard band 生效、unsupported strategy 和 metadata 字段。
- [x] 5.2 添加或更新训练/评估 workflow focused tests，覆盖运行产物记录 split eligibility、strict-ineligible split 被 summary 排除、metadata 缺失时保守处理。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py tests/test_training_io_workflow.py -q`。
- [x] 5.4 运行 `openspec validate fix-mmw-window-split-leakage --strict`。
- [x] 5.5 运行 `openspec status --change fix-mmw-window-split-leakage`，确认实现任务状态和 OpenSpec artifact 状态一致。
