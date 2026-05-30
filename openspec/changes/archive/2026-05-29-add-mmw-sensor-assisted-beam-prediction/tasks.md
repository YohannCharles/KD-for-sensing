## 1. 配置与数据路径预检

- [x] 1.1 新增 MMW sensor-assisted HiST-Beam 配置，默认 `model.modalities=[image,gps,lidar,radar]`、`data.dataset.enabled_modalities=[image,gps,lidar,radar]`，并明确不启用 `mmwave` 输入。
- [x] 1.2 新增或更新 smoke 配置，使 `loso.budgets=[10]`、`loso.seeds=[0,1]`，并标记 `profile=sensor_assisted_quick_validation` 或等价 metadata。
- [x] 1.3 使用 `conda run -n kd_mm_beam <python smoke>` 验证三个 ready MMW scenario 的 sample 能返回 image、gps、lidar、radar_ra、radar_da 和 target beam。
- [x] 1.4 验证 radar derived CSV 自动生成或已存在，缺失 radar map 时给出 actionable error，而不是训练中途失败。
- [x] 1.5 验证 LiDAR/radar cache、num_workers 和 CPU thread 配置能写入 run/preflight metadata。

## 2. 输入边界与防泄漏

- [x] 2.1 调整配置解析或 dataset factory，确保 sensor-assisted profile 的 enabled modalities 不包含 `mmwave`、`csi`、`channel`、`path` 或 `beam_power`。
- [x] 2.2 增加 focused test，使用 `conda run -n kd_mm_beam pytest <sensor-assisted config/dataset tests> -q` 验证 model forward kwargs 不包含 `mmwave_batch`。
- [x] 2.3 确认 V6/V8 在 sensor-assisted profile 中只能把 beam_power/path/radio 字段用于 source auxiliary、prototype 或 offline diagnostics，不作为 sensing input。
- [x] 2.4 扩展或复用 leakage flags，确保 summary 记录 `used_target_beam_power_for_training`、`used_target_path_label_for_training`、`used_target_radio_label_for_training` 等字段。

## 3. HiST-Beam 模型与运行计划

- [x] 3.1 确认 `hist_beam_fusion` 能使用 `image+gps+lidar+radar` 构建 V3/V4/V5/V6/V8/full fine-tuning 变体。
- [x] 3.2 若 radar 或 lidar encoder 对输入 shape 有假设，补充 shape 校验和错误信息。
- [x] 3.3 更新 LOSO planner/runner，使 sensor-assisted quick validation 默认只产生 `budget=10` 和两个 seeds 的 run plan。
- [x] 3.4 run plan 必须记录 modality profile、budgets、seeds、target/source scenes 和 quick validation scope。
- [x] 3.5 若用户通过 CLI override 扩大 budgets 或 seeds，run plan metadata 必须记录 override 后的矩阵，并保留 quick/full matrix 区分。

## 4. Summary 与诊断指标

- [x] 4.1 在 LOSO summary 中新增或填充 adapted-source Top-1/Top-3/Top-5 delta、power delta 和 negative-transfer flag。
- [x] 4.2 在 summary 中聚合 last-beam diagnostic baseline Top-1/Top-3，并标记它是否为可比较 baseline。
- [x] 4.3 在 summary 中记录 trainable ratio、adaptation time、enabled modalities、cache policy 和 sensitive-field usage flags。
- [x] 4.4 增加 V8 vs V6、V8 vs path-condition-off、V8 vs source-only、V8 vs full fine-tuning 的 budget10/2-seed 对比输出。
- [x] 4.5 增加 focused test，使用 `conda run -n kd_mm_beam pytest <sensor-assisted summary tests> -q` 验证 delta 和 last-beam 字段存在且语义正确。

## 5. 快速实验

- [x] 5.1 运行单 run smoke：使用 `conda run -n kd_mm_beam kd-sensing-hist-beam-loso ... --execute --max-runs 1` 验证完整 source train、adaptation、evaluation 和 summary 流程。
- [x] 5.2 运行 sensor-assisted quick matrix：预算只用 `10`，seeds 只用两种，变体至少覆盖 source-only、adapter-only、V6 radio、V8 path、path condition off 和 full fine-tuning。
- [x] 5.3 记录 quick matrix 的 run_count、completed count、失败原因和总耗时；若某个 LiDAR/radar run 失败，保留失败 metadata 并优先修复数据路径。
- [x] 5.4 分析是否存在 few-shot negative transfer；若 adapted-source delta 为负，记录目标场景、变体和可疑原因。
- [x] 5.5 明确报告 sensor-assisted 结果不替代完整 budgets/seeds sweep，也不声称 town/weather 泛化。

## 6. 回归与 OpenSpec

- [x] 6.1 运行相关 focused tests：`conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_mmw_town10_preparation.py tests/test_architecture_boundaries.py -q` 或更窄等价测试。
- [x] 6.2 运行 CLI 快速检查：`conda run -n kd_mm_beam kd-sensing-hist-beam-loso --help`。
- [x] 6.3 运行 OpenSpec 校验：`openspec validate add-mmw-sensor-assisted-beam-prediction --strict`。
- [x] 6.4 更新 `tasks.md` 的实验结论，写实记录 budget10/2-seed 结果、负迁移情况、last-beam baseline 和后续是否值得扩大矩阵。

## 实验结论

- 单 run smoke：`outputs/validation_logs/mmw_sensor_assisted_smoke_execute` 完成 `v3_decoupled/budget10/seed0` 的 source train、source-only target eval、target adaptation、adapted eval 和 summary；preflight 记录 enabled modalities 为 `image/radar/gps/lidar`，excluded sensitive fields 为 `mmwave/csi/channel/path/beam_power`。
- quick matrix：使用 `configs/hist_beam/mmw_sensor_assisted_smoke.yaml` 的 smoke 规模设置并行运行 `outputs/validation_logs/mmw_sensor_assisted_parallel36`，覆盖 3 个 ready scenario target、`budget=10`、seeds `[0,1]` 和 6 个变体（source-only、adapter-only、V6 radio、V8 path、path-condition-off、full fine-tuning）。总计 36/36 runs completed，failed=0，missing=0；并行墙钟约 360 秒。
- 结果：该 smoke 规模矩阵中各变体 final Top-1 均为 0.0，adapted-source Top-1 delta 为 0.0 或不适用；未观察到 Top-1 negative transfer。V6 radio 的 normalized received power delta 均值约 -0.0011，beam power loss dB delta 均值约 +0.0346 dB，属于小幅负向功率变化，需更大矩阵确认。
- last-beam diagnostic baseline 仍很强：不同 target/seed 的 last-beam Top-1 约为 0.818-1.0，Top-3 为 1.0，并在 summary 中标记为 diagnostic、不可比较主 baseline。
- 结论边界：本结果只证明 sensor-assisted `image+gps+lidar+radar` 数据路径、模型 forward、LOSO runner、summary/diagnostic 和防泄漏 metadata 可用；不替代完整 budgets/seeds sweep，也不声称 leave-one-town-out 或 weather-shift 泛化。已将后续 sensor-assisted quick/smoke 配置统一为 `seq_len=5`、`num_pred=3` 以削弱 last-beam shortcut；既有 `mmw_sensor_assisted_parallel36` 结果仍是修改前历史窗口/horizon，重新运行后需单独记录。
