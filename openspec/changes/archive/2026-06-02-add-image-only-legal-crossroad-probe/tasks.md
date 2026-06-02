## 1. 配置与运行入口

- [x] 1.1 新增 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`，声明 target/source scenes、`modalities: ["image"]`、disabled/excluded sensitive fields、legal protocol flags、feature cache、I0-I3 probe mode 默认参数。
- [x] 1.2 新增 `scripts/run_image_only_legal_crossroad_probe.sh`，按 I0/I1/I2/I3 运行 source_train、target eval、target adaptation、summary 和 eligibility check；脚本内 Python 命令必须使用 `conda run -n kd_mm_beam`。
- [x] 1.3 确保每个 run 输出到 `outputs/image_only_legal_seed0/<mode>`，并保存 resolved config、source/target scene、seed、label budget、enabled/disabled modalities metadata。

## 2. Image-only 数据与 batch 契约

- [x] 2.1 梳理现有 MMW/HiST-Beam dataset、collate 和 batch preparation 中的模态解析路径，确认 canonical beam label key 与 source/target split key。
- [x] 2.2 在 image-only protocol 下实现 batch allowlist，只向模型、loss、adaptation 和 evaluator 暴露 image、beam label、scene、sample_id、split 等合法字段。
- [x] 2.3 确保 collate 不要求 GPS、LiDAR、radar、mmWave、CSI、channel、path 或 beam_power key，且原始字段存在时只记录为 available fields，不进入 consumed fields。
- [x] 2.4 增加或更新 dataloader smoke test，使用 `conda run -n kd_mm_beam pytest ...` 验证 image-only one batch keys 与禁用字段隔离。

## 3. HiST-Beam image-only 模型路径

- [x] 3.1 在 HiST-Beam 模型构建或注册逻辑中接入 `image_only_v8_v9_probe` 或等价配置路径，复用 image encoder/projection，默认 `fusion_mode: identity`。
- [x] 3.2 确保 image-only forward 不访问禁用模态，输出 `logits`、`logits_final`、`features`、可选 `source_logits` 和可选 `target_logits`。
- [x] 3.3 让 evaluator 兼容 source-only 模式下 `target_logits` 为空或缺失的输出 dict。
- [x] 3.4 增加 source forward smoke test，使用 `conda run -n kd_mm_beam pytest ...` 验证 image-only output keys、shape、device 和 dtype。

## 4. I0-I3 适配模式

- [x] 4.1 实现 I0 `image_source_only`：source image-only training、source-only target_test evaluation、不执行 target adaptation，并输出标准指标和 collapse diagnostics。
- [x] 4.2 实现 I1 `image_target_linear_probe`：冻结 image backbone/projection/fusion/source head，只训练 `target_linear_head`，并记录 `[image-only A2] trainable parameter names` 与 trainable ratio。
- [x] 4.3 实现 I2 `image_v8_target_prior_head`：target prior 只由 target support beam labels 加 Gaussian smoothing 初始化，final logits 不混入 source logits，beta 使用 cap 或日志记录固定值。
- [x] 4.4 实现 I3 `image_v9_sector_proto`：按 target support feature 的 `beam // sector_size` 构建 sector prototype，映射回 beam logits，禁用 beam-level prototype，并输出 sector/proto 前后 top beam 日志。
- [x] 4.5 增加 target adaptation forward 与 loss backward smoke test，使用 `conda run -n kd_mm_beam pytest ...` 覆盖 I1/I2/I3 的合法 support label 使用和冻结参数集合。

## 5. Feature cache 与诊断产物

- [x] 5.1 新增 image feature cache 写入/读取逻辑，按 source_train、target_support、target_test split 保存 feature、label、scene、sample_id、split 和 `cache_meta.json`。
- [x] 5.2 实现 cache metadata 校验，覆盖 checkpoint、feature_dim、modalities、image_encoder、source scenes、target scene、label_budget 和 dtype；不匹配时拒绝复用或要求 overwrite。
- [x] 5.3 确保 target adaptation 只读取 target_support cache labels，target_test cache labels 只能在 evaluation scope 用于最终 metrics。
- [x] 5.4 新增 `prediction_hist.json` 和 `confusion_by_true_beam.json` 写出逻辑，覆盖 true/pred hist、top beams、unique predicted beams、top beam ratio、MAE 和 within-k。
- [x] 5.5 新增 `combined_summary.csv` 汇总逻辑，包含 mode、Top1/Top3/Top5、Within1/2/3、MAE、BPL dB、NRP、prediction collapse 指标、eligible、eligibility reasons 和 trainable ratio。

## 6. Eligibility checker 修复

- [x] 6.1 定位当前产生 `target_oracle_fields_used`、`target_radio_label_supervision`、`target_path_label_supervision` 和 `split_eligibility_unknown` 的 checker 与 summary 聚合逻辑。
- [x] 6.2 将 eligibility 判断改为基于 stage-level consumed fields、protocol flags 和 split diagnostics，而不是基于 raw dataset/manifest 中字段是否存在。
- [x] 6.3 对 image-only legal run 记录 `used_target_oracle_fields=[]`、enabled/disabled modalities、excluded sensitive fields、available fields 与 consumed fields。
- [x] 6.4 当 split eligibility 无法判断时，输出缺失 metadata、config path 或 diagnostics path；不得静默设置 unknown 后缺少定位信息。
- [x] 6.5 增加 eligibility 单元或集成测试，使用 `conda run -n kd_mm_beam pytest ...` 验证合法 image-only run 不被 oracle 字段误排除，实际消费禁用 oracle 时会被排除。

## 7. 验证与 OpenSpec 校验

- [x] 7.1 运行 image-only 最小 smoke test：dataloader one batch、source forward、target adaptation forward、loss backward、eval metrics、eligibility check，命令必须使用 `conda run -n kd_mm_beam`。
- [x] 7.2 运行相关快速测试，例如 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 和新增/更新的 image-only、eligibility 测试。
- [x] 7.3 运行 OpenSpec 校验：`openspec validate add-image-only-legal-crossroad-probe --strict`。
- [x] 7.4 运行 `openspec status --change add-image-only-legal-crossroad-probe`，确认 artifacts complete 且 ready for apply。
- [x] 7.5 如环境和数据可用，执行 `bash scripts/run_image_only_legal_crossroad_probe.sh`，并汇总 I0/I1/I2/I3 的指标、prediction top beams、eligible 状态和 `combined_summary.csv` 路径。
