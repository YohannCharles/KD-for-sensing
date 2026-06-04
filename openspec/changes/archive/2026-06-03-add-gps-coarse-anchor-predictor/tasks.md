## 1. Anchor 契约与模块骨架

- [x] 1.1 新增 GPS coarse anchor 数据结构或 TypedDict，覆盖 `coarse_logits`、`center_beam`、`beam_scores`、`confidence`、`residual_anchor_beam` 和 metadata。
- [x] 1.2 新增 anchor 配置解析，支持 `anchor_source`、`num_classes`、`group_size`、boresight/calibration、confidence、loss weights 和 artifact 开关。
- [x] 1.3 复用或包装 `gps_window` 几何工具，统一 angle-to-beam、boresight、beam direction、beam offset、circular distance 和 score kernel 语义。
- [x] 1.4 为 DeepSense6G/MMW batch 或 prediction rows 增加 GPS anchor adapter，只读取 GPS/pose/GPS-Rel-Polar 与合法 split metadata。

## 2. BeamBench-style 几何 Anchor

- [x] 2.1 实现 `geometry_calibrated` anchor builder，从 GPS/pose 或 GPS-Rel-Polar 生成 calibrated azimuth、center beam 和 coarse group。
- [x] 2.2 实现 coarse logits 与可选 beam score 生成，支持 `group_size`、neighbor expansion、score width 和 confidence。
- [x] 2.3 实现 source 或 target_adapt support 的 boresight/direction/offset 校准，并记录 fit/selection split、样本数和 effective 参数。
- [x] 2.4 加入 target_test oracle guard，禁止用 target_test label、beam_power argmax、path/radio/channel oracle 或其它模态生成 anchor。
- [x] 2.5 输出 geometry anchor diagnostics，包括 GPS coverage、fallback status、center beam circular error 和 residual preview 所需字段。

## 3. GPS Neural Coarse Head

- [x] 3.1 新增 GPS coarse head 模块或 wrapper，接收 GPS encoder/temporal representation 并输出 `[B, H, G]` coarse logits。
- [x] 3.2 在配置解析中显式支持 `coarse_anchor.enabled=true`，未启用时保持 `gps_teacher`/`gps_student` 默认 forward 契约不变。
- [x] 3.3 实现 coarse label 生成和 coarse CE loss，校验 `num_classes % group_size == 0`。
- [x] 3.4 支持可选 beam auxiliary logits/loss 与 anchor confidence calibration，并记录 loss diagnostics。
- [x] 3.5 添加 GPS neural anchor smoke 配置，覆盖 source-only train/eval 与 held-out target eval。

## 4. 跨场景评估与产物

- [x] 4.1 新增 GPS anchor evaluation runner 或复用现有评估入口的 opt-in profile，输出 anchor metrics 和 prediction artifact。
- [x] 4.2 实现 anchor metrics：coarse accuracy、center beam Top-1/Top-3、circular beam error、confidence summary、DBA/beam power 可用性记录。
- [x] 4.3 实现 seen/unseen scene summary，记录 source scenes、target scene、calibration split、evaluation split 和 split protocol。
- [x] 4.4 实现 residual preview：计算 anchor residual histogram、entropy、top-k 邻域覆盖率，并标记其只用于 evaluation diagnostics。
- [x] 4.5 新增 DeepSense6G Scenes 31-34 或 MMW LOSO anchor 配置，默认输出到 `outputs/gps_coarse_anchor/` 且不提交运行产物。

## 5. HiST-Beam 显式接入

- [x] 5.1 在 HiST-Beam batch preparation 中支持 `hist_beam.gps_anchor.enabled=true` 时提供 GPS anchor 字段。
- [x] 5.2 在 `HistBeamFusionNet` 或窄 wrapper 中添加 anchor-conditioned 输入路径，可将 coarse distribution、center beam embedding 和 confidence 拼入 coarse/fine/residual 分支。
- [x] 5.3 未启用 GPS anchor 时保持现有 HiST-Beam forward、loss 和 quick validation 行为不变。
- [x] 5.4 启用 GPS anchor 时，prediction artifact 写出 anchor center beam、anchor coarse top-k、anchor confidence 和 model 相对 anchor 的 residual 改善。
- [x] 5.5 缺失 anchor 必需字段时抛出清晰错误，不静默回退到普通 HiST-Beam 输入语义。

## 6. 测试与验证

- [x] 6.1 添加 anchor 契约单元测试，验证输出 shape、`group_size` 校验、metadata 字段和默认关闭行为。
- [x] 6.2 添加几何 anchor 测试，覆盖 boresight 中心化、direction/offset、coarse group、beam score 和 confidence。
- [x] 6.3 添加防泄漏测试，验证 target_test calibration、future beam、beam_power argmax、path/radio/channel oracle 违规会失败或标记不合格。
- [x] 6.4 添加 GPS neural coarse head 测试，覆盖 forward、coarse loss、beam auxiliary loss 和现有 GPS 模型兼容性。
- [x] 6.5 添加 HiST-Beam anchor-conditioned 测试，覆盖启用/关闭 forward、缺失字段错误和 prediction artifact 字段。
- [x] 6.6 运行相关测试：`conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_hist_beam_loso.py -q`，并根据新增测试文件补充命令。
- [x] 6.7 运行架构/CLI 快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_cli_help.py -q`。
- [x] 6.8 运行 OpenSpec 校验：`openspec validate add-gps-coarse-anchor-predictor --strict`。
