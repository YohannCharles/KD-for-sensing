## 1. 现状审计与命名收敛

- [x] 1.1 审计现有 `CameraAEImageEncoder`、`ResNet18ImageEncoder`、`GpsFeatureExtractor`、`gps_strong/gps_lightweight` 和 `CLSTokenTransformerFusionNet` 的输入输出契约，确认哪些 baseline 可直接复用、哪些需要新增 wrapper。
- [x] 1.2 确定并记录 registry 名称与 preset 名称，至少包含 `camera_ae_gps`、`resnet_gps`、`transformer_image_gps` 和 `gps_only_neural`。
- [x] 1.3 确认 image profile、GPS feature mode、`num_pred`、64 类 label space 和 top-k metric 字段在四类 preset 中的默认值。

## 2. 模型与 registry 实现

- [x] 2.1 实现或整理 image+gps late-concat baseline 模型，使其支持 Camera AE encoder 与 ResNet encoder 可配置切换。
- [x] 2.2 实现 GPS-only neural baseline wrapper，明确与非神经 GPS window baseline 区分，并输出现有 engine 可识别的 logits/input_features/output_features。
- [x] 2.3 复用或扩展 CLS-token transformer fusion，使 image+gps Transformer preset 能记录 token 组织方式、`d_model`、heads、layers 和 max sequence length。
- [x] 2.4 将新增模型注册到 `MODELS`，并更新轻量导出或默认组件导入，保持包级导入不牵出重依赖。
- [x] 2.5 为 encoder checkpoint 缺失、image/GPS sequence 维度不一致、image profile 不兼容等情况添加清晰错误。

## 3. 配置与运行 metadata

- [x] 3.1 新增或扩展 canonical/virtual config recipe，提供四类 baseline preset 的可加载训练配置。
- [x] 3.2 确保 image+gps preset 只启用 image、gps、input beam 和 target beam 字段，不要求 radar、LiDAR、mmWave 或 CSI。
- [x] 3.3 在 run metadata 或 train log 中记录 `baseline_preset`、enabled modalities、encoder 类型、GPS feature mode、temporal aggregation、image profile、normalization artifact 和 mock/real data 标记。
- [x] 3.4 确保 Camera AE + GPS paper-style preset 在 `require_checkpoint=true` 且缺少 AE checkpoint 时清晰失败，并在 metadata 中记录是否使用官方权重/官方测试集/官方搜索流程。

## 4. 指标、训练评估闭环与产物边界

- [x] 4.1 确认四类 baseline 均可通过现有 supervised loss、optimizer、checkpoint 和 validation 流程训练。
- [x] 4.2 确认 `kd-sensing-evaluate` 能对四类 baseline 计算 top-1 和 top-3 accuracy，并保留 DBA/circular metric 的口径字段。
- [x] 4.3 确保 mock 或 synthetic smoke 运行产物包含 `mock_data: true`，真实训练输出、cache、checkpoint 和 TensorBoard 文件继续写入 ignored 本地产物路径。
- [x] 4.4 确保多 horizon 输出不会把历史 `input_beam` 拼入未来标签，且 top-k metrics 按预测 horizon 计算。

## 5. 测试覆盖

- [x] 5.1 添加模型 forward 单元测试，使用小型随机 image/GPS batch 覆盖 Camera AE + GPS、ResNet + GPS、Transformer image+gps 和 GPS-only neural logits shape。
- [x] 5.2 添加配置加载测试，覆盖四类 baseline preset 的 enabled modalities、primary model 类型、`num_pred` 和 64 类输出语义。
- [x] 5.3 添加数据字段选择测试，验证 image+gps preset 不读取未启用模态，GPS-only neural 不读取 image/radar/LiDAR/mmWave/CSI。
- [x] 5.4 添加错误路径测试，覆盖 AE checkpoint 缺失、image profile 不兼容、image/GPS sequence 维度不一致。
- [x] 5.5 添加 metrics/metadata 测试，验证 top-1/top-3 字段、metric profile、`baseline_preset` 和 `mock_data` 标记。

## 6. 文档与验证

- [x] 6.1 更新 README 或 `docs/experiment_matrix.md`，写明四类 baseline 的推荐命令、配置路径、mock/smoke 用法和产物位置。
- [x] 6.2 运行 `openspec validate add-vision-position-baselines --strict` 并修复所有 OpenSpec 问题。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [x] 6.4 运行新增 baseline 相关 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_vision_position_baselines.py -q`。
- [x] 6.5 如改动触碰训练、评估或数据字段公共路径，运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。
