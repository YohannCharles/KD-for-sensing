## 1. 现状审计与实现边界确认

- [x] 1.1 审计现有 DeepSense6G dataset、batch preparation、model output、focal loss、DBA metric 和 fusion config recipe，确认 `bev_fusion_2604` 可复用的输入键、label 形状和输出契约。
- [x] 1.2 审计 scenes 32/33/34 现有 sequence CSV 字段，确认 image、radar、gps、bs_gps、lidar、future_beam1 是否满足 2604 full 配置；缺失字段必须记录为 blocked reason。
- [x] 1.3 确定 GPS spatial pathway 的坐标来源和 ROI/grid bounds 策略，明确是否新增 `gps_bev_xy` batch 字段或等价 coordinate provider。
- [x] 1.4 确定 report 中使用的 linear DBA、Top-K、macro DBA 和 weighted overall 字段命名，避免与 circular DBA 混用。

## 2. GPS BEV 坐标与 batch 数据契约

- [x] 2.1 在 DeepSense6G dataset 或 transform helper 中新增可选 GPS relative XY/BEV coordinate 读取路径，确保坐标未被训练集 StandardScaler 改写。
- [x] 2.2 将 GPS BEV 坐标接入 batch preparation，使 BEV 模型可接收 `gps_bev_xy_batch` 或清晰命名的等价输入。
- [x] 2.3 为 GPS spatial 坐标缺失、维度错误、ROI 越界裁剪或 mask 生成添加清晰错误与 diagnostics。
- [x] 2.4 添加不依赖真实数据的 GPS BEV 坐标单元测试，并使用 `conda run -n kd_mm_beam pytest <相关测试> -q` 验证。

## 3. BEV-Fusion 2604 模型实现

- [x] 3.1 新增 `src/kd_sensing/models/bev_fusion_2604.py`，实现 `BEVFusion2604Net` 并注册 `MODELS.register("bev_fusion_2604")`。
- [x] 3.2 实现 paper/smoke 可配置 camera 2D backbone 和 learned camera-to-BEV cross-attention，输出 `[B,T,D,H_bev,W_bev]`。
- [x] 3.3 实现 LiDAR BEV projection、Radar RA/DA projection 和 GPS spatial mask projection，并统一对齐到模型 `bev_size`。
- [x] 3.4 实现 GPS dual-path global embedding 与 gated residual 注入，支持 `dual_path`、`spatial_only`、`global_only` ablation。
- [x] 3.5 实现 BEV spatial fusion block、temporal transformer、single-frame/mean-pool/1D fusion ablation core 和 `[B,1,64]` beam classifier。
- [x] 3.6 确保模型 forward 返回 engine 可适配的 dict，包含 `logits`、`input_features`、`output_features`、`bev_features`、effective modalities 和关键 diagnostics。
- [x] 3.7 更新模型 lazy exports 和默认组件注册，保持 `kd_sensing.config`、路径工具和模态契约轻量导入边界。

## 4. 配置族与 ablation matrix

- [x] 4.1 新增 `configs/fusion/experiments/bev_fusion_2604/` 配置族，至少包含 paper full、low-memory 和 smoke 配置。
- [x] 4.2 在 paper full 配置中设置 5 帧历史、`num_pred=1`、64 beam、scenes 32/33/34、四模态、128x128 BEV、`d_model=256`、temporal transformer 4 层 4 heads。
- [x] 4.3 配置 focal loss `gamma=2`、AdamW `lr=1e-4`、`weight_decay=1e-2`，并记录 class-weight/alpha 的 train-only fit 策略。
- [x] 4.4 新增 ablation 配置或 recipe，覆盖 without camera/LiDAR/radar/GPS、1D fusion、single-frame、mean-pooling temporal、GPS spatial-only 和 GPS global-only。
- [x] 4.5 为配置加载、enabled modalities、BEV size、loss、optimizer、metric profile 和 ablation metadata 添加测试，并使用 `conda run -n kd_mm_beam pytest <相关测试> -q` 验证。

## 5. 训练评估报告与安全增强

- [x] 5.1 实现或扩展 2604 report helper，汇总 S32/S33/S34 DBA、Top-K、macro DBA、weighted overall、论文目标值、差距、split/seed/sample count 和 metric profile。
- [x] 5.2 在 training/evaluation runtime metadata 中写入 `paper_exact_split_available`、`mock_data`、`paper_approximation`、model size、BEV shape、GPS pathway 和 ablation 名称。
- [x] 5.3 若实现 horizontal flip augmentation，则同时实现 beam index reversal 与单元测试；未完成前 paper full 配置必须禁用或拒绝该增强。
- [x] 5.4 增加可选参数量和本机 latency 统计，报告必须记录硬件；无 H100 时不得声称复现论文 H100 latency。
- [x] 5.5 确认所有 report、metrics、checkpoint、cache 和 TensorBoard 输出写入 ignored 的 `outputs/`、`logs/` 或等价本地产物目录。

## 6. 测试、文档与 OpenSpec 验证

- [x] 6.1 添加 synthetic 四模态 forward smoke test，验证 `bev_fusion_2604` logits shape、diagnostics、缺失模态错误和 GPS pathway ablation。
- [x] 6.2 添加 light import/architecture boundary 测试，确认新增模型不破坏配置、路径和模态契约轻量导入。
- [x] 6.3 更新 `docs/experiment_matrix.md` 或新增 2604 BEV-Fusion 实验说明，写明数据/cache 准备、推荐命令、ablation 命令、报告位置和可比性 caveat。
- [x] 6.4 运行 `openspec validate reproduce-bev-fusion-2604 --strict` 和 `openspec status --change reproduce-bev-fusion-2604`，修复所有 OpenSpec 问题。
- [x] 6.5 运行 focused tests：`conda run -n kd_mm_beam pytest <新增和相关测试> -q`。
- [x] 6.6 涉及架构、CLI 或公共 workflow 后，运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_config_load_characterization.py tests/test_cli_help.py -q`。
- [x] 6.7 如模型、数据或训练公共路径改动较大，最终运行 `conda run -n kd_mm_beam pytest -q`。
- [x] 6.8 最终检查 `git status --short`，确认未纳入真实数据、训练输出、日志、cache、新生成 checkpoint 或临时验证产物。
