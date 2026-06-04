## 1. 现状巡检与模块骨架

- [x] 1.1 巡检本地 MMW manifest/dataset batch 中 GPS、CAV pose、RSU pose、timestamp、beam label、beam power 字段，记录可用字段和缺失路径。
- [x] 1.2 确定 GPS window baseline 模块边界，新增或选择 `src/kd_sensing/baselines/` 下的实现目录，并保持不接入神经网络训练器。
- [x] 1.3 定义 `GpsWindowBaselineConfig`、`GpsWindowSample`、`GpsWindowPrediction` 和 run metadata 数据结构。
- [x] 1.4 复用或包装现有几何工具，统一 angle-to-beam、circular beam distance、top-k neighbor 和 num_classes 语义。

## 2. GPS 滑动窗口预测核心

- [x] 2.1 实现 `geometry_last` predictor：使用最后历史 GPS/pose 与 RSU pose 生成 beam score。
- [x] 2.2 实现 `constant_velocity` predictor：从滑动窗口估计速度，按 horizon 外推位置并生成 beam score。
- [x] 2.3 实现环形角度平滑和角速度外推，避免 0/360 度边界被普通线性平均破坏。
- [x] 2.4 实现 beam score kernel：支持中心 beam、环形邻域扩展、temperature/width 参数和 `[N, H, C]` 输出。
- [x] 2.5 实现 fallback 机制：GPS 缺失、历史窗口不足或几何置信度低时，按配置使用 majority、last-beam 或 transition fallback。
- [x] 2.6 为每个预测输出记录 GPS coverage、fallback status、beam offset、algorithm 参数和 per-sample diagnostics。

## 3. 数据集适配与防泄漏

- [x] 3.1 实现从现有 MMW sequence/batch 构造 `GpsWindowSample` 的适配器，只读取预测时刻之前的 GPS/pose 历史窗口。
- [x] 3.2 增加 target_test 输入 guard，禁止 prediction/calibration 读取 target_test future beam、beam_power argmax、path/radio/channel oracle 字段。
- [x] 3.3 实现 calibration split 选择：source-only calibration 与 target_adapt labeled support calibration，并记录样本数和 split 来源。
- [x] 3.4 写出 oracle usage metadata，合法 GPS-only run 的 `used_target_oracle_fields` 必须为空列表。

## 4. 评估、参数搜索与调参闭环

- [x] 4.1 实现 GPS baseline runner，输出与现有 HiST-Beam 评估兼容的 score tensor、labels、sample metadata 和 metrics。
- [x] 4.2 复用现有 Top-1/Top-3/Top-5、coarse/fine、beam power dB 和预测直方图计算逻辑。
- [x] 4.3 实现 deterministic 参数网格搜索，覆盖 window size、smoothing、velocity decay、beam offset、score width、fallback 权重。
- [x] 4.4 确保参数排序只使用 calibration split，target_test 仅用于最终评价。
- [x] 4.5 实现 iteration report：记录每轮参数、calibration metrics、final eval metrics、误差分桶、预测直方图和 run id。
- [x] 4.6 实现 next-candidate summary：根据 calibration 排名、beam offset 诊断和误差分桶推荐下一轮候选参数。

## 5. CLI、配置与全场景矩阵

- [x] 5.1 新增 GPS window baseline CLI，支持 `--config`、`--scenes`、`--source-scenes`、`--target-scenes`、`--sweep`、`--execute`、`--output-dir` 和 `-o key=value`。
- [x] 5.2 在 `pyproject.toml` 注册 console script，例如 `kd-sensing-gps-window-baseline`。
- [x] 5.3 新增最小 smoke 配置，用于单场景或少量样本快速验证。
- [x] 5.4 新增 all-scenes 配置，枚举本地 ready MMW scenarios 并输出 per-scene summary。
- [x] 5.5 新增 source-target matrix 配置，支持 sunny Town10 三场景之间的指定 source/target 评估。
- [x] 5.6 输出 plan-only artifact，记录算法、参数网格、场景、split、claim scope 和输出目录。

## 6. 测试与验证

- [x] 6.1 添加 synthetic geometry 单元测试，验证 angle-to-beam、beam offset、环形距离和 top-k neighbor 排序。
- [x] 6.2 添加 predictor 单元测试，覆盖 `geometry_last`、`constant_velocity`、angle smoothing 和 fallback。
- [x] 6.3 添加防泄漏测试，验证 target_test calibration、future beam、beam_power argmax、path/radio/channel oracle 违规会标记不合格或失败。
- [x] 6.4 添加 CLI help 测试，使用 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` 或等价测试命令验证入口可发现。
- [x] 6.5 添加 artifact 测试，验证 `metrics.json`、prediction artifact、iteration report 和 next-candidate summary 的关键字段。
- [x] 6.6 运行相关测试：`conda run -n kd_mm_beam pytest tests/test_gps_window_baseline.py tests/test_cli_help.py -q`。
- [x] 6.7 运行 OpenSpec 校验：`openspec validate add-gps-window-baseline-beam-predictor --strict`。

## 7. 小规模实跑与结果驱动调整

- [x] 7.1 使用 smoke 配置执行一次单场景 GPS baseline：`conda run -n kd_mm_beam kd-sensing-gps-window-baseline --config <smoke-config> --execute --output-dir outputs/gps_window_baseline_smoke`。
- [x] 7.2 检查 smoke metrics、prediction histogram、GPS coverage、fallback 使用率和 oracle usage metadata。
- [x] 7.3 执行一次小网格 sweep，比较 `geometry_last`、`constant_velocity`、angle smoothing 和 beam offset calibration。
- [x] 7.4 根据 calibration split 结果调整默认参数，确保调整理由写入 iteration report，不使用 target_test 选择默认值。
- [x] 7.5 输出一份全场景 summary，列出每个场景的 Top-K、DBA、coverage、fallback rate、majority/last-beam/transition 对比和下一轮建议。

## 8. BeamBench-style 校准增强

- [x] 8.1 新增 `boresight_angle_degrees` 与 `auto_calibrate_boresight_angle`，在 angle-to-beam 映射前做角度中心化。
- [x] 8.2 新增 target_adapt support holdout：fit 子集估计 mapping/boresight，selection 子集用于 sweep 排名。
- [x] 8.3 在 metrics、run metadata、prediction diagnostics 和 iteration report 中记录 effective boresight、fit/selection split 与样本数。
- [x] 8.4 补充 boresight 校准与 support holdout 的单元测试，并更新 target-calibrated 配置。
