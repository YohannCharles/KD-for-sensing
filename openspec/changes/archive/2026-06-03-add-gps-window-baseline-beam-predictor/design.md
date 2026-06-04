## Context

现有 MMW HiST-Beam 实验已经能运行跨场景 LOSO、few-shot target adaptation、history-anchored residual 和若干诊断 baseline。最近的 quick validation 暴露了三个问题：一是神经网络模型在部分 target 场景出现 source prior collapse 或负迁移；二是 v9 adapted eval 存在 AMP overflow 风险；三是严格可声明结果需要清晰区分 target oracle 使用边界。

本变更引入一个更基础的 GPS-only 非神经网络 baseline。它不训练模型、不加载 checkpoint，只读取预测时刻之前的 GPS/pose 滑动窗口，估计几何方位与短期运动，再映射为 beam 预测。该 baseline 的价值不是替代神经网络，而是作为可解释参照：判断“只靠几何和短期连续性”在所有场景中能做到什么水平，并为后续模型设计提供误差分桶。

## Goals / Non-Goals

**Goals:**

- 提供仅使用 GPS/pose 历史窗口的 beam prediction baseline，覆盖所有本地可用 MMW 场景和指定 source-target 矩阵。
- 支持多种无训练算法配置：last observed geometry、constant velocity、window smoothing、angle smoothing、beam transition fallback、top-k angular neighbor expansion。
- 输出可与现有 HiST-Beam 评估对齐的 metrics、prediction histogram、collapse diagnostics、per-scene summary 和参数迭代记录。
- 严格记录输入字段与防泄漏状态，确保预测不消费 future beam、target_test label、beam_power argmax、path/radio/channel oracle。
- 支持快速参数网格搜索和结果驱动的下一轮候选建议，用于逐步提高预测精度。

**Non-Goals:**

- 不新增神经网络、蒸馏、adapter、prototype 或 checkpoint 训练流程。
- 不把 target_test 标签用于参数选择、阈值选择或算法调参；target_test 只用于最终评价。
- 不改变现有 HiST-Beam、history-anchored 或 sensor-assisted 默认配置。
- 不声称跨 town/weather 结论，除非数据可用性与 split metadata 已支持对应协议。

## Decisions

### 1. 预测核心采用纯函数和 dataclass 配置

将 GPS baseline 拆成 `GpsWindowBaselineConfig`、`GpsWindowSample`、`GpsWindowPrediction` 和纯函数预测器。核心函数接收历史 GPS/pose、RSU pose、codebook/num_classes 和算法参数，返回 per-horizon logits 或 score 向量。

理由：纯函数易测、可复现，也避免把非神经网络 baseline 误接入训练器状态。备选方案是复用 PyTorch model 接口，但这会制造“模型训练”错觉，并增加 checkpoint/AMP/device 复杂度。

### 2. beam 映射优先复用几何 residual/角度工具

实现时优先复用已有几何工具中的相对位置、azimuth、angle-to-beam、环形距离和 residual 映射逻辑；如现有工具缺少 GPS pose 适配器，只新增薄包装函数。

理由：beam codebook 的环形语义必须与现有 residual/diagnostics 保持一致。备选方案是单独写一套角度到 beam 的公式，但容易出现 offset、方向或 modulo 不一致。

### 3. 算法族分层而不是一次写复杂优化器

默认算法按复杂度递进：

- `geometry_last`: 使用最后一个历史 GPS/pose 计算当前相对方位并映射 beam。
- `constant_velocity`: 用滑动窗口估计 CAV 速度，外推到预测 horizon 后映射 beam。
- `smoothed_angle`: 对历史相对角做环形移动平均或 Savitzky-Golay 平滑，再结合角速度外推。
- `transition_fallback`: 当 GPS/pose 缺失或几何置信度低时，回退到 source/target_adapt 中估计的 beam delta transition。
- `hybrid_grid`: 对窗口长度、平滑强度、速度衰减、beam offset、neighbor temperature 和 fallback 权重做小网格搜索。

理由：用户希望“根据每次结果调整算法”，分层算法能明确看到每一步带来的收益。备选方案是直接做黑盒超参搜索，但可解释性弱，也容易在 target_test 上过拟合。

### 4. 参数选择只允许使用 source split 或 target_adapt support

参数调优支持两种合法模式：source-only calibration 使用 source split；few-shot calibration 使用 target_adapt labeled support。target_test 标签只在最终评估阶段读取，不能参与排序、早停、调参或候选推荐。

理由：这与 MMW target adaptation protocol 的防泄漏边界一致。备选方案是对所有场景整体调参后报告平均结果，但如果包含 target_test 指标驱动，会污染主结论。

target_adapt support 可进一步确定性拆分为 calibration-fit 与 selection 两段。fit 段用于估计 direction、offset、boresight 和 fallback 统计；selection 段用于 sweep 排名。样本不足时退回全 support 选择，并在 metadata 记录原因。这样保留 few-shot 校准能力，同时降低 offset 在小 support 上偶然取优的风险。

### 4a. boresight 校准进入 angle-to-beam 前的角度中心化

新增 `boresight_angle_degrees` 与 `auto_calibrate_boresight_angle`。预测时先计算 `calibrated_azimuth = relative_azimuth - boresight_angle_degrees`，再走既有 `direction/offset` 和 beam score kernel。显式 boresight 可来自物理外参；自动 boresight 只能从合法 calibration split 标签估计，并把估计角度、score 和样本数写入 metrics/diagnostics。

理由：BeamBench 风格校准强调先把几何角度中心化到基站 boresight，再映射到 beam id。保留 discrete offset/direction 是为了在当前数据缺少稳定外参时仍能作为保守 fallback；二者的 effective 值必须分开记录，避免把物理旋转和 label-bin 枚举混在一起解释。

当前默认 target-calibrated 配置不自动启用 boresight 估计，只保留 `direction/offset` 自动校准作为主 baseline。自动 boresight 作为诊断或外参可用时的候选路径，需通过 support selection 或显式配置启用。

### 5. 输出 logits 形状对齐现有评估

GPS baseline 虽然不训练神经网络，仍输出形状 `[N, H, C]` 的 beam score/logits。score 可由环形距离核、top-k 邻域扩展和 transition prior 相加得到，然后复用现有 `calculate_hist_beam_metrics`、beam power 指标和预测 artifact 写出流程。

理由：统一指标和 artifact 结构能直接进入已有 summary/comparison 工具。备选方案是只输出 argmax beam，但会损失 Top-3/Top-5、直方图和 dB power 对比。

### 6. 新增独立 CLI 而不是扩展训练入口

新增类似 `kd-sensing-gps-window-baseline` 的 CLI，支持 `--config`、`--scenes`、`--source-scenes`、`--target-scenes`、`--sweep`、`--execute`、`--output-dir` 和 `-o key=value` override。该 CLI 只构建数据集和 runner，不调用训练器。

理由：该能力是 baseline/evaluation，不是 training。备选方案是塞进 `hist_beam_loso.py` 的 variant，但会让 stage 语义更复杂，并容易误触 target adaptation 逻辑。

## Risks / Trade-offs

- [Risk] GPS/pose 坐标系、RSU pose 或 beam codebook offset 方向不一致，导致几何预测系统性偏移。→ Mitigation：加入 beam offset 网格、角度-标签散点诊断、per-scene circular error 分桶和 synthetic pose 测试。
- [Risk] constant velocity 对路口急转弯或遮挡区域外推失败。→ Mitigation：支持速度衰减、短窗口/长窗口对比、角速度 clipping 和低置信 fallback。
- [Risk] 参数搜索如果使用 target_test 指标会泄漏。→ Mitigation：runner 强制区分 calibration split 与 evaluation split，metadata 记录 `used_target_test_for_calibration=false`，测试覆盖违规输入。
- [Risk] 仅 GPS baseline 可能远弱于 last-beam history baseline。→ Mitigation：报告中同时输出 last-beam/majority/transition 诊断指标，并明确其作为非神经网络参照。
- [Risk] 全场景 sweep 读取大量 manifest 可能较慢。→ Mitigation：支持 plan-only、resume、max-scenes/max-runs、小网格默认参数和可缓存的 per-scene sample table。

## Migration Plan

- 新能力以独立模块、独立配置和独立 CLI 形式加入，不迁移现有训练配置。
- 现有输出目录、checkpoint、logs 不变；新运行产物默认写入 `outputs/gps_window_baseline/`。
- 若新 CLI 或 baseline 出现问题，可删除/停用对应 console script 与配置，不影响 HiST-Beam 训练和评估入口。

## Open Questions

- 当前 MMW manifest 中 GPS/pose 字段是否已经稳定包含 RSU 和 CAV 坐标系；实现阶段需要对本地三场景做字段巡检。
- beam codebook 的几何角度零点和方向是否与 derived beam label 完全一致；如果不一致，需要默认启用可审计的 beam offset calibration。
- 参数推荐器第一版应只给出 deterministic grid ranking，还是同时输出启发式下一轮候选；建议第一版先 deterministic，避免目标不清的自动调参。
