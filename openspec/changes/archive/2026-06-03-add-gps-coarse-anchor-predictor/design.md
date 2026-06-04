## Context

当前项目已经具备三类相关能力：`gps_teacher`/`gps_student` 可做 GPS-only beam classification；HiST-Beam 已支持 coarse/fine 层次化 beam label；`gps_window` baseline 已实现 BeamBench-style 的几何 angle-to-beam、boresight 校准、环形距离和防泄漏 metadata。用户希望采用两阶段路线：先由 GPS 给出粗略位置/粗 beam anchor，再由其它模态学习相对 anchor 的残差。

BeamBench 的直接启发是：GPS 是低维且跨模态可匹配的空间先验，角度到 beam 的映射需要对基站 boresight 做场景校准；跨场景泛化应在 seen/unseen Scene 分开报告，Scene 31 这类未见场景能暴露 mobility pattern 和环境分布偏移；仅监督学习绝对 beam 可能泛化有限，后续多模态 late fusion 或残差学习应建立在稳定 anchor 上。

## Goals / Non-Goals

**Goals:**

- 实现一个显式 opt-in 的 GPS coarse anchor profile，能输出 coarse group logits、center beam、beam score、confidence 和 residual anchor metadata。
- 支持几何校准 anchor 与轻量神经 coarse head 两条路径，并用统一 `GpsCoarseAnchor` 契约交给评估和后续 residual 模型。
- 提供跨场景评估矩阵，至少覆盖 DeepSense6G Scenes 31-34 或本地可用 MMW LOSO，并明确区分 source scenes、target_adapt calibration 和 target_test evaluation。
- 复用现有 GPS window 几何工具、HiST-Beam coarse/fine label、Top-K/DBA/beam power 指标与 prediction artifact 写出逻辑。
- 为后续其它模态 residual head 预留接口：其它模态可读取 GPS anchor，但本次不实现完整 residual fusion sweep。

**Non-Goals:**

- 不把 image/radar/lidar/mmwave residual head 作为本变更的交付范围。
- 不改变现有 GPS-only `gps_teacher`/`gps_student` 默认 forward 返回值。
- 不替换或删除已经实现的 GPS window baseline CLI。
- 不使用 target_test label、beam_power argmax、channel/path/radio oracle 做 anchor 校准或参数选择。
- 不声称跨 town/weather 泛化，除非数据 availability 和 split metadata 支持对应协议。

## Decisions

### 1. Anchor 表达为 coarse beam/sector，而不是重新定义物理坐标回归

GPS 粗略位置在工程上落为校准后的方位、center beam 和 coarse group。默认 `num_classes=64`、`group_size=8` 时，anchor 输出 8 个 coarse group logits，并可同时输出 `[H, 64]` 的 beam score kernel。

理由：项目的训练、评估、artifact 和 residual label 都围绕 beam codebook；直接回归坐标会引入坐标系、尺度和 RSU 外参的新契约。备选方案是预测 `(x, y)` 或 range/azimuth 回归，但它不能直接进入现有 DBA/Top-K，也更难与后续 residual beam label 对齐。

### 2. 几何 anchor 是默认保底，神经 coarse head 是 opt-in 增强

第一版提供两种 anchor source：

- `geometry_calibrated`：复用 `gps_window` 的 boresight、direction、offset、angle-to-beam 和 beam score kernel，不训练模型。
- `gps_neural_coarse`：在 GPS encoder 后增加 coarse head，训练 coarse CE，可选 beam-level auxiliary CE 或 anchor confidence calibration。

理由：几何 anchor 跨场景可解释，适合作为保底和诊断；神经 coarse head 可以学习 GPS 噪声、轨迹模式和非线性映射，但需要防止 source prior collapse。备选方案是只做神经模型，风险是未见场景失败时很难判断是 GPS 信息不足、校准偏移还是模型过拟合。

### 3. 统一 `GpsCoarseAnchor` 契约

无论来自几何还是神经模型，系统都生成统一字段：

- `coarse_logits`: `[B, H, G]`
- `center_beam`: `[B, H]`
- `beam_scores`: 可选 `[B, H, C]`
- `confidence`: `[B, H]`
- `residual_anchor_beam`: `[B, H]`
- `metadata`: anchor source、boresight、calibration split、GPS coverage、oracle usage

理由：后续其它模态 residual head 只需要消费 anchor 契约，不必关心 anchor 来自几何还是神经模型。备选方案是把几何 baseline 和神经模型输出分开处理，但会让 residual 阶段重复适配两套输入。

### 4. Anchor-conditioned HiST-Beam 只显式 opt-in

在 HiST-Beam 中新增配置开关，例如 `hist_beam.gps_anchor.enabled=true`。启用后，fusion model 可将 anchor coarse distribution、center beam embedding 或 confidence 拼入 coarse/fine/residual 分支；未启用时现有 `v1_hierarchical`、adapter、prototype、history residual 等默认语义不变。

理由：GPS anchor 是新的实验假设，不应污染已有 quick validation 矩阵。备选方案是默认把 GPS anchor 注入所有 GPS 模态模型，但会让历史结果不可比。

### 5. 跨场景评价按 anchor 与 downstream 两层记录

GPS anchor evaluation 至少输出：

- anchor coarse accuracy、center beam Top-K、circular beam error、DBA 或不可用原因。
- seen/unseen scene breakdown，尤其标记 BeamBench-style Scene 31 held-out 或 MMW LOSO target。
- source-only、target_adapt calibrated 和 target_test final eval 三类阶段 metadata。
- residual preview：`true_beam - residual_anchor_beam` 的环形分布、残差熵和超出邻域比例。

理由：只有 anchor 本身足够稳定，后续残差学习才有意义；残差分布能提前判断其它模态需要补偿的是小偏移、系统性 offset 还是完全错误 anchor。备选方案是直接跑完整 residual fusion 后再看结果，但定位问题会慢很多。

### 6. 校准遵守 BeamBench 风格但保持防泄漏

Boresight、beam direction、beam offset 可从 source split 或 target_adapt support 估计；target_test 只用于最终评价。所有自动校准都记录样本数、split、effective boresight 和 `used_target_test_for_calibration=false`。

理由：BeamBench 的角度中心化是 GPS 到 beam 的关键步骤，但在跨场景实验里必须明确校准数据来源。备选方案是用全量场景统一拟合三阶 angle-beam 映射；如果混入 target_test，会破坏主结论。

## Risks / Trade-offs

- [Risk] GPS coarse anchor 在非 LoS、反射主导或坐标/boresight 偏移场景中系统性错误。→ Mitigation：保留 beam score 分布和 confidence，不强制下游只使用 top-1；输出 residual preview 和 per-scene circular error。
- [Risk] 神经 coarse head 学到 source scene beam prior，未见场景泛化弱。→ Mitigation：几何 anchor 作为主 baseline；神经 head 使用 coarse CE、label smoothing、source scene balanced sampling 和 seen/unseen breakdown。
- [Risk] target_adapt 小样本 boresight 校准偶然取优。→ Mitigation：支持 support holdout、记录 calibration/selection split；target_test 不参与选择。
- [Risk] anchor-conditioned HiST-Beam 增加模型接口复杂度。→ Mitigation：统一 dataclass/字典契约，默认关闭，测试覆盖启用/关闭两种 forward。
- [Risk] DeepSense6G 和 MMW 的 GPS 字段语义不同。→ Mitigation：anchor adapter 按 dataset family 分离，输出统一 anchor 字段；缺失字段时记录不可用原因而不是伪造结果。

## Migration Plan

- 新增模块、配置和测试，不迁移既有配置。
- 先实现 geometry anchor export 和 standalone evaluation，验证指标/artifact。
- 再实现 GPS neural coarse head 与 opt-in 训练配置。
- 最后接入 HiST-Beam anchor-conditioned forward，但默认 quick validation 不启用。
- 若新路线效果不稳定，可停用对应配置和 console/profile；现有 GPS window baseline、GPS-only 和 HiST-Beam 默认流程不受影响。

## Open Questions

- DeepSense6G 当前本地 CSV 中是否保留了 BeamBench 所需的场景 boresight/calibration 信息，还是只能依赖已有 GPS-Rel-Polar 特征。
- `group_size=8` 是否作为 GPS anchor 默认最优，还是需要在 residual preview 后比较 `group_size=4/8/16`。
- 神经 coarse head 是否应先只在 GPS-only 模型实现，还是直接作为 HiST-Beam 的可插拔 head；建议实现阶段先做 GPS-only anchor export，再接入 fusion。
