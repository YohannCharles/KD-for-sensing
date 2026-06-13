## Context

项目当前已经支持 DeepSense6G / BeamBench 风格的 image、GPS 与 image+GPS fusion 训练评估，已有 GPS-conditioned JEPA 预训练、JEPA context encoder 下游复用、GPS-query pooling、Vision-Position baseline preset，以及 `kd-sensing-jepa-visual-analysis` 离线分析入口。现有能力能回答“某个配置在 clean split 上表现如何”，也能做轻量 drop modality / GPS noise / image masking slicing，但还没有把 GPS shortcut 失效构造成一个统一、可复现、可审计的论文级 benchmark。

本 change 的核心是把研究问题显式转成实验系统问题：GPS 在 clean 或同分布条件下可能是强 shortcut，但在 urban drift、missing、错误同步或人为 misleading intervention 下会变成 spurious feature；Image-JEPA 的优势必须通过 latent predictive representation 在这些 regime 下的稳定性来验证，而不是只用普通 augmentation 证明鲁棒性。

## Goals / Non-Goals

**Goals:**

- 提供一个 benchmark manifest，用统一 schema 声明模型组、扰动套件、强度 sweep、种子、split、指标、图表和报告输出。
- 复用现有训练、评估和 JEPA visual analysis 入口，保持 `src/kd_sensing` 包内架构和现有模型 registry 边界。
- 支持 GPS reliability collapse、image physical degradation、asynchronous multimodal drift 和 GPS as distractor intervention 四类核心压力测试。
- 将 asynchronous multimodal drift 收敛为 Scenario C / Asynchronous Position Feedback：当前图像序列和 beam label 对齐在 `t`，GPS 可以来自更早时刻、低采样率轨迹或缺失反馈。
- 输出论文可用的 robustness table、curve、counterfactual table、modality reliance summary、case payload 和 caveat report。
- 让所有 perturbation deterministic、可复现、可在 manifest 中审计，并避免修改训练 checkpoint、split CSV、真实 dataset 或既有 run 目录。

**Non-Goals:**

- 不重新设计 Stage 1 JEPA 预训练目标，不改变 target encoder EMA、mask sampler 或 checkpoint schema。
- 不引入旧 KD、旧 GPS residual、Top8 selector、Raymobtime s008 或 Multimodal-NF 路线。
- 不把本地 `outputs/`、`logs/`、cache、checkpoint 或真实数据纳入源码。
- 不承诺真实论文指标一定优于 baseline；系统只提供可复现 benchmark 与防过度声称的报告结构。

## Decisions

### Decision 1: 用 benchmark manifest 编排实验，而不是新增硬编码脚本

新增 YAML/JSON manifest 作为单一入口，声明 `models`、`perturbation_suites`、`sweeps`、`metrics`、`figures`、`seeds` 和 `outputs`。runner 只解析 manifest 并调用现有 dataset/model/evaluation/analysis 组件。

替代方案是新增多个专用脚本，例如 `run_gps_noise.py`、`plot_gps_dropout.py`。这种方式会复制配置解析、模型加载和指标逻辑，也容易违反当前“不要新增旧入口”的项目边界。manifest 方式让 benchmark 可组合、可审计，也方便把同一 perturbation 输入给 JEPA visual analysis。

### Decision 2: perturbation 在评估/分析数据流中按需注入，而不是改写 dataset 或 split

GPS jitter、drift、dropout、distractor swap、image fog/night/occlusion/blur 和 temporal delay 都实现为 deterministic runtime transform。transform 接收 sample metadata、全局 seed、suite id、severity 和 split 信息，输出被扰动的 batch，同时保留原始 sample id 和 target。

替代方案是生成新的 CSV 或 cache 数据集。那样会把实验产物和输入数据混在一起，增加泄漏和提交本地产物的风险。runtime transform 更符合当前 dataset runtime contract，也能用单元测试验证 shape、seed 和 metadata。

### Decision 3: Scenario C 只移动 GPS 输入，不移动 label 或当前视觉序列

Asynchronous Position Feedback 的语义固定为预测当前 `y[t]`，同时把 GPS 构造成 `G[t-delta]`、低频 forward-fill/stale value 或 dropout/missing。实现必须为模型和报告提供 `gps_valid_mask`、`gps_delay_steps`、可选 source index/timestamp metadata，并保证任意时间步都不会读取未来 GPS。若真实 timestamp 可用，优先选择 `time <= image_time - delta_t` 的最近 GPS；否则使用 frame-index delay。

替代方案是把整个样本或 label 一起 shift。那会把任务改成另一个同步预测问题，掩盖 stale GPS 对当前 beam 的伤害，也会削弱 Image-JEPA 是否能补偿 GPS shortcut 失效的研究结论。

### Decision 4: benchmark 只消费已有或显式声明的模型配置与权重

模型组使用现有 Vision-Position baseline 与 JEPA downstream 配置：GPS-only neural、Camera AE + GPS、ResNet/Transformer image+GPS、JEPA mean pooling、JEPA GPS-query pooling。manifest 必须显式声明每个模型的 config、weights 或 training plan；评估时不得从不匹配 scene/scenegroup registry 静默回退。

替代方案是 runner 自己构建模型细节。那会绕过 `MODELS` registry 和当前配置生命周期。显式 config/weights 能保留场景、split、normalization artifact 和 checkpoint provenance。

### Decision 5: 指标分为任务性能、鲁棒性下降和 shortcut 依赖三层

任务性能继续复用 Top-K、DBA、loss 等正式指标。鲁棒性下降记录 clean baseline、每个 severity 的 absolute metric、delta、relative drop 和 area-under-robustness-curve。shortcut 依赖通过 drop GPS、misleading GPS、GPS delay、GPS-only collapse slope、attention/gradient/ablation summary 等诊断近似表达。

替代方案是只画 accuracy 曲线。曲线直观但不足以审计论文 claim；三层指标能把“性能好”“退化慢”“不依赖 GPS shortcut”分开，报告中也能清楚标记不能过度声称的部分。

### Decision 6: JEPA visual analysis 负责图表和报告，benchmark runner 负责矩阵调度

benchmark runner 输出结构化 `benchmark_manifest.json`、`robustness_summary.csv`、`metrics_by_condition.csv` 和 perturbation cache metadata。`kd-sensing-jepa-visual-analysis` 扩展为可读取这些产物或同一个 benchmark manifest，生成论文图、case study、attention/reliance 图和 `report.md`。

替代方案是把所有图表逻辑放进 runner。分开后 runner 更容易测试，visual analysis 继续承担已有的离线分析职责，也减少训练/评估路径被可视化依赖污染的风险。

## Risks / Trade-offs

- [Risk] GPS drift 或 distractor 设计可能不符合具体场景的物理语义 → Mitigation: manifest 必须记录 perturbation type、参数、seed、scene/split 和 caveat；默认报告不得把 synthetic intervention 写成真实环境结论。
- [Risk] 图像退化实现过度像普通 augmentation，无法支撑 wireless coupling 叙事 → Mitigation: degradation suite 必须记录 physical label，例如 fog/rain、night、occlusion、motion blur，并在报告中分开呈现，不把它们混成单一 augmentation 指标。
- [Risk] 不同模型 checkpoint 使用不同 split 或 normalization artifact → Mitigation: runner 必须校验 split metadata、样本数、label space、metric profile 和 normalization provenance，不一致时 fail fast 或标记不可比较。
- [Risk] 扰动 sweep 计算量大 → Mitigation: 支持 severity 子集、样本上限、缓存只读复用和 smoke profile；默认测试使用 mock/synthetic batch，不依赖真实数据。
- [Risk] modality reliance 的 attention 或 gradient 诊断不是严格因果证明 → Mitigation: 报告必须区分 counterfactual perturbation 结果和解释性诊断，不把 attention/gradient 单独作为因果证据。
- [Risk] Scenario C 若只做 batch roll 或同步 shift label，会误把异步反馈写成另一个干净任务 → Mitigation: spec 和测试必须覆盖 label 不变、当前图像不变、早期历史 invalid、无未来 GPS 泄漏、mask/delay metadata 与 deterministic replay。

## Migration Plan

1. 新增 benchmark manifest schema、perturbation transforms 和 focused 单元测试。
2. 补强 Scenario C transform，使其生成 stale/low-rate/dropout GPS、validity/delay metadata，并用 toy sequence 测试验证无未来泄漏和 label 不变。
3. 新增或扩展包内 CLI / analysis 配置入口，先支持 evaluation-only benchmark，再接入可选 training plan。
4. 扩展 JEPA visual analysis，使其能读取 benchmark manifest 或 runner 产物并生成统一 robustness figures 与 report sections。
5. 添加 canonical smoke config，使用 mock/synthetic 数据验证 runner、metrics aggregation 和报告 schema。
6. 在 README 或实验文档中只登记入口与复现命令，不提交真实 outputs、checkpoint 或 cache。

Rollback 方式是删除新增 benchmark manifest/runner 配置和 CLI 声明；现有训练、评估、JEPA 预训练和 JEPA visual analysis 基础行为不需要迁移。

## Open Questions

- 默认论文主表应以 BeamBench-fair 2604-style split 为主，还是同时保留多 scene group 与 LOSO 视角。
- GPS distractor intervention 默认使用 scene 内随机错配、轨迹邻近错配，还是跨 scene 错配；实现阶段需要根据现有 metadata 可用性选择最稳妥的默认。
- modality reliance 若启用 gradient norm，需要确定是否对所有模型都可用；不可用模型应降级到 ablation/reliance summary。
