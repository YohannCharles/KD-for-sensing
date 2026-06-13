## Why

当前项目已经具备 GPS-conditioned JEPA、JEPA 下游复用、GPS-query pooling、Vision-Position baseline 和离线 JEPA 可视化分析能力，但还缺少一个把这些能力组织成论文级压力测试的统一基准。为了支撑“GPS 在分布偏移中成为 shortcut / spurious feature，而 Image-JEPA 的预测 latent 表征能保持鲁棒性”的研究结论，需要把 GPS 崩溃、视觉退化和异步漂移设计成可复现、可审计、可画图的实验矩阵。

## What Changes

- 新增 JEPA vs GPS shortcut benchmark 能力，用统一 benchmark manifest 描述模型组、扰动场景、训练或评估协议、随机种子、输出表格和论文图。
- 新增 deterministic perturbation 契约，覆盖 clean GPS、Gaussian jitter、cumulative drift、missing/dropout、GPS as distractor intervention、fog/rain、night、occlusion、motion blur 和 image/GPS 时间错位。
- 将 Scenario C 明确为 Asynchronous Position Feedback Benchmark：预测目标始终保持当前 beam label `y[t]`，只对 GPS 输入构造 delayed/stale/low-rate/dropout 条件，并输出 `gps_valid_mask`、`gps_delay_steps` 和可审计的 source metadata，防止未来 GPS 泄漏。
- 新增 benchmark runner / config contract，用于复用现有 `kd-sensing-train`、`kd-sensing-evaluate` 与 `kd-sensing-jepa-visual-analysis` 能力，而不是新增旧式仓库根脚本。
- 新增论文级对比产物，包括 GPS noise/dropout 曲线、image degradation 曲线、temporal delay 曲线、模型矩阵汇总、drop-GPS / misleading-GPS 反事实表、modality reliance 诊断和 caveat 报告。
- 扩展现有 JEPA visual analysis 的鲁棒性切片契约，使它可以消费 benchmark 级 perturbation manifest，并输出跨模型、跨强度、跨场景的统一鲁棒性表和图。
- 不引入 breaking change；现有 baseline、JEPA downstream 配置、训练入口和分析入口继续保持兼容。

## Capabilities

### New Capabilities

- `jepa-gps-shortcut-benchmark`: 定义 Image-JEPA 与 GPS-centric baseline 在 GPS collapse、视觉退化、异步漂移和 GPS distractor intervention 下的可复现实验矩阵、runner 输入输出、指标、图表和复现 metadata。

### Modified Capabilities

- `jepa-visual-analysis-suite`: 将现有轻量 test-time robustness slicing 扩展为可由 benchmark manifest 驱动的多场景 perturbation sweep，并要求输出统一鲁棒性表、曲线、warnings 和报告段落。

## Impact

- 代码影响：新增或扩展 `src/kd_sensing/diagnostics/` 或相邻 benchmark/analysis 模块中的 perturbation、benchmark manifest 解析、runner 编排和图表导出逻辑；Scenario C transform 必须显式保留当前 label、禁止未来 GPS、暴露 validity/delay metadata；复用现有模型 registry、dataset/runtime metadata、evaluation metrics 和 JEPA visual analysis 产物结构。
- 配置影响：新增 benchmark analysis config / YAML presets，引用现有 GPS-only、Camera AE + GPS、ResNet/Transformer image+GPS、JEPA mean pooling 和 JEPA GPS-query pooling 配置；新增 C0_sync、C1_mild_stale、C2_low_rate、C3_random_async、C4_severe_async preset；不得要求提交真实数据、checkpoint、训练输出或缓存。
- CLI 影响：优先扩展现有包内 CLI 或新增包内 console script；所有项目 Python 命令必须通过 `conda run -n kd_mm_beam ...` 运行；不得新增仓库根旧入口或兼容聚合脚本。
- 测试影响：需要覆盖 perturbation determinism、manifest schema、模型输入 shape 不变性、指标表字段、图表/报告降级行为和 CLI help smoke。
