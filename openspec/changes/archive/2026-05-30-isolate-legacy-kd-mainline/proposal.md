## Why

项目当前研究主线已经收敛到少样本跨场景波束预测：source 场景训练、target 场景少量标签自适应。这个问题的核心矛盾是跨场景绝对 beam ID 语义漂移、历史 beam anchor 缺失和场景私有校准不足，而不是 teacher-student 模型压缩或蒸馏知识迁移。

继续让 KD 配置、teacher checkpoint、teacher/student forward 和蒸馏 loss 参与主干训练路径，会污染后续 HiST-Beam、history-anchored residual、adapter/prototype/calibration 等方法开发，并增加每个新功能都要兼容 KD 的技术债。因此需要把 KD 从 active mainline 中隔离，同时保留历史代码、历史实验和可选 baseline 的可追溯性。

## What Changes

- 新增 legacy KD 隔离能力：
  - 将 teacher-student KD 主流程、KD loss、temperature/alpha/kd_weight、teacher checkpoint 解析、feature/relation distillation 和 KD-specific evaluator/script 从默认 active 方法路径中移出。
  - 保留历史 KD 代码为 `legacy` 或明确标注的 optional baseline，默认不参与新主线配置、quick validation、README 推荐入口和主结论 summary。
  - 为需要保留的 KD baseline 定义显式 opt-in 入口、metadata 和评估边界，避免与 no-KD 主线、history-anchored residual 或 adapter/prototype 方法混淆。
- 调整项目架构约束：
  - `kd_sensing.distillation` 若继续存在，只承担轻量、纯张量级算法或 legacy/baseline 支持；不得构建模型、读取 dataset、解析 checkpoint 或侵入训练主循环。
  - 新增方法扩展必须以 no-KD supervised/adaptation 路径为默认，不得要求 teacher runtime 或 KD 配置字段存在。
  - 架构边界测试应拒绝 active mainline 重新依赖 legacy KD 聚合入口。
- 调整 HiST-Beam 跨场景适配要求：
  - HiST-Beam、history-anchored residual、shared/private、adapter/prototype/calibration 的主线训练和 few-shot adaptation 默认不启用 teacher-student KD。
  - KD 只能作为显式 baseline 或后续增强实验，例如 LLM/strong-model teacher 到 lightweight student、privileged modality distillation 或 self-distillation regularizer；这些不属于当前主线默认行为。
- 调整 soft beam label 命名和语义边界：
  - beam-aware soft label / angular soft target 是 beam-space prior 或 label smoothing，不等价于 KD soft target。
  - 默认 no-KD supervised loss 可以使用 beam soft target，但字段、配置和日志命名不得暗示 teacher-student distillation。
- 文档与配置清理：
  - README 和推荐配置应把当前主线描述为 few-shot cross-scene beam prediction / history-anchored adaptation，而不是 KD-first 项目。
  - KD 相关配置文件、脚本和文档保留时必须标明 legacy、baseline 或 optional，不得作为默认 quickstart 或主线实验矩阵。

## Capabilities

### New Capabilities
- `legacy-kd-isolation`: 定义 KD 退役、隔离、可选 baseline、metadata、入口生命周期和防回流约束。

### Modified Capabilities
- `project-architecture`: 增加 active mainline 与 legacy KD 的模块边界、轻量导入和架构测试要求。
- `hist-beam-cross-scene-adaptation`: 增加 HiST-Beam 主线 no-KD 默认语义，约束 KD 只能作为显式 optional baseline/regularizer。
- `soft-beam-label-training`: 澄清 beam-aware soft target 与 KD soft target 的差异，要求命名、日志和配置避免把 label smoothing 误标为 KD。
- `experiment-workflow`: 调整默认实验配置和推荐入口，避免 KD-first 配置继续作为主线 quickstart；保留 KD 实验时必须显式标注 baseline/legacy。

## Impact

- 主要受影响代码：
  - `src/kd_sensing/distillation/`：保留纯算法工具或迁移到 legacy/baseline 边界；移除 active mainline 对运行时构建职责的依赖。
  - `src/kd_sensing/engine/`：训练、验证、evaluation、optimizer/checkpoint 构建路径默认不要求 teacher runtime；KD 作为显式 opt-in extension。
  - `src/kd_sensing/config/` 和 `configs/`：默认主线配置移除或禁用 KD 字段；legacy KD 配置独立命名并标注不可用于主结论。
  - HiST-Beam/LOSO 配置和 summary：主线矩阵以 no-KD source-only、adapter/prototype、history-anchored residual 和 calibration 为核心；KD baseline 分开汇总。
  - README/docs/OpenSpec：更新项目叙述、入口边界和历史 KD 保留策略。
- 需要新增或扩展测试：
  - 架构边界测试：active code path 不新增 legacy KD 聚合依赖，默认配置不加载 teacher checkpoint。
  - 配置解析测试：默认 no-KD 主线无需 KD 字段；legacy KD 配置必须显式 opt-in。
  - soft label 测试：beam soft target 的配置、日志和 loss 字段不使用 KD 命名。
  - summary/metadata 测试：KD baseline 与主线 run 可区分，`main_conclusion_eligible` 不被误置为 true。
- 不新增外部依赖；所有项目相关 Python 验证命令继续使用 `conda run -n kd_mm_beam`。
