## ADDED Requirements

### Requirement: Scene31-34 missing-modality mainline documentation
主线实验文档 MUST 记录 Scene31-34 pooled multi-scene 缺失模态主实验的当前地位、运行入口、指标口径、输出边界和 claim 状态。文档 MUST 明确 `prototype + random subset exposure` 是冻结主方法候选，Uniform 是 ablation，reliability fusion 与 PatternFiLM 不晋升。

#### Scenario: 主线目录记录 Scene31-34 主设定
- **WHEN** 开发者阅读 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 指向 Scene31-34 主实验 runner、summary、missing-count figures、paper tables 和 final conclusion 的本地输出路径
- **AND** 文档 MUST 说明真实 metrics、figures、tables、logs 和 checkpoint 仍属于 ignored runtime artifacts，不纳入源码变更

#### Scenario: 文档不推广 excluded methods
- **WHEN** 文档描述 Scene31-34 缺失模态主实验
- **THEN** 文档 MUST 不把 reliability fusion、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 或 weakKD 写成下一步主线搜索方向
- **AND** AMR/AMBER-lite MUST 只作为可选 multi-scene maskfix external baseline 说明

#### Scenario: 文档记录论文 baseline 与成本补齐
- **WHEN** 文档描述 Scene31-34 主实验的最终论文产物
- **THEN** 文档 MUST mention classifier baselines、AMR/AMBER-lite maskfix external baselines、compute profile table and final all-baseline paper tables as local/manual outputs
- **AND** generated metrics、profile CSV、paper tables and conclusions MUST remain ignored runtime artifacts under `outputs/`
