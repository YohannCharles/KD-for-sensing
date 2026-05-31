## ADDED Requirements

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 no-KD supervised baseline、HiST-Beam LOSO、MMW sensor-assisted、history-anchored residual、target prior/prototype/calibration、Raymobtime s008 selection、CSI hardening 和 viewer manifest。KD baseline、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional 或 historical/supporting workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐旧诊断脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `scripts/eval_modality_subsets.py` 或 `scripts/eval_modality_perturbation.py`
- **AND** 若需要 subset 评估，文档 MUST 指向配置化 evaluation 或包内 CLI 路径

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前 few-shot cross-scene 主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行 HiST-Beam/MMW target adaptation

### Requirement: 健康检查反映保留入口
快速健康检查 MUST 覆盖当前仍支持的架构边界、包内 CLI、viewer manifest、Raymobtime、modality visual diagnostics 和 HiST-Beam/MMW 相关 focused tests。健康检查 MUST 不要求已退役的模态失衡诊断脚本或 fusion KD virtual alias 可用。

#### Scenario: focused validation 不依赖退役入口
- **WHEN** 开发者执行本 change 的 focused 验证
- **THEN** 验证命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 不包含已退役的独立模态诊断脚本
- **AND** 验证 MUST 覆盖配置加载失败、架构边界和保留 evaluation subset 能力
