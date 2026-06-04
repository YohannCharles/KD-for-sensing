## ADDED Requirements

### Requirement: DeepSense6G Top8 selector 配置驱动工作流
项目 MUST 提供 `configs/deepsense6g_top8_selector.yaml`，用于驱动 DeepSense6G GPS Top8 Candidate Selector 的 manifest、训练、评价、绘图和 GPS v2 comparison。配置 MUST 声明数据场景、label space、GPS v2 sweep root、TopK analysis dir、output root、candidate topk、optional modality、model、attention、loss、train、experiment、metrics 和 outputs。

#### Scenario: 默认配置字段
- **WHEN** 开发者查看 `configs/deepsense6g_top8_selector.yaml`
- **THEN** 配置 MUST 包含 scenario31-34、`mapping_disabled`、`num_beams=64`、`gps_v2_default_support_ratio=0.15`、`topk=8`、`require_saved_logits=true` 和 `output_root=outputs/analysis/deepsense6g_top8_selector`
- **AND** 配置 MUST 包含默认 ablations 和 `target_adapt_beambench_top8_selector` protocol

#### Scenario: 命令行覆盖 support ratio 和 label space
- **WHEN** 用户通过 Top8 selector CLI 传入 `--support-ratio`、`--label-space` 或 `--topk`
- **THEN** 系统 MUST 使用命令行值覆盖配置默认值
- **AND** 输出目录 MUST 按 ratio tag 和 label space 分离

#### Scenario: 运行产物保存配置快照
- **WHEN** Top8 selector workflow 完成一次运行
- **THEN** result dir MUST 保存 resolved config 或等价配置快照
- **AND** run metadata MUST 记录 workflow、support ratio、label space、topk、train mode、ablation、source scenes、target scene、support/query count、GPS v2 artifact path 和 query label usage

### Requirement: Top8 selector 验收命令
项目 MUST 记录并支持 Top8 selector 的分层验收命令。所有项目相关 Python 命令 MUST 使用 `conda run -n kd_mm_beam` 环境运行。

#### Scenario: manifest 验收命令
- **WHEN** 开发者运行 manifest 验收
- **THEN** 推荐命令 MUST 为 `conda run -n kd_mm_beam kd-sensing-prepare-deepsense6g-top8-candidate-manifest --config configs/deepsense6g_top8_selector.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`
- **AND** 命令 MUST 写出 `manifest/top8_candidate_manifest.csv`

#### Scenario: selector 训练评价验收命令
- **WHEN** 开发者运行 selector workflow 验收
- **THEN** 推荐命令 MUST 为 `conda run -n kd_mm_beam kd-sensing-run-deepsense6g-top8-selector --config configs/deepsense6g_top8_selector.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8`
- **AND** 命令 MUST 写出 summary、predictions、selection events 和 run metadata

#### Scenario: plot 与 comparison 验收命令
- **WHEN** 开发者运行 plot 与 comparison 验收
- **THEN** 推荐命令 MUST 覆盖 `kd-sensing-plot-deepsense6g-top8-selector` 和 `kd-sensing-compare-deepsense6g-top8-selector-with-gps-v2`
- **AND** plotter MUST 写出 `figures/`
- **AND** comparison MUST 写出 `comparison_with_gps_v2.csv` 和 `comparison_report.md`

#### Scenario: 测试验收命令
- **WHEN** 开发者运行 Top8 selector 单元测试
- **THEN** 推荐命令 MUST 覆盖 `tests/test_topk_candidate_manifest.py`、`tests/test_topk_candidate_selector.py`、`tests/test_topk_candidate_losses.py`、`tests/test_candidate_attention_selector.py` 和 `tests/test_circular_metrics.py`
- **AND** 最终回归仍 MUST 使用 `conda run -n kd_mm_beam pytest -q`

### Requirement: Top8 selector README 工作流说明
README MUST 新增 “DeepSense6G GPS Top8 Candidate Selector” 章节。该章节 MUST 保持 quickstart 风格，说明为什么从 residual correction 改成 Top8 selector、候选生成与 selector 输入输出、candidate soft label、GPS prior fusion、miss head、完整运行流程、结果文件和有效性判断。

#### Scenario: README 说明主方法与反例
- **WHEN** 用户阅读 README 的 Top8 selector 章节
- **THEN** 文档 MUST 明确 GPS v2 是 candidate generator
- **AND** 文档 MUST 明确其他模态只做候选内选择或重排
- **AND** 文档 MUST 说明 64 类 direct modality prediction 和 no-GPS-prior fusion 不是主推荐方法

#### Scenario: README 说明结果判读
- **WHEN** 用户阅读 README 的结果判读说明
- **THEN** 文档 MUST 指向 `summary_overall.csv`、`summary_by_scene.csv`、`summary_by_top8_hit_miss.csv`、`predictions.csv`、`selection_events.csv` 和 `comparison_report.md`
- **AND** 文档 MUST 说明如何判断 selector 是否超过 GPS top1 baseline、是否接近 Top8 oracle、target-in-Top8 样本是否提升、scenario32/34 是否受 Top8 上限限制、camera AE 是否优于 GPS context-only selector
