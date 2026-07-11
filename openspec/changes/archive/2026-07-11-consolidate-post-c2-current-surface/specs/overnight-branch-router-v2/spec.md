## MODIFIED Requirements

### Requirement: Overnight summary outputs
项目 MUST 保留 `scripts/summarize_overnight_branch_router_v2.py` 作为 final C2 summary 直接消费的 read-only supporting parser。该 parser MUST 能读取既有 overnight root 和 baseline roots，并继续提供 final C2 所需的 summary、drop-count、pattern 和 router diagnostics 数据；它 MUST 不重新启动训练或要求历史 launcher 存在。

#### Scenario: Final C2 复用历史 summary parser
- **WHEN** `scripts/summarize_final_c2_ablation_v1.py` 读取 overnight branch-router 结果
- **THEN** retained parser MUST 提供 final C2 当前使用的解析函数和缺失值语义
- **AND** parser MUST 只读取显式输入并将生成内容写入 ignored output root
- **AND** current docs MUST 将其标记为 supporting/historical parser，而不是推荐训练入口

### Requirement: Focused regression coverage
项目 MUST 保留 focused regression 覆盖 soft static weighting、oracle router target、masked softmax、focus pattern alias 和 retained summary parser。测试 MUST 使用合成或 fake artifact，不得读取真实 `dataset/`、依赖已删除 launcher 或写入受保护 runtime 产物。

#### Scenario: Retained parser focused pytest
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest -q tests/test_overnight_branch_router_v2.py tests/test_final_c2_ablation_v1.py`
- **THEN** 测试 MUST 覆盖保留的 router helper、summary parser 和 final C2 消费路径
- **AND** 测试产物 MUST 写入 pytest 临时目录或 ignored output root

## REMOVED Requirements

### Requirement: Overnight launcher matrix
**Reason**: A/B/C 训练矩阵和结果已经冻结，launcher 不再有 current config、CLI 或主动研究 workflow 消费；保留它只会维持历史 GPU 调度与失败恢复代码。
**Migration**: 历史命令、manifest 和结果 provenance 通过 dated OpenSpec archive、claim notes 与 git 查询；final C2 继续使用 retained read-only summary parser。

#### Scenario: 历史 launcher 退出 current surface
- **WHEN** post-C2 cleanup 完成
- **THEN** `scripts/launch_overnight_branch_router_v2.py` 和只覆盖其 dry-run/调度的测试 MUST 删除
- **AND** 项目 MUST 不提供同名 wrapper、alias 或替代 launcher framework
