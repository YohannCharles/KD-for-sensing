## ADDED Requirements

### Requirement: Post-C2 public CLI 必须收敛到主线、MMW 和治理入口
项目在 post-C2 清理后 MUST 只把仍维护的主线训练/评估/预处理、final C2 或缺失模态评估、MMW/CSI workflow、以及必要治理/claim 入口声明为 public console script。非主线 dashboard、preview、architecture summary、training throughput、dataset audit、source-audit 或历史复现 CLI MUST 删除、降级为 internal-only，或在 lifecycle 文档中明确保留理由。

#### Scenario: 非主线 CLI 删除前同步引用
- **WHEN** implementation 从 `pyproject.toml` 删除某个 `kd-sensing-*` console script
- **THEN** README、docs、OpenSpec current specs、CLI help smoke 和 project surface inventory MUST 不再把该命令描述为 current public entrypoint
- **AND** 删除后项目 MUST 不提供同名 console script、module alias 或 thin wrapper

#### Scenario: 保留 CLI 有生命周期锚点
- **WHEN** post-C2 清理后某个 `kd-sensing-*` console script 仍保留
- **THEN** 它 MUST 在 inventory 或 current docs 中有 owner module、run class、输出边界和 focused validation
- **AND** 它 MUST 不依赖已删除的非主线 script 或 historical config

### Requirement: MMW 入口必须继续可发现
MMW 相关 package CLI、数据准备入口和必要 local/manual helper MUST 在 post-C2 清理中保留生命周期说明。删除其它非主线入口时 MUST 不让 MMW users 失去当前推荐运行、plot、compare、inspect 或 preparation 路径。

#### Scenario: MMW public CLI 保留
- **WHEN** implementation 更新 public CLI lifecycle
- **THEN** `kd-sensing-mmw-town-gps-v2` 和 `kd-sensing-inspect-mmw-physics` 或等价 MMW current CLI MUST 保留，除非另有独立 MMW change 替代
- **AND** README/docs MUST 继续指向 MMW current package CLI，而不是退回已退役脚本

#### Scenario: MMW local helper 不被误判
- **WHEN** `scripts/` 或 `scripts/mmw/` 中的 helper 仍服务 MMW 数据准备或 label distribution 诊断
- **THEN** inventory MUST 将其分类为 MMW dataset preparation、research diagnostic 或 local/manual helper
- **AND** 架构边界检查 MUST 不仅因其位于 `scripts/` 就要求删除

### Requirement: 一次性脚本删除必须保留结论
只服务历史 sweep、人工复盘、临时诊断或旧 runbook 的 `scripts/` 文件 MUST 被删除或降级为 historical/local manual。删除时 MUST 把仍有价值的结论、caveat、复跑方式或替代入口记录到 current docs、inventory、claim registry 或 mainline experiment history。

#### Scenario: 历史分析脚本删除
- **WHEN** implementation 删除 `scripts/analyze_*`、`scripts/summarize_*`、`scripts/diagnose_*` 或旧 Scene31 shell runbook
- **THEN** deletion ledger MUST 记录该脚本不属于主线、MMW、current docs/specs/tests 或 claim provenance
- **AND** 若脚本产出的结论仍被论文或组会材料需要，MUST 先迁移摘要到 docs 或 claim notes

#### Scenario: final C2 和主线 helper 保留
- **WHEN** `scripts/` 中的 launcher、summary 或 helper 被 final C2、当前缺失模态主线或 protected YAML/manifest 消费
- **THEN** implementation MUST 保留该脚本或先提供等价 owner
- **AND** 删除候选 MUST 标记为 protected-mainline，而不是 historical one-shot
