## ADDED Requirements

### Requirement: 所有 checkpoint 与 sidecar 必须原子发布并可校验
训练与 registry workflow MUST 让 `last.pth`、`best.pth`、`best_top1.pth`、`best_<selection>.pth` 和 registry 副本使用同一个原子 publisher。current sidecar MUST 作为发布完成标记，包含 checkpoint SHA-256、size、schema version 和 role；current reader MUST 在选择、恢复或最终评估前验证这些字段。

#### Scenario: 发布任一训练 checkpoint role
- **WHEN** 训练保存 last、objective-best、Top-1-best 或自定义 selection checkpoint
- **THEN** 系统 MUST 在同目录临时文件完成 checkpoint 写入与 flush 后原子替换目标
- **AND** 系统 MUST 基于最终 checkpoint 计算 SHA-256 和 size
- **AND** 系统 MUST 再原子发布包含相同摘要的 sidecar

#### Scenario: Checkpoint 写入中断
- **WHEN** checkpoint 序列化、flush 或 replace 之前发生异常
- **THEN** 既有目标 checkpoint 和 sidecar MUST 保持可读取且语义一致
- **AND** 临时文件 MUST 被清理或保持为不可选的未发布文件

#### Scenario: Sidecar 发布前中断
- **WHEN** 新 checkpoint 已替换但新 sidecar 尚未完整发布
- **THEN** current reader MUST 将该 checkpoint 视为未完成发布并 fail-closed
- **AND** reader MUST 不根据文件名或旧 sidecar 把它选为 best/resume/final-test candidate

#### Scenario: Digest 或 size 不匹配
- **WHEN** current checkpoint 的内容与 sidecar 记录的 SHA-256 或 size 不一致
- **THEN** resume、registry selection 和 final test MUST 拒绝该文件
- **AND** 错误 MUST 包含 checkpoint、sidecar 路径和不匹配字段

#### Scenario: Registry 副本发布
- **WHEN** best checkpoint 被归档到 scene/scenegroup registry
- **THEN** registry MUST 使用同一原子 checkpoint/sidecar publisher
- **AND** registry sidecar MUST 记录源 checkpoint digest 与 registry 副本 digest
- **AND** 源 run checkpoint MUST 不被移动或重写

### Requirement: 每个 checkpoint 文件必须记录自身准确的 selection provenance
每个 current checkpoint payload 与 sidecar MUST 记录该文件自身的 `checkpoint_role` 和 selection 对象，至少包含 metric、mode、value、selected epoch、source run 和 final-test candidate 状态。系统 MUST 不把 manager 的全局 selection 字段无差别复制到所有文件。

#### Scenario: Objective best provenance
- **WHEN** early-stopping/objective 指标产生 `best.pth`
- **THEN** 该文件 MUST 记录 objective metric、mode、value 和对应 epoch
- **AND** provenance MUST 不把该文件标为 Top-1 或自定义 selection checkpoint

#### Scenario: Top-1 与自定义 best provenance
- **WHEN** 训练分别发布 `best_top1.pth` 和 `best_<selection>.pth`
- **THEN** 每个文件 MUST 记录自身真实 selection metric、mode、value 和 epoch
- **AND** 两个文件的 provenance MUST 不因同 epoch 写出而互相覆盖

#### Scenario: Last checkpoint provenance
- **WHEN** 训练发布 `last.pth`
- **THEN** 该文件 MUST 记录 last/fixed-epoch role 和实际 epoch
- **AND** 它 MUST 不声称自己是 best checkpoint
- **AND** payload MUST 保留独立 selection catalog 指向此前真实 best 候选

#### Scenario: 恢复 selection catalog
- **WHEN** current checkpoint 被用于同 run 或跨 run resume
- **THEN** 恢复流程 MUST 加载每个候选的 role、路径、digest 和 source run
- **AND** final resolver MUST 只使用与请求策略一致且摘要验证通过的候选

### Requirement: Current checkpoint schema 不得伪造 test loss
新 checkpoint schema MUST 用 `validation_loss` 表达该 epoch 的真实 validation loss，用独立 final-test artifact 表达 test metrics。新 payload 和 sidecar MUST 不写把 validation loss 复制而来的 `test_loss`。

#### Scenario: Current checkpoint 保存 validation loss
- **WHEN** current schema 在运行过 validation 的 epoch 保存 checkpoint
- **THEN** payload MUST 记录真实 `validation_loss`
- **AND** payload MUST 不包含兼容 alias `test_loss`

#### Scenario: 未运行 validation 的 current checkpoint
- **WHEN** current epoch 跳过 validation 或 fixed-epoch workflow 没有 validation
- **THEN** payload MUST 将 `validation_loss` 记录为 `null` 或 unavailable
- **AND** payload MUST 不从历史 validation 或 final test 伪造该值

#### Scenario: Legacy test_loss alias
- **WHEN** unversioned legacy checkpoint 只有历史 `test_loss` 而没有 `best_val_loss`
- **THEN** 只有 resume legacy migration owner MAY 将其迁移为旧 validation-loss state
- **AND** current artifact reader、selection resolver 和 final-test writer MUST 不把它解释为真实 test evaluation
- **AND** load provenance MUST 记录迁移 warning

### Requirement: Final test artifact 必须具有独立 provenance 边界
训练 run 的最终 test 结果 MUST 写入独立 `final_test_metrics.json` 或等价明确命名 artifact。该 artifact MUST 记录实际 checkpoint path/role/digest/source、selection policy、evaluation split 和 model-selection split，并由 run index/status 只读引用。

#### Scenario: Final test artifact 完整发布
- **WHEN** final test 成功完成
- **THEN** 系统 MUST 在全部 provenance 字段就绪后原子发布 final-test artifact
- **AND** `train_log.json`、final config 和 run status MUST 引用同一实际 selected checkpoint

#### Scenario: 保留 validation metrics artifact
- **WHEN** training run 同时存在 validation metrics 和 final test metrics
- **THEN** 两类 artifact MUST 有不同文件边界和 split 标签
- **AND** final test MUST 不覆盖或重命名 validation `metrics.json` 来冒充独立结果

