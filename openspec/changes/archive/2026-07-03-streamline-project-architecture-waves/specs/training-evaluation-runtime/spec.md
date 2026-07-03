## ADDED Requirements

### Requirement: Training runtime is organized into auditable phases
训练 runtime MUST 将 `cfg` 到 run 资源的构建、训练循环、checkpoint、validation、final evaluation、artifact 写出和 shutdown/finalization 拆成可审计 phases。Public `train(cfg)` 行为 MUST 保持兼容，但内部 MUST 使用 run context 或等价结构表达共享状态，避免 `_train_inner` 继续吸收新 workflow 逻辑。

#### Scenario: Run context preserves behavior
- **WHEN** training wave 引入 `TrainingRunContext` 或等价结构
- **THEN** run directory、status file、artifact writer、dataloaders、normalization artifacts、device/model/optimizer/scheduler/scaler、checkpoint manager、TensorBoard writer、extension 和 early stopping state MUST 可从 context 追踪
- **AND** `train_log.json`、`final_config.yaml`、checkpoint layout 和 runtime metadata MUST 保持兼容

#### Scenario: 新训练扩展不修改主循环
- **WHEN** 新增 current training extension、auxiliary loss、metadata handoff 或 final evaluation 行为
- **THEN** 实现 MUST 优先落在 extension、phase helper、runtime metadata helper 或 evaluation owner
- **AND** 不得向 `_train_inner` 添加 suite-specific 大段私有 helper

### Requirement: Evaluation pass is split by schema responsibility
共享 evaluation pass MUST 将 batch iteration、difficulty application、model step、objective label preparation、output recording、metadata recording、metric aggregation 和 prediction artifact schema 拆为职责明确的 helper。拆分 MUST 不改变 validator、evaluator、diagnostics real-forward 和 final-test evaluation 的 public output schema。

#### Scenario: 评估输出 schema 兼容
- **WHEN** `run_evaluation_pass` 内部被拆分
- **THEN** validation metrics、prediction records、objective outputs、metadata rows、difficulty replay metadata 和 diagnostics payload MUST 与变更前兼容
- **AND** `validator.validate`、`evaluator.evaluate` 和 diagnostics real-forward MUST 继续复用同一 shared evaluation pass

#### Scenario: 新 objective 不复制 evaluation loop
- **WHEN** 新增或修改 prediction objective、auxiliary target 或 metric
- **THEN** 实现 MUST 更新 objective metadata、batch labels、loss/metric helper 和 evaluation schema helper
- **AND** 不得新增模型或 objective 专属 validation loop 来绕开 shared evaluation pass

### Requirement: Runtime finalization remains failure-safe
训练和评估 runtime 拆分后 MUST 保持 failure status、dataloader shutdown、TensorBoard close、checkpoint finalization 和 artifact flush 的失败安全语义。异常路径 MUST 继续写出可定位的 failed status，且不得吞掉原始异常。

#### Scenario: 失败路径保持可诊断
- **WHEN** training 或 evaluation phase 抛出异常
- **THEN** runtime MUST 尝试写入 failed status 并关闭可关闭资源
- **AND** 原始异常 MUST 继续向调用方传播，不能被 cleanup/finalization 异常覆盖

