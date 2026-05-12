## Context

当前项目的第一阶段架构治理已经有效：`src/kd_sensing` 包结构清晰，registry 和模态契约集中，builder、transform、轻量导入边界都有 spec 和测试约束。新的问题发生在第二阶段增长路径上：G2D、CRAF、MARF、Phase 1.5、互补分析和 Gradio viewer 都是合理功能，但它们通过训练主循环、诊断 core 和配置矩阵的方式接入，导致单个文件持续变大、依赖方向变模糊。

现状可以概括为：

```text
configs/*.yaml
    │
    ▼
engine.trainer.train()
    ├─ 普通 KD / no-KD
    ├─ G2D teacher ensemble / SMP / diagnostics
    ├─ CRAF beam soft / unimodal / counterfactual / reliability KD
    ├─ MARF residual / prior / subset training
    ├─ validation metrics aggregation
    ├─ TensorBoard / JSON / checkpoint / registry
    └─ final_config / teacher_metrics / curves
```

这个结构的风险不是单个算法文件太长，而是所有算法都要穿过同一个编排函数。继续按此方式增加方法，会让每个功能都扩大训练主循环、验证路径和配置复制。

## Goals / Non-Goals

**Goals:**

- 把训练方法接入收敛到明确扩展点，使新增 G2D/CRAF/MARF 类似功能时主要新增方法模块和测试，而不是继续扩写 `trainer.py`。
- 统一 batch preparation + task forward，避免 trainer、validator、viewer predictions、teacher ensemble 分别维护相同任务分支。
- 把 G2D runtime 从 distillation 算法层移出，让 `distillation/g2d.py` 保持 loss、confidence、feature/logit 对齐职责。
- 将 CRAF/MARF 训练期 extra loss 和 scalar diagnostics 从 `trainer.py` 移到职责模块，并保持现有指标键和训练行为。
- 让 `diagnostics/visualization` 的子模块承载真实实现，`core.py` 不再作为主要实现聚合文件。
- 为高级 fusion 方法提供 overlay 配置路径，减少 CRAF/MARF/G2D 和 ablation YAML 复制。
- 增加架构边界测试，防止新代码回到大文件聚合模式。

**Non-Goals:**

- 不改变模型结构、forward 输出契约、state_dict key 或 checkpoint 加载语义。
- 不改变 G2D、CRAF、MARF 的 loss 数值定义、默认权重、diagnostics 字段名或训练结果解释。
- 不删除现有公开 facade、CLI 或实体 YAML；本变更允许兼容入口继续存在。
- 不把所有诊断、viewer 或互补分析一次性重写成新框架。
- 不引入 Hydra、Lightning、Accelerate 等外部训练框架；当前目标是控制本项目内部增长。

## Decisions

### 1. 训练主循环采用“生命周期编排 + 扩展点”

`engine.trainer.train()` 保留为唯一公开训练入口，但内部不再直接包含每种方法的细节。新增一个轻量扩展接口，例如：

```text
TrainingExtension
  setup(context) -> ExtensionState
  before_epoch(context, state) -> None
  after_forward(context, state, batch_state) -> LossBundle
  after_backward(context, state, batch_state) -> None
  after_epoch(context, state, epoch_state) -> dict
```

实际命名可在实现时调整，但职责必须稳定：

- G2D extension 负责 teacher ensemble runtime、distiller.compute、SMP active modalities 和 G2D epoch diagnostics。
- CRAF extension 负责 beam soft、unimodal aux、counterfactual gate、prior regularization、reliability KD 和 CRAF scalar diagnostics。
- MARF extension 负责 residual norm、anchor prior、anchor entropy、subset training 和 MARF scalar diagnostics。
- baseline/no-KD/KD 公共逻辑仍由 core loop 处理，扩展点只添加方法特有行为。

替代方案是把 trainer 拆成多个 trainer class，例如 `G2DTrainer`、`CRAFTrainer`、`MARFTrainer`。这会复制 checkpoint、validation、history、AMP、scheduler 和 logging 逻辑，短期代码量更大。扩展点方案保留一个训练生命周期，减少横切重复。

### 2. 使用共享 forward runtime 替代多处任务分支

新增或扩展 `engine` 中的 runtime helper，例如 `prepare_task_inputs()`、`forward_task_model()`、`run_model_step()`，统一完成：

- `normalize_batch`
- labels preparation
- 按 task/modalities 准备 image/radar/gps/lidar/mmwave 输入
- `force_modality_mask`、`force_reliability_gate`、`gate_temperature` 透传
- `adapt_model_output`
- `select_prediction_slots`

trainer、validator、G2D teacher runtime、viewer predictions 和 counterfactual forward 都调用同一 helper。这样新增模态或修改 task forward 只需要改一个 runtime 模块和对应测试。

替代方案是只把 `_forward_for_task()` 从 trainer 移到 `engine.batch`。这能减少重复，但仍没有表达“完整 step”的边界，validator 和 viewer predictions 仍可能继续拼装自己的分支。

### 3. Distillation 算法层不负责构建 runtime 对象

`distillation/g2d.py` 保持算法职责：校验 logits/labels、计算 supervised/logit KD/feature KD、teacher confidence、ranking、SMP active modalities。G2D teacher ensemble 的模型构建、checkpoint 解析、batch preparation 和 device 处理迁移到 `engine` 下的方法 runtime 模块。

`distillation` 可以依赖轻量张量契约，例如 `ModelOutput`；但不得依赖 `engine.optim.build_model`、checkpoint registry、dataset builder 或 batch preparation。若 `ModelOutput` 继续放在 `engine.model_output` 会形成轻量依赖，短期可接受；更干净的后续方案是移到 `kd_sensing.contracts` 或 `kd_sensing.runtime_types`。

替代方案是保留 `distillation/teacher_ensemble.py` 原地，只在文件名上说明 runtime。这样不会解决依赖方向问题：算法层仍会构建模型和加载 checkpoint。

### 4. CRAF/MARF extra loss 模块化，以行为等价为验收

CRAF/MARF 的 extra loss 不是普通 loss registry 能完全表达，因为它们需要访问 model diagnostics、额外 forward、modality masks、epoch schedule 和 scalar diagnostics。因此不强行塞进现有 `LOSSES` registry，而是放进训练扩展模块。

迁移时以现有测试和新等价测试为准：

- 相同输入下 extra loss key 集合保持不变。
- `epoch_log` 和 TensorBoard 使用的 scalar key 保持不变。
- subset training、counterfactual、reliability KD 的启用条件和错误信息保持兼容。

替代方案是把所有 extra loss 函数移到 `distillation/craf_losses.py`。这会把训练 forward、mask sampler 和 diagnostics 聚合放入 distillation，反而扩大算法层职责。

### 5. 诊断可视化做真实拆分，不只 re-export

`diagnostics/visualization/core.py` 应收敛为入口编排或兼容 facade。主要实现移动到已有子模块：

- `config.py`: `VisualizationConfig`、parse/final config snapshot
- `datasets.py`: diagnostic dataset 构建、scene metadata、CSV frame 选择
- `sampling.py`: candidate collection、sample selection、sampling summary
- `stats.py`: tensor/modality/split statistics
- `render.py`: PNG/render record 构建
- `writers.py`: JSON/JSONL/CSV/final output paths

替代方案是只保留现状并靠测试约束行为。这不能解决维护问题，因为开发者仍会自然编辑 `core.py`。

### 6. 高级 fusion 配置使用 overlay 组合，实体 YAML 继续兼容

对 CRAF/MARF/G2D 等高级 fusion 方法，新增配置组合规则：

```text
base:        scene-neutral five-modality or selected-modality fusion base
method:      craf / marf / g2d method overlay
ablation:    optional no-prior / no-residual / subset-training / horizon/global/lite overlay
scene:       dataset.scene override
```

实体 YAML 可以继续存在并优先于生成配置，但新推荐路径应尽量使用 generator/overlay，避免复制 100 行以上配置。最终 `final_config.yaml` 仍写出完整解析结果，保证实验可复现。

替代方案是继续手写每个高级实验 YAML。短期直观，但每个 ablation 都复制 data/model/training/output 字段，容易出现 scene、run_name、teacher registry 和 loss schedule 漂移。

## Risks / Trade-offs

- [Risk] 训练扩展接口设计过大，变成新的复杂抽象层。  
  Mitigation: 只暴露当前 G2D/CRAF/MARF 已经需要的生命周期点；不预留未使用的 hook；用 tests 约束行为等价。

- [Risk] 迁移 trainer 过程中改变 loss 标量或日志字段。  
  Mitigation: 先增加针对现有 key 的 characterization tests，再移动实现；每个方法单独迁移和验证。

- [Risk] 共享 forward runtime 抽象不当，影响 viewer predictions 或 validation。  
  Mitigation: 从现有 `_forward_for_task` 和 validator 分支提取最小公共行为，保留窄参数；运行训练 IO、G2D、viewer prediction 相关测试。

- [Risk] overlay 配置生成器让调试时看不到完整配置。  
  Mitigation: `load_config()` 和训练入口继续 dump 完整 `final_config.yaml`；测试覆盖实体 YAML 优先级和 overlay 解析结果。

- [Risk] 诊断可视化拆分造成循环导入。  
  Mitigation: 子模块按 config -> datasets -> sampling/stats/render/writers 单向依赖，公开入口只从 facade 导出。

## Migration Plan

1. 新增 characterization tests，锁定当前 trainer history keys、epoch_log scalar keys、G2D diagnostics 输出、CRAF/MARF extra loss 启用条件和 validator forward 行为。
2. 提取共享 forward runtime，并将 trainer、validator、counterfactual forward、viewer predictions、G2D teacher runtime 逐步切过去。
3. 新增训练扩展接口和 extension context/loss bundle 数据结构，先接入 no-op extension 验证训练主循环行为不变。
4. 迁移 G2D runtime：teacher ensemble 构建和 checkpoint 解析进入 engine extension，`distillation/g2d.py` 保持算法层。
5. 迁移 CRAF extra loss，再迁移 MARF extra loss；每步保留原测试通过。
6. 拆分 diagnostics visualization core，让子模块承载真实实现并保持公开入口兼容。
7. 实现高级 fusion overlay 配置解析或生成器，保留现有实体 YAML 兼容和优先级。
8. 增加架构边界测试，限制 trainer 主循环和 diagnostics core 重新膨胀。
9. 使用分层测试和全量回归验收。

## Open Questions

- `ModelOutput` 是否在本变更中移动到 `kd_sensing.contracts`，还是先保留在 `engine.model_output` 作为轻量契约？
- 高级 fusion overlay 的用户入口采用缺失路径虚拟生成、显式 `inherits` 字段，还是单独的 recipe 文件？建议先复用当前 canonical virtual config 机制，降低 CLI 变化。
- 是否设置硬性文件行数上限？建议第一版不设硬阈值，只用“禁止方法细节新增到 trainer/core”测试约束增长路径。
