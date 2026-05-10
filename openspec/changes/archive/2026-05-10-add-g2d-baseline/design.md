## Context

项目当前已经完成 future-only 标签收敛：`prepare_labels()` 返回 `labels[:, :num_pred]`，`select_prediction_slots()` 以 `num_pred` 作为 horizon，CRAF/MARF 也直接输出 `num_pred` 个 prediction slot。默认 `num_pred=3`，对应 `t+1/t+2/t+3`，不再包含历史最后 beam 或 `num_pred + 1` 语义。

现有训练 loop 只支持一个 teacher model，与 no-KD、logits KD、RKD 的扁平 `[B*H, C]` 蒸馏接口绑定。G2D 需要同时加载多个单模态 teacher，对 fusion student 执行 supervised CE、feature KD、logit KD，并基于 teacher confidence 做 SMP 梯度屏蔽和诊断，因此必须新增一条结构化训练路径。

实际代码包名是 `src/kd_sensing/distillation/`，而不是实验文档中建议的 `distillers/` 与 `losses/` 顶层目录。本设计使用现有包和 registry 边界，以减少迁移成本。

## Goals / Non-Goals

**Goals:**

- 实现 `distillation.type: g2d`，支持 `lite`、`global`、`horizon_diagnostic` 三种 mode。
- 严格保持 logits `[B, 3, 64]` 与 labels `[B, 3]` 契约，任何 G2D teacher 输出 horizon 不等于 `num_pred` 都必须报错。
- 支持五个单模态 teacher：`image`、`radar`、`gps`、`lidar`、`mmwave`。
- 为 G2D 输出 per-horizon teacher confidence、weak-to-strong ranking、active modalities、loss breakdown 和 horizon-wise top-k 指标。
- 保持 CRAF/MARF、legacy fusion 和现有 KD 训练行为不变。

**Non-Goals:**

- 本变更不重新定义 MARF 主方法，也不把 G2D SMP 与 MARF router 合并。
- `g2d-horizon` 先只做 horizon-wise 诊断，不实现 per-horizon backward 或 per-horizon 梯度屏蔽。
- 不兼容输出 `[B, 4, 64]` 的旧 checkpoint；这类 teacher 需要重新训练或显式修正。

## Decisions

### 1. 使用现有 `kd_sensing.distillation` 包承载 G2D

新增模块放在：

- `src/kd_sensing/distillation/g2d.py`
- `src/kd_sensing/distillation/teacher_ensemble.py`
- `src/kd_sensing/distillation/g2d_smp.py`
- `src/kd_sensing/diagnostics/g2d_diagnostics.py`

`registries.import_default_components()` 继续导入 `kd_sensing.distillation.distillers`，该模块再注册或导入 G2D 组件。这样 `build_distiller()` 不需要新增 registry 类型。

备选方案是新增 `src/kd_sensing/distillers/` 和 `src/kd_sensing/losses/` 包，但这会绕开当前 `DISTILLERS`、`LOSSES` 和配置加载约定，增加重复抽象。

### 2. G2D 走结构化训练路径，旧 KD 路径不变

`G2DDistiller` 注册为 `DISTILLERS["g2d"]`，但它不复用当前旧式 `forward(student_logits, teacher_logits, targets, ...)` 签名，而是提供结构化方法：

```python
compute(
    student_output: ModelOutput,
    teacher_outputs: dict[str, ModelOutput],
    labels: torch.Tensor,
    *,
    epoch: int,
) -> G2DStepResult
```

训练 loop 在 `distillation.type == "g2d"` 时：

1. 只构建 fusion student 作为主模型。
2. 构建 `TeacherEnsemble`，而不是单个 `model.teacher`。
3. student 前向后保留 `[B, H, C]` logits 与 features。
4. teacher ensemble 在 `torch.no_grad()` 中返回每个模态的 `ModelOutput`。
5. `G2DDistiller.compute()` 返回 total loss、supervised loss、feature KD、logit KD、teacher confidence 和 ranking。

旧 no-KD、logits KD、RKD 继续使用现有扁平调用，避免改动已经稳定的路径。

### 3. Teacher 配置使用 `distillation.g2d.teachers`

G2D 配置以当前项目字段为准：

```yaml
distillation:
  type: g2d
  g2d:
    mode: lite
    teachers:
      image:
        model:
          type: image_teacher
        checkpoint: null
        strict_load: true
```

`checkpoint: null` 表示尝试从当前场景 best checkpoint registry 解析该单模态 teacher；解析失败时必须报错。显式路径优先于 registry。所有 teacher 加载后必须 `eval()` 且 `requires_grad=False`。

### 4. Feature KD 以“可用特征优先、自动投影”为原则

teacher feature 从 `ModelOutput.input_features`、`output_features` 或 diagnostics 中的 feature 字段提取。student 侧优先使用 per-modality features：

- CRAF/MARF：从 `token_features: [B,K,T,D]` 按 `modalities` 拆分。
- legacy `fusion_student`：新增可选 `modality_features` diagnostics，保存各分支进入 fusion layer 前的 `[B,T,D_m]` embedding。
- 如果某个模态缺少可用 feature，G2D feature KD 必须报错，除非配置显式关闭 `feature_weight` 或该模态 feature KD。

维度不一致时由 G2D loss 按模态使用 lazy/auto projection，将 student feature 投影到 teacher feature 维度。projection 属于 student 训练参数，随 optimizer 更新。

### 5. Logit KD 与 supervised CE 都按 `[B,H,C]` 计算

G2D loss 先检查：

- student logits ndim 为 3。
- teacher logits ndim 为 3。
- `H == num_pred`。
- labels shape 为 `[B, H]`。

supervised CE 使用全部 horizon 默认展平为 `[B*H,C]`。logit KD 对每个 teacher 的 `[B,H,C]` soft target 计算 KL 后按模态平均。默认 `horizons: all` 等价于 `[0,1,2]`。

### 6. SMP 在 backward 后屏蔽 inactive modality encoder 梯度

`g2d-global` 根据每个模态三步 teacher confidence 平均值做弱到强排序。SMP scheduler 按 `per_modality_tau` 逐段只激活一个模态，最后激活全部模态。梯度处理放在 backward 之后、optimizer step 之前：

- inactive modality encoder/projector 梯度清零。
- fusion layer、GRU、transformer、prediction head、G2D projection 参数保持梯度。
- AMP 启用时先 `unscale_`，再执行 SMP mask 与 grad clip。

参数定位同时支持 `model.encoders.<modality>` 风格和 legacy `fusion_student` 的 `image_cnn_layers`、`radar_cnn_layers`、`gps_projection`、`lidar_cnn_layers`、`mmwave_projection` 命名。

### 7. Diagnostics 采用 epoch 聚合 JSON

每个 epoch 保存：

```text
outputs/<scene>/<run_name>/diagnostics/g2d_epoch_<epoch>.json
```

诊断 accumulator 在 batch 级聚合 teacher confidence、student branch confidence、loss breakdown、ranking 和 active modalities。`g2d-horizon` 额外保存 `t+1/t+2/t+3` 各自的 weak-to-strong ranking，但训练调制仍使用三步平均 confidence。

### 8. Metrics 保留现有数组并增加扁平字段

当前 `metrics.json` 已保存 `topk` 数组和 `dba` 数组。G2D 需要更方便的汇总字段，因此 validator 增加：

- `val_top1_t1`、`val_top1_t2`、`val_top1_t3`、`val_top1_avg`
- `val_top3_t1`、`val_top3_t2`、`val_top3_t3`、`val_top3_avg`
- `val_top5_t1`、`val_top5_t2`、`val_top5_t3`、`val_top5_avg`

这些字段不替代现有数组，避免破坏已有读取逻辑。项目目前没有输出 `top1_h0`、`beam8_acc` 等旧字段，G2D 实现不得新增这些字段。

## Risks / Trade-offs

- [Risk] 多 teacher 前向会显著增加训练时间和显存占用 → Mitigation: teacher 全部 `no_grad()`，默认 AMP 兼容，诊断按 epoch 聚合，先以 Scene9 smoke test 验证吞吐。
- [Risk] legacy `fusion_student` 当前没有 per-modality feature 输出 → Mitigation: 仅新增 diagnostics 字段，不改变 logits 和旧 tuple 输出契约；用测试覆盖 feature dict 解析。
- [Risk] teacher registry 中可能没有某个模态 checkpoint → Mitigation: G2D 启动时 fail fast，错误包含模态名、解析路径和 registry 信息。
- [Risk] SMP 梯度屏蔽可能误伤 fusion/head 参数 → Mitigation: 使用白名单式 modality encoder 参数定位测试，显式检查 fusion/head 梯度保留。
- [Risk] 旧 checkpoint 输出 `[B,4,64]` → Mitigation: G2D 不调用 `select_prediction_slots()` 静默截断 teacher logits，而是按 `num_pred` 严格校验并报错。

## Migration Plan

1. 新增 G2D distillation、teacher ensemble、SMP、diagnostics 模块和单元测试。
2. 扩展 `trainer.py`，仅在 `distillation.type: g2d` 时走 G2D 分支。
3. 扩展 `fusion_student` diagnostics，暴露 per-modality branch features。
4. 新增五模态 G2D 三个配置文件和结果汇总脚本。
5. 运行定向测试：`conda run -n kd_mm_beam pytest -q tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py`。
6. 运行全量测试：`conda run -n kd_mm_beam pytest -q`。
7. 使用 Scene9 跑 `g2d-lite` 至少 1 epoch，再跑 `g2d-global` 验证 active modalities 日志。

## Open Questions

- 单模态 teacher checkpoint registry 是否已经覆盖 Scene9 和 Scene32 的五个模态。如果没有，G2D 配置需要先显式填写 checkpoint 路径。
- G2D 默认 student 使用 legacy `fusion_student` 还是 token transformer fusion。实现优先兼容 legacy `fusion_student`，但配置可以在后续实验中切换为 `token_transformer_fusion`。
