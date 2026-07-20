## Why

现有 T2 Router 在联合缺失/退化压力下虽然优于 Uniform，但不能稳定超过同 checkpoint 的训练集静态模态先验；已保存 trace 表明其样本级权重变化与单模态通信效用变化几乎不相关。根因是 5 帧质量证据在时序池化前丢失，现有 Router 主要记住模态身份而不能识别当前样本中的局部退化，因此需要将动态可靠性融合从小型权重预测器升级为可验证的第二个核心方法。

## What Changes

- 在 T2 内新增共享的原型可靠性路由组件，并以组合配置提供四个 inner-only 候选：PATR、H2R、CoRe 和 Unified-HPR；候选不是新的主模型，也不复制训练循环。
- PATR 从逐帧 beam prototype 分布提取拓扑与时序质量证据，并在仅由训练集拟合的静态模态先验上学习有界动态残差。
- H2R 在时序池化前估计 `T×M` 时间块健康度，再将帧级健康门控与模态级路由权重分层组合，避免坏帧先污染模态表示。
- CoRe 以 leave-one-out 跨模态 prototype 共识计算分歧证据，在不读取退化类型或严重度的前提下识别可用但不可靠的模态。
- 新增配对反事实联合退化训练：相同样本、相同 availability/Drop mask 形成未退化与退化视图，退化状态仅用于数据生成和训练审计，不作为模型输入。
- 支持标签拓扑效用和 normalized beam-power 效用两种互斥 Router 监督；两者共享数据、checkpoint、优化预算和评估协议。
- 新增 8 卡 seed1 开发筛选 launcher、不可变 resolved config/manifest、训练日志和状态记录；PATR/H2R/CoRe/Unified-HPR 各运行两种监督版本。
- 预注册 Dynamic Router 相对 Uniform、train-fit Global Prior、Current Router 和 Oracle 的机制 Gate，并保护 Clean 与纯 Drop 条件；筛选结果在后续多 seed 确认前不得升级为 canonical T2 或正式 outer claim。

## Capabilities

### New Capabilities

- `dynamic-prototype-reliability-routing`: 定义逐帧原型质量、先验锚定残差、分层时间门控、跨模态共识、配对反事实训练和 seed1 候选筛选的行为与证据边界。

### Modified Capabilities

- `u-mask-beam-jepa`: 扩展 T2 supervised Router，使其可在池化前消费逐帧原型证据，并保持现有 T2/S1 默认路径和输出契约不变。
- `training-evaluation-runtime`: 增加配对联合退化训练、训练集先验拟合以及候选筛选 provenance 的传播与 fail-closed 要求。

## Impact

- 主要影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/losses/u_mask_beam_jepa*.py`、现有 MMW corruption helper、新增筛选 launcher/config 生成逻辑及聚焦测试。
- 不新增第三方依赖，不修改 `dataset/`、历史 checkpoint、正式 evidence 结果或现有 canonical T2/S1/baseline recipe。
- 新生成的配对 mask cache、resolved config、checkpoint、日志、评估 trace 和汇总全部写入 ignored `outputs/`。
