## Context

新主线只包含两个可分离创新：训练期的 beam-topology prototype supervision，以及推理期的 TBCP-7 finite probing。sensing 模型不读取 CSI/channel/beam power；probing policy 不更新模型。

## Architecture

```text
image/radar/gps/lidar
        -> independent encoders
        -> one shared temporal transformer
        -> one shared 64-beam prototype bank
        -> per-modality probabilities
        -> availability-masked arithmetic mean = p_sense[64]
        -> stateless circular posterior statistics
        -> TBCP-7 requested-measurement policy
```

## Decisions

### 1. 单阶段、严格四模态

registry id 固定为 `four_modal_topology_predictor`。模型只接受 canonical 四模态和 temporal/availability masks；不存在 CSI 参数、risk/evidence/fusion 参数、`training_stage` 或 `fusion_mode`。任一旧字段或旧 model id 必须在 config load 时失败。

### 2. 无参数 posterior fusion

每个可用模态通过同一个 prototype bank 产生 64 类 probability。`p_sense` 是可用模态 probability 的 arithmetic mean；缺失模态严格为零，Single mask 退化为对应单模态 probability。模型不学习模态权重。

### 3. 唯一训练 loss

单次训练同时优化 fused hard CE、availability-normalized unimodal hard CE、可选环形 soft CE 与 prototype alignment。topology-off 只将 soft/prototype 项置零；模型、数据、预算、mask schedule 与 seed 其余完全一致。

### 4. 原生 15-mask evidence

evaluator 对同一 validation-best checkpoint 直接遍历四模态 15 个非空 availability mask并保存 `fused_probability`。evidence schema 的 modalities 必须恰好为 `[image, radar, gps, lidar]`；五模态或 31-mask evidence 必须拒绝。

### 5. TBCP 保持独立

train-only topology likelihood、joint relative-dB update、expected-terminal-gain acquisition、K=7、requested-only simulator、batch schedules、synthetic measurement-error 与 defensive ablations 保持既有数学定义。likelihood artifact 不进入模型 state dict，也不构成训练 stage。

### 6. 断代删除

删除旧源码、tracked templates、tests、docs 和确认清单中的 ignored outputs；不建立 alias、migration loader、旧 checkpoint converter 或 archive copy。保留正式 MMW split、train-only likelihood、ULA-DFT topology audit 和通用四模态数据缓存。

## Training and Ablation

- topology `{off,on}` × train seed `{1,2,3}`，共六个 fresh single-stage runs。
- 相同 `mmw_id_stratified_block_v1` seed 0、epochs、batch、workers、preprocessing、mask sampling 和 validation selection。
- 每个 checkpoint 生成原生 15-mask evidence，再运行 Direct / Posterior Top-7 / TBCP-7 2×3 nested ablation。
- 本 change 只清理和建立可运行契约，不启动上述长训练。

## Risks

- 删除 ignored outputs 不可恢复：执行前使用显式路径清单，保留 split/topology/calibration。
- 大范围删除可能误伤通用 trainer：删除前以引用搜索确认 owner，删除后运行 full pytest 与 compile。
- 旧 checkpoint 全部失效是预期行为，不通过放宽 strict load 解决。
