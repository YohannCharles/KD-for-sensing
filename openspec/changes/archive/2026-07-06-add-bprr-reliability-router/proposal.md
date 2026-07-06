## Why

上一轮 `outputs/pcpg_radar_balance_v1` 显示 `e5_low_encoder_lr` 综合最强、`e6_hard_subset_jepa` 重缺失最强，但普通 confidence / reliability gate 仍可能继续偏向 image/lidar。现在需要在不改变默认行为、不破坏旧脚本和旧结果的前提下，补齐 oracle 上界，并新增全局校准、pattern-aware、prototype-aware 的可靠性路由实验。

## What Changes

- 新增 BPRR（Beam-Prototype Reliability Router）作为显式 opt-in fusion 选项 `fusion_type: bprr` / `--fusion bprr`，第一版优先实现 logits 层融合、masked softmax、单模态 gate=1 和 gate diagnostics。
- 新增 raw confidence gate baseline `fusion_type: raw_conf_gate`，只基于 logits confidence/margin/entropy 和 available mask 生成 gate，用作普通 confidence gate 的反例基线。
- 扩展 eval-only oracle gate，补齐 `e3_oracle_gate_eval`，基于可用 unimodal branch 与 ground truth beam 的距离选择 oracle 分支，并输出 chosen modality distribution。
- 新增 BPRR temperature calibration、gate balance regularization 和 radar gate floor regularization，均默认关闭并只通过显式参数启用。
- 新增 7 组 local/manual 实验矩阵、8 GPU 并行 launcher、manifest/log 写出和只读 summary helper，输出限定在 `outputs/bprr_reliability_router_v1/`。
- 新增 focused tests 覆盖 BPRR masked softmax、temperature diagnostics、radar gate regularization、oracle gate、launcher dry-run 和 summary parser。
- 不新增 package console script、不复制训练框架、不提交真实训练输出、checkpoint、日志或 generated config。

## Capabilities

### New Capabilities

- `bprr-reliability-router`: 覆盖 raw confidence gate、BPRR fusion、calibration、gate regularization、oracle eval、7 组实验矩阵、local/manual launcher、summary helper 和 focused tests。

### Modified Capabilities

- 无；本 change 以新增 opt-in capability 表达，现有训练、评估、配置和脚本默认行为保持不变。

## Impact

- 主要影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/models/jepa.py` 或现有缺失模态 fusion helper、`src/kd_sensing/engine/` 的训练 extension/diagnostics 消费、评估 oracle 输出以及 `scripts/` 下本地实验 launcher/summary helper。
- 配置和 CLI 继续复用既有 `kd-sensing-train` / `kd-sensing-evaluate` 和 config override，不新增长期 public CLI。
- 新增脚本为 local/manual experiment surface，必须支持 dry-run 或只读 summary，并将日志、manifest 和汇总写入 ignored `outputs/bprr_reliability_router_v1/`。
- 不引入新的运行时依赖；所有项目 Python 验证仍通过 `conda run -n kd_mm_beam ...` 执行。
