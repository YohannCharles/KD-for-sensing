## Why

`scene31_baseline_pack_lmdb` 的 fresh eval 暴露出 AMR-lite / AMBER-lite 在 `full`、`missing_gps`、`radar_only`、`lidar_only` 等 pattern 上指标完全相同，说明 `missing_mask` 被 batch forward 签名过滤后没有真正进入 modular model。与此同时，可信 proto 结果显示 `proto_randomdrop_subset_es40` 已超过旧的 `proto_sampler_uniform_es40`，需要把主线 reference、后续 reliability fusion 与 PatternFiLM 组合都切到 randomdrop subset 口径。

## What Changes

- 修复 `ModularSequenceModel.forward` 的缺失模态参数契约，让 fresh eval 传入的 `missing_mask`、`missing_modality_metadata`、`available_modalities` 与 `modality_mask` 不再被过滤，并在 fusion 前真实屏蔽缺失模态。
- 新增 AMR/AMBER-lite missing-mask 诊断和 mask-suspect 检查，只重跑已有 best checkpoint 的 fresh eval，不重训旧 run，不覆盖旧 checkpoint。
- 将 `proto_randomdrop_subset_es40` 设为当前可信 proto reference，`proto_sampler_uniform_es40` 降级为 ablation；summary 的 delta、ranking 和 promotion 逻辑都基于 subset reference。
- 审计并复用或补齐最小 reliability mask weighted fusion，使其兼容 proto prediction 与 randomdrop subset 训练，输出 reliability weight 日志。
- 新增 Scene31 subset reliability local/manual runner、reliability fusion 配置组合、randomdrop subset + PatternFiLM d8 配置组合和 combined summary；不默认继续 JTT/MVFR/MPDRO/beamsoft/condBTAPA/weakKD。

## Capabilities

### New Capabilities
无。

### Modified Capabilities
- `modular-sequence-model`: 明确 modular model 必须接收并消费 missing-mask 相关 forward 参数，缺失模态在进入 fusion/core 前必须被屏蔽。
- `local-missing-modality-baselines`: AMR-lite / AMBER-lite fresh eval 必须检测 mask 是否生效，mask_suspect 结果不得进入正式 winner ranking。
- `scene31-next-round-experiment-workflow`: 新增 subset reference、subset reliability/PatternFiLM local/manual workflow、runner、summary、ranking 和 promotion 规则。
- `observability-aware-fusion`: 补充 proto-compatible reliability mask weighted fusion 的最小契约、缺失模态 zero weight 和日志要求。

## Impact

- 影响源码：`src/kd_sensing/models/modular.py`、`src/kd_sensing/engine/batch.py` 或现有 missing-mask runtime 相关 helper、必要的 reliability fusion 组件。
- 影响脚本与配置：新增 `scripts/diagnose_modular_missing_mask.py`、`scripts/run_scene31_baseline_pack_maskfix_eval.sh`、`scripts/run_scene31_subset_reliability.sh`、`scripts/summarize_scene31_subset_reference.py`、`scripts/summarize_scene31_subset_reliability.py`，以及必要的 Scene31 subset reliability 配置/生成逻辑。
- 影响测试：新增 synthetic mask-forward、诊断/summary fixture、runner dry-run 或 shell/static focused tests；真实训练和 fresh eval 产物仍写入 ignored `outputs/`。
- 不新增 package CLI，不恢复旧入口，不删除或覆盖历史 checkpoint、训练输出或旧 summary。
