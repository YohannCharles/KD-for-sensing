## Why

当前 Scene31 主线 V3 依赖 strong encoder、reliability weighted-sum、missing mask 和 beam prototype alignment，但现有 prototype target 仍偏普通平滑分类。BTAPA 将 beam index 邻接关系显式纳入 prototype 对齐，用最小增量验证 beam-neighborhood semantic space 是否提升缺失模态鲁棒性。

## What Changes

- 在现有 beam prototype alignment 上新增 BTAPA soft target、fusion/modality prototype loss 权重和可选 ADBA-aware auxiliary loss。
- 新增 `configs/scene31/main_v3_strong_reliability_btapa*.yaml` 消融配置，保留旧 `main_v3_strong_reliability_proto` baseline 不变。
- 训练 diagnostics/metrics 中记录 `beam_ce_loss`、`proto_loss`、`btapa_fusion_loss`、`btapa_modality_loss`、`adba_proto_loss` 和 `total_loss`。
- 新增 BTAPA smoke test、串行 launcher 和只读分析脚本，用于比较旧 V3 与 BTAPA 消融。
- 不启用 RBMA、JEPA、KD、full auxiliary loss，也不新增旧 root 训练入口。

## Capabilities

### New Capabilities

- `beam-topology-prototype-alignment`: BTAPA prototype soft target、loss 组合、ADBA-aware auxiliary loss、Scene31 消融配置、日志和分析边界。

### Modified Capabilities

- `rbma-prototype-kd-missing-workflow`: 将现有 beam prototype alignment 扩展为可配置 target 类型和 BTAPA diagnostics，但不改变 RBMA/KD 默认边界。
- `experiment-workflow`: 允许 Scene31 local/manual BTAPA ablation 配置、launcher 和分析产物作为本地实验 workflow。

## Impact

- 主要代码影响 `src/kd_sensing/losses/beam_prototype_alignment.py` 与 `src/kd_sensing/losses/u_mask_beam_jepa.py`。
- 新增 `configs/scene31/` BTAPA overlay、`scripts/run_btapa_experiments.sh`、`scripts/analyze_btapa_runs.py` 和 `scripts/smoke_test_btapa.py`。
- 不新增依赖，不读取真实 `dataset/`，新输出仍限定在 ignored `outputs/scene31/` 与 `logs/`。
