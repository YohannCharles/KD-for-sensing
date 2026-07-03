## Why

最新 apples-to-apples 三 seed 结果显示，ordinary prototype baseline 整体更稳，BTAPA tau1 只在 radar_only / lidar_only 等弱单模态上有收益，不能继续宣称全局提升 missing robustness。现在需要一套短 epoch、early stopping、两 seed 的 night grid，系统筛选“选择性吸收 BTAPA 弱模态收益且不损伤关键 missing pattern”的候选。

## What Changes

- 基于 ordinary proto baseline 新增 pattern-conditional BTAPA，只对指定 missing pattern 使用 BTAPA soft beam target，其它样本回退 ordinary prototype target。
- 扩展训练时 missing pattern sampler，支持 uniform、weak/sensing/missing_gps oversample 和 easy/hard curriculum，并记录 epoch pattern 分布。
- 新增 hard pattern CE reweight，默认只加权 CE，不加权 prototype loss。
- 新增轻量 mask-conditioned adapter，在 fusion 后用 available mask FiLM 调制 fused feature。
- 新增 weak-pattern KD 和 lightweight latent prediction probe，均为 opt-in 探路辅助项，不作为默认主线。
- 新增 A-F 共 58 个 run 的配置生成脚本和 manifest，并加入 6 个 proto/BTAPA reference run，总计 64 个 run。
- 新增 8 卡 night-grid launcher、fresh eval、analysis 和 summary 兼容能力，所有训练进程单进程单 GPU，不默认启用 DDP。
- 不扩展 RBMA，不恢复复杂 JEPA，不把 KD/fullaux 作为主线，不覆盖已有 outputs/logs/checkpoints/eval。

## Capabilities

### New Capabilities

无。该变更扩展现有 Scene31 missing-modality 实验能力、BTAPA/prototype loss、训练 runtime、评估 workflow 和脚本入口，不新增独立长期 capability。

### Modified Capabilities

- `beam-topology-prototype-alignment`: 增加 pattern-conditional BTAPA 的 sample-wise loss 选择与 diagnostics 契约。
- `rbma-prototype-kd-missing-workflow`: 扩展 pattern-balanced missing mask sampler，用于本轮 non-RBMA proto night grid。
- `training-evaluation-runtime`: 增加 hard pattern CE reweight、mask adapter、weak-pattern KD、light latent prediction probe 与训练 metrics 记录契约。
- `experiment-workflow`: 增加 64-run night grid 配置生成、8 GPU launcher、fresh eval、analysis 和 summary 兼容契约。
- `modality-contracts`: 复用并要求统一 missing pattern API 覆盖 weak/sensing/single pattern 分类。

## Impact

- 影响 `src/kd_sensing/models/`、`src/kd_sensing/engine/`、`src/kd_sensing/eval/`、`src/kd_sensing/utils/` 下已有 loss、forward、mask 和 checkpoint helper。
- 新增 `configs/scene31/templates/main_v3_proto_es20_base.yaml`、`configs/scene31/night_grid/` 生成配置与 manifest。
- 新增或增强 `scripts/generate_experiment_grid.py`、`scripts/run_night_grid_8gpu.sh`、`scripts/eval_night_grid.py`、`scripts/analyze_night_grid.py`、`scripts/summarize_missing_runs.py`。
- 验证使用 `conda run -n kd_mm_beam ...`；实现阶段只运行 dry-run/help/focused tests，不启动 64 个真实训练。
