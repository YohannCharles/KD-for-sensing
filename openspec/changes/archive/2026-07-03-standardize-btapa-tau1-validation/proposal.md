## Why

BTAPA tau1 在本轮 Scene31 缺失模态实验中显示出更好的 missing robustness 和 radar-only 表现，但旧 V3、当前 proto baseline 与 BTAPA tau1 的 only-radar 指标差异过大，存在 checkpoint、missing pattern、best 选择或评估入口不一致的风险。现在需要先统一评估口径并补齐 tau1 多 seed 与短 epoch 验证，避免在不稳定证据上继续扩展主线。

## What Changes

- 统一四模态缺失 pattern 的标准命名、模态顺序、alias 和 mask 构造，保留 `missing_gps` 与 `non_gps_only` 两个统计名称。
- 新增 apples-to-apples 复评脚本，用同一套 checkpoint load、missing pattern 和指标计算复评旧 V3、proto baseline 与 BTAPA tau1，并输出 CSV/Markdown/delta/manifest。
- 新增 BTAPA tau1 seed2/seed3 配置和 seed mean±std 分析脚本，不覆盖已有 tau1 结果。
- 新增 BTAPA tau1 es20 配置族，必要时以现有训练 runtime 的轻量 early stopping 能力记录早停状态。
- 修正 missing run summary 对 completed、completed_early_stopped、incomplete_has_checkpoint 和 killed_or_failed 的状态识别。
- 新增只跑关键 BTAPA tau1 验证的串行 launcher；增强 BTAPA 分析脚本的 candidate main 和 paper-ready observation 输出。
- 补强 proto vs BTAPA tau1 复核：统一 checkpoint resolver、debug eval consistency 报告、ordinary proto 三 seed、8 卡并行 launcher、fresh apples-to-apples eval 和 proto-vs-BTAPA mean±std 分析。
- 不扩展 RBMA、JEPA、KD、fullaux，也不把 tau4、ADBA、modw1、fusiononly 作为新主线。

## Capabilities

### New Capabilities

无。该变更收敛现有 BTAPA、模态契约、训练评估 runtime 和实验 workflow，不新增独立 capability。

### Modified Capabilities

- `modality-contracts`: 增加四模态 missing pattern 的统一构造、标准顺序、alias 和 pattern 名称契约。
- `training-evaluation-runtime`: 增加 apples-to-apples 复评、checkpoint 选择、early-stopped run metadata 和 summary 状态识别契约。
- `beam-topology-prototype-alignment`: 增加 BTAPA tau1 candidate main、多 seed 分析、es20 验证和只读分析报告契约。
- `experiment-workflow`: 增加关键 BTAPA tau1 验证 launcher、seed 配置和不覆盖已有运行产物的 workflow 契约。

## Impact

- 影响 `src/kd_sensing/eval/` 或等价评估 helper、训练 runtime、BTAPA/summary 分析脚本、Scene31 BTAPA 配置与 shell launcher。
- 新增或修改 `scripts/` 中的本地手工分析/验证入口，不新增 package console script，不恢复旧根训练入口。
- 新配置和脚本默认写入 ignored `outputs/scene31/analysis/`、`outputs/scene31/<run>` 或日志目录，不提交 checkpoint、训练输出、cache 或真实数据。
- 验证命令使用 `conda run -n kd_mm_beam ...`；长期训练命令仅作为 launcher 输出或用户显式运行，不在实现验证中启动完整训练。
