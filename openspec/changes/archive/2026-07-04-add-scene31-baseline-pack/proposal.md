## Why

当前 Scene31 缺失模态主线已经收敛到 pattern-balanced exposure / uniform missing-pattern sampler，但论文还缺少可复核的 baseline pack 来回答 reviewer 最可能追问的问题：uniform sampler 是否只是 random modality dropout，以及该训练暴露原则是否能迁移到其它轻量融合 backbone。

本 change 补齐本地可训练 baseline、统一 fresh eval、runner 和 summary，让后续结果能在同一套 miss1/miss2/miss3 评估口径下保守比较，而不继续扩展 PatternFiLM、JTT、MVFR、MPDRO 等已不推荐路线。

## What Changes

- 新增 Scene31 baseline pack local/manual workflow，默认输出到 `outputs/scene31_baseline_pack_lmdb`，不覆盖既有 next-round、BC、beamsoft weak、magic overnight 或 funnel 结果。
- 新增 random modality dropout 训练扰动，覆盖 Bernoulli keep-prob 和 random non-empty subset 两种策略，并记录实际训练暴露分布。
- 新增或补齐 AMR-lite、AMBER-lite 和可选 FeatureMod-lite baseline 的配置与轻量模型能力，复用现有训练和 fresh eval 管线。
- 新增 `scripts/run_scene31_baseline_pack.sh`，支持按 group 训练、复评、自动评估、跳过、覆盖和多 GPU 串行 worker 调度。
- 新增 `scripts/summarize_scene31_baseline_pack.py`，读取本轮 baseline pack、旧 uniform reference 和可用 proto baseline 结果，输出 per-run、mean/std、delta、rank、参数和保守结论产物。
- 不引入完整 AMBER 官方复现、不恢复旧入口、不继续训练已明确不建议主推的 PatternFiLM/JTT/MVFR/MPDRO/beamsoft/condBTAPA/weakKD 路线。

## Capabilities

### New Capabilities
- `scene31-baseline-pack`: Scene31 baseline pack 的 run group、fresh eval 复用、summary、sanity check、输出边界和保守结论规则。

### Modified Capabilities
- `local-missing-modality-baselines`: 将 AMR-lite、AMBER-lite baseline-pack 训练策略对照和 FeatureMod-lite 轻量 feature adaptation baseline 纳入本地 experimental baseline 边界。

## Impact

- 影响源码：`src/kd_sensing/models/`、`src/kd_sensing/engine/` 或已有 missing/difficulty runtime、必要 registry/default component wiring。
- 影响配置与脚本：`configs/scene31/` local/manual baseline pack 生成或实体配置、`scripts/run_scene31_baseline_pack.sh`、`scripts/summarize_scene31_baseline_pack.py`。
- 影响测试：新增 synthetic forward、dropout 分布、配置加载、summary/sanity check 和 runner dry-run/focused tests。
- 运行产物仍只写入 ignored `outputs/`、`logs/` 或显式本地路径；checkpoint、fresh eval 输出、CSV summary 和训练日志不进入源码变更。
