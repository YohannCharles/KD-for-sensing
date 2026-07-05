## Why

当前 Scene31 funnel fresh eval 已经修到 complete run 可稳定使用 best checkpoint，但 PatternFiLM d8 只有 seed1，且 fresh eval summary 的 miss2 bucket 为空，无法支撑保守的 n=5 结论判断。

本变更只补齐 PatternFiLM d8 多 seed、fresh eval 缺两个模态的 patterns、以及对应 summary/sanity 输出；不扩展 JTT、MVFR、MP-DRO、beamsoft、condBTAPA、weakKD 或新模型方向。

## What Changes

- 确认并补齐 Scene31 funnel 中 PatternFiLM d8 的配置和实现消费路径，确保 `d8` 对应 dim=8、`init_identity=true`、`apply_at=pre_head`，且只基于可用/缺失模态 pattern 条件化。
- 在 funnel manifest/generator 中新增 `proto_sampler_uniform_pattern_film_d8_es40_seed2/3/4/5`，保持除 seed 外与 seed1 一致。
- 扩展 fresh eval pattern 列表，使缺两个模态的 patterns 出现在 pattern-wise CSV、bucket mapping 和 summary 中。
- 新增薄 runner `scripts/run_scene31_patternfilm_d8.sh`，只编排本次需要的训练/eval group，并复用现有训练入口与 apples-to-apples fresh eval helper。
- 新增/扩展 PatternFiLM d8 专用 summary，输出 per-run、method mean/std、delta、rank、bucket mapping、conclusion 与 sanity check。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `scene31-next-round-experiment-workflow`: 收紧 Scene31 funnel quick-screen 后续补跑契约，要求 PatternFiLM d8 seed1-5、miss2 fresh eval patterns、完整 missing bucket mapping 和保守晋级结论输出。

## Impact

- 影响 `configs/scene31/funnel` 生成逻辑、Scene31 local/manual runner、fresh eval pattern 默认值、funnel/PatternFiLM summary 脚本和少量模型实现。
- 不新增 package CLI、不新增外部依赖、不改本地输出和 checkpoint 归档边界。
- 训练和 fresh eval 输出仍写入 ignored 的 `outputs/scene31_funnel_lmdb` 或用户显式指定 root。
