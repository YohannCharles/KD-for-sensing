## Why

Scene31 单场景与 Scene31-34 quick seed1 已经不足以支撑当前论文主实验结论；现在需要把多场景 Scene31-34 pooled training/eval 正式提升为主设定，并用 seed1/2/3 与缺失数量退化曲线支撑主表。

现有 quick seed1 显示 `proto_randomdrop_subset_es40` 是 pooled winner，但主线结论仍缺少多 seed、完整 baseline、per-scene 稳定性、missing_count=0/1/2/3 曲线和论文表格导出。

## What Changes

- 新增 Scene31-34 主实验 local/manual runner，默认只补 prototype 主线 baseline 的 seed2/3，并在需要时补齐 Bernoulli randomdrop seed1/2/3。
- 扩展 fresh eval 产物口径，要求每个 run 输出带 scene、pattern、missing_count、missing_ratio、可用/缺失模态和 per-sample prediction 的结构化 CSV。
- 新增 Scene31-34 主 summary，聚合 old-root seed1 与 new-root seed2/3，输出 per-run、method mean/std、per-scene、missing-count curve、delta、ranking 和 conclusion。
- 新增缺失数量退化曲线绘图脚本，生成 Top1、Within@3、MAE 以及 per-scene Top1 图的 PNG/PDF。
- 新增论文表格导出和最终主结论脚本，明确主方法冻结为 prototype + random subset exposure；Uniform 只作为 ablation，reliability fusion 和 PatternFiLM 不晋升。
- AMR/AMBER-lite 多场景 baseline 只允许作为可选 maskfix 配置准备，不阻塞 prototype 主实验。
- 新增普通 classifier CE head baseline、AMR/AMBER-lite Scene31-34 external-lite seed1 group、compute profile 表、最终 all-baseline summary/table/conclusion 输出，用于论文主表补齐非原型和外部 baseline 证据。
- 不继续 reliability fusion seed2/3、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 或 weakKD。

## Capabilities

### New Capabilities

- `scenes31-34-main-missing-modality-workflow`: 定义 Scene31-34 多场景主实验 runner、fresh eval 输出、summary、缺失数量退化曲线、论文表格和最终结论的契约。

### Modified Capabilities

- `mainline-experiment-documentation`: 主线文档需要记录 Scene31-34 作为缺失模态主实验设定、prototype + random subset exposure 作为冻结主方法，以及本地输出/claim 边界。

## Impact

- 影响 `scripts/` 下 Scene31-34 runner、summary、plot、paper table 和 conclusion 脚本。
- 可能复用并扩展 `scripts/scene31_runner_common.*`、`scripts/reevaluate_apples_to_apples.py` 和既有 Scene31 summary helper，但不得复制训练、DataLoader、模型加载或指标计算主逻辑。
- 影响 `configs/scene31/scenes31_34_*` 的 manifest/config family；新增或生成的训练 YAML 必须保持本地实验输出写入 ignored `outputs/`。
- 影响 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md` 和必要的 inventory 说明，但真实 metrics、figures、tables、logs、checkpoint 不纳入源码。
- 验证至少包括 `openspec validate promote-scenes31-34-main-missing-count --strict`，以及相关脚本/架构 focused tests 或无数据 smoke。
