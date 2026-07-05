## Context

Scene31 当前最稳定结果来自 `proto_sampler_uniform_es40`，但现有证据主要证明 proto + uniform sampler 强于若干复杂候选。论文还需要一组更基础、更可解释的 baseline：natural proto、random modality dropout、AMR-lite、AMBER-lite，以及可选 FeatureMod-lite，并且这些结果必须走同一套 fresh eval 和 miss1/miss2/miss3 bucket summary。

仓库已有 Scene31 local/manual workflow、apples-to-apples fresh eval、missing bucket summary、`modular_sequence` 组件边界和本地 AMBER-lite baseline 规格。本 change 不新增长期 package CLI，不复制训练/评估循环，只补 baseline-pack 的本地配置、轻量模型组件、runner 和汇总脚本。

## Goals / Non-Goals

**Goals:**
- 为 proto natural、random dropout、AMR-lite、AMBER-lite 和 FeatureMod-lite 提供可训练配置或生成路径。
- 为 random dropout 记录实际训练暴露分布，用于和 pattern-balanced exposure 区分。
- 对 AMR-lite/AMBER-lite 同时支持 natural/random 与 pattern-balanced exposure 训练策略对照。
- 统一 fresh eval 口径：best checkpoint、无 `--max-batches`、full/miss1/miss2/miss3、within@3、MAE、balanced。
- 输出保守 summary，不把单 seed quick screen 或 local-lite baseline 夸大为最终 claim。

**Non-Goals:**
- 不完整复现 AMBER 官方架构或外部论文训练协议。
- 不继续训练 PatternFiLM/JTT/MVFR/MPDRO/beamsoft/condBTAPA/weakKD。
- 不用 fresh eval/test set 拟合校准或 checkpoint selection。
- 不把本地训练输出、checkpoint、日志或 summary CSV 纳入源码。

## Decisions

1. **baseline pack 作为 Scene31 local/manual workflow 实现。**  
   复用已有 shell runner 风格和 `scripts/scene31_runner_common.py`，新增 `scripts/run_scene31_baseline_pack.sh` 只做 run 选择、GPU worker、skip/overwrite、train/eval 调度和日志路径管理。这样不扩大 package CLI 表面，也符合现有 Scene31 manifest-backed workflow 边界。

2. **random modality dropout 进入共享训练扰动路径。**  
   新增配置字段采用 `random_modality_dropout.enabled/mode/keep_prob/ensure_at_least_one_modality`。训练 batch 中只修改输入模态和 availability/missing mask metadata，不修改 target、split、sample id 或 label。每 epoch 写出 `random_dropout_pattern_stats.csv`，用于 sanity check 和 summary。

3. **AMR-lite/AMBER-lite/FeatureMod-lite 优先走组件路径。**  
   AMR-lite 用轻量 imputation token + modality/channel gate 表达；AMBER-lite 用一层小型 transformer fusion core 表达；FeatureMod-lite 用 missing-modalities condition 的小 adapter 表达。只有现有组件边界无法承载时，才允许极小 whole-model exception，并必须补 registry/forward/metadata/summary tests。

4. **summary 只读取产物并保守归并。**  
   `scripts/summarize_scene31_baseline_pack.py` 读取本轮 root、旧 uniform reference root 和可选 proto baseline root，过滤 `status != ok`、`missing_config`、`missing_checkpoint` 等不可均值项，按 method/model_family/training_strategy 输出 mean/std、delta、rank 和 `baseline_conclusion.txt`。字段名跟当前 missing bucket/fresh eval summary 对齐。

5. **参数量来自真实模型或明确标注来源。**  
   可构建模型用 `named_parameters()` 统计 total/trainable params；无法构建的历史 reference 或外部旧结果只允许为空或声明来源，不把估算当真实参数量。

## Risks / Trade-offs

- **训练全量 baseline pack 耗时很长** → runner 支持 group、train-only、eval-only、auto-eval、skip 和失败不中断；最终交付可先完成实现和 dry-run/focused tests，再由用户选择长训窗口。
- **random subset dropout 与 uniform sampler 容易混淆** → 实现和日志分开：random subset 是样本级随机非空 available set，uniform sampler 是显式 pattern-balanced exposure；summary 输出实际分布差异。
- **AMR-lite/AMBER-lite 过度实现** → 只实现最小可比较版本，不引入大型 transformer、temporal alignment、class-former 或外部论文专有训练流程。
- **summary 读取旧产物格式不一致** → reader 宽容解析已知 fresh eval CSV/JSON，无法确认的 run 标记 warning，不参与 method 均值。

## Migration Plan

1. 新增 OpenSpec delta、配置/组件/runner/summary 和 focused tests。
2. 运行 OpenSpec validate、配置加载、模型 forward、summary fixture 和架构边界 focused tests。
3. 需要真实实验时，用 `bash scripts/run_scene31_baseline_pack.sh --group <group> --gpus <ids> --auto-eval` 在本地输出 root 续跑；默认跳过已完成训练和已有 ok fresh eval。
4. 回滚时删除 baseline-pack 配置、runner、summary 和新增组件；既有 Scene31 roots 与历史结果不受影响。

## Open Questions

- 是否全量跑完所有 3-seed baseline，取决于可用 GPU 时间；实现层面必须支持，但本 change 不把长训完成作为源码测试前置。
- FeatureMod-lite 是建议项，若实现成本超过轻量组件边界，可只保留配置/summary 支持并在结论中标记 skipped。
