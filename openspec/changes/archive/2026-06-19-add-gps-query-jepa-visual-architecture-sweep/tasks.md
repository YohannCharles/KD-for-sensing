## 1. 基础抽象与默认兼容

- [x] 1.1 新增 visual token encoder metadata 结构或等价 helper，记录 `variant_id`、token source、image size、effective stride、token grid、token count、positional encoding 和 `checkpoint_policy`。
- [x] 1.2 新增 visual token encoder 构建入口或 registry，默认 `patch_vit` 必须保持现有 `VisualPatchTokenEncoder` patch16 行为兼容。
- [x] 1.3 为现有 patch16 JEPA Stage 1 和 downstream 配置补充 runtime metadata，不改变既有 forward 输出和 checkpoint 加载行为。
- [x] 1.4 添加 focused tests，覆盖默认 patch16 encoder 的 shape、metadata 和旧配置无修改加载。

## 2. JEPA Stage 1 visual tokenizer variants

- [x] 2.1 实现 patch granularity variants，支持 patch16、patch14、patch8 和可配置 `max_tokens`，并在 token budget 超限时报清晰错误。
- [x] 2.2 实现 overlap patch tokenizer，支持 kernel/stride 可配置，并输出正确 token grid 和 effective stride metadata。
- [x] 2.3 实现 stacked 3x3 conv stem tokenizer，作为 patchify stem 的卷积化替代，保持输出 `[B,T,N,D]`。
- [x] 2.4 实现 local token mixing variant，例如 depthwise FFN、local preblock 或等价轻量局部归纳偏置模块。
- [x] 2.5 实现 CvT-style convolutional projection 或 depthwise token mixing variant，不新增重依赖。
- [x] 2.6 扩展 JEPA mask sampler 和 GPS angle biased sampling，使其从 token/grid metadata 读取 token count，不硬编码 196 或 14x14。
- [x] 2.7 实现 visual encoder checkpoint policy metadata，区分 `exact_reuse`、`partial_reuse`、`pos_interpolate`、`fresh_stage1_required` 和 `supervised_only_anchor`。
- [x] 2.8 添加 Stage 1 synthetic forward tests，覆盖每个 tokenizer variant 的 tokens、grid、mask sampler 和 checkpoint policy。

## 3. Downstream token sources、pooler 与 fusion core

- [x] 3.1 扩展 `jepa_context_image` downstream，使其能接收新 visual token metadata 并继续默认输出 `[B,T,D]`。
- [x] 3.2 实现 CNN feature-map token source，至少支持 ResNet18 layer3/layer4 tokens，并记录 backbone、stage、grid、token count、pretrained/freeze policy。
- [x] 3.3 实现多尺度 token source，支持 layer3+layer4 或等价 low/high resolution token concat，并添加 scale embedding 或等价 metadata。
- [x] 3.4 保留现有 Image ResNet+GPS frame embedding anchor，并在 sweep metadata 中标记为 `supervised_only_anchor`。
- [x] 3.5 扩展 GPS-query/hybrid/Predictive GPS-query++ diagnostics，记录 token grid、attention shape、attention entropy/peakiness、branch/gate weights 和 condition feature source。
- [x] 3.6 实现显式 opt-in 的 K-token pooler output mode，并新增或复用 token-aware representation core；默认 pooler 输出必须保持 `[B,T,D]`。
- [x] 3.7 添加 downstream forward tests，覆盖 JEPA tokens、CNN tokens、多尺度 tokens、K-token output mode、不兼容 core 报错和 diagnostics。

## 4. Sweep 配置矩阵与 manifest

- [x] 4.1 定义 architecture sweep manifest schema，包含 candidate metadata、strict comparability fields、checkpoint policy、run tier、metrics path 和 command provenance。
- [x] 4.2 新增 smoke 配置矩阵，覆盖 baseline、patch granularity、overlap tokenizer、conv stem、local token mixing、CNN tokens、多尺度 tokens、frame embedding anchor、pooler/core ablation 和 non-transformer control。
- [x] 4.3 新增 lowmem 配置矩阵，为 patch8、高分辨率和多尺度候选设置 batch size、AMP、gradient accumulation 或 token budget fallback。
- [x] 4.4 新增 strict 配置矩阵，继承匹配 2604 S32/S33/S34 或 BeamBench-fair baseline 的 split、label space、metric profile、history window、seed 和 GPS feature mode。
- [x] 4.5 为需要 Stage 1 retrain 的 tokenizer variants 新增 pretraining configs，并为 downstream configs 显式引用对应 checkpoint path 或 placeholder provenance。
- [x] 4.6 为 checkpoint-compatible downstream variants 新增派生 configs，只覆盖 pooler、adapter、freeze policy、parameter groups、run name 或 ablation metadata。
- [x] 4.7 写出 train/evaluate command manifest，所有 Python 项目命令必须使用 `conda run -n kd_mm_beam`。

## 5. Sweep 诊断、结果汇总与 claim gate

- [x] 5.1 实现或扩展 sweep summary writer，汇总 Top-1、Top-3、Top-5、DBA、相邻 beam error、token count、trainable params、compute proxy 和 diagnostics 状态。
- [x] 5.2 扩展 JEPA visual analysis 或相关 diagnostics，使 attention/activation/branch/gate summary 能按 variant 写入 CSV/JSON。
- [x] 5.3 增加 GPS shortcut 诊断字段，确保 wrong-GPS、counterfactual GPS、P3/P4 或 P0-P5 condition metrics 能与 clean 指标分开记录。
- [x] 5.4 实现 strict comparability gate：字段不一致、只完成 smoke/lowmem 或 checkpoint policy 不可比的候选不得升级为主 claim。
- [x] 5.5 确保所有 sweep 运行产物默认写入 ignored `outputs/analysis/jepa_visual_architecture_sweep/` 或配置声明的 ignored output root。

## 6. 测试覆盖

- [x] 6.1 添加 visual token encoder registry/build tests，覆盖未知 variant 报错和默认 variant 兼容。
- [x] 6.2 添加 tokenizer variant shape tests，覆盖 patch、overlap、conv stem、local/CvT-like、CNN token 和多尺度 token。
- [x] 6.3 添加 JEPA mask sampler tests，覆盖非 14x14 grid、token budget、GPS angle biased sampling 和多尺度 metadata。
- [x] 6.4 添加 downstream pooler/core tests，覆盖 `[B,T,D]` 默认输出、K-token opt-in、attention diagnostics 和不兼容 core 报错。
- [x] 6.5 添加 sweep config load tests，覆盖 smoke/lowmem/strict 配置和缺失 checkpoint 时的可诊断行为。
- [x] 6.6 添加 architecture boundary tests，确保未新增 root-level legacy scripts、旧 KD/HiST/Top8/camera residual/GPS residual 路线或绕过 `src/kd_sensing` 的入口。
- [x] 6.7 添加 manifest/summary tests，覆盖 strict comparability fields、variant metadata、run commands 和 ignored output root。

## 7. 验证命令

- [x] 7.1 运行 `openspec validate add-gps-query-jepa-visual-architecture-sweep --strict`。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q`。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_cls_token_transformer_fusion.py -q` 或新增 token-aware core focused tests。
- [ ] 7.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 7.6 如新增诊断输出，运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py -q` 或新增 JEPA sweep diagnostics focused tests。
