## 1. 契约测试与基线 fixture

- [x] 1.1 新增 `tests/test_model_architecture_summary.py` focused test 骨架，覆盖 schema 顶层字段、JSON 序列化、source kind、参数来源、warning 和 renderer。
- [x] 1.2 添加 synthetic module fixture，验证 total/trainable/frozen 参数去重统计、shared parameter 去重、effective/excluded 参数口径和 unknown component role。
- [x] 1.3 添加 `modular_sequence` image-only ResNet fixture，验证 image encoder、projector、representation core、beam head 分组和旧 startup 参数字段兼容。
- [x] 1.4 添加 `modular_sequence` image+GPS fixture，验证 image encoder params、GPS encoder params、fusion core params、enabled modalities 和组件 metadata 合并。
- [x] 1.5 添加 TinyViT scratch fixture，验证 variant、backbone_dim、output_dim、freeze policy、trainable stages、checkpoint metadata、potential unused classifier head warning 或 excluded parameter group。
- [x] 1.6 添加 TinyViT override preflight fixture，覆盖从 ResNet config 切到 TinyViT 但保留 `unfreeze_stages: [layer4]` 时输出 `incompatible_encoder_option` warning 或清晰构建错误。
- [x] 1.7 添加 JEPA sweep 参数 fixture，锁定 `patch14_stage1_gps_query` 约 0.197M total params、0.117M image encoder params、0.088M visual/context encoder params。
- [x] 1.8 添加 ResNet token sweep fixture，锁定 `resnet18_layer4_tokens` 约 11.32M/11.24M/11.21M 和 `resnet18_layer3_layer4_tokens` 约 14.13M/14.05M/14.02M 参数口径。

## 2. 核心 summary helper

- [x] 2.1 新增 `src/kd_sensing/models/architecture_summary.py` 或等价窄模块，定义 schema version、数据结构 helper、JSON-safe 转换和公共 API。
- [x] 2.2 实现实例级参数统计，基于 `named_parameters()` 去重统计 total/trainable/frozen，并保留参数路径、module path 和 class。
- [x] 2.3 实现 effective/excluded 参数口径，支持组件声明或已知模式标记未参与 downstream forward 的参数组，首批覆盖 TinyViT downstream unused classifier head。
- [x] 2.4 实现 component role 推断，优先识别 `modular_sequence` 的 encoders/projectors/representation_core/heads/geometry_prior/logit_fusion/reranker。
- [x] 2.5 实现已知视觉组件语义分组，支持 image encoder params、visual/context encoder params、backbone/projection/pooler/head 等聚合字段。
- [x] 2.6 实现 metadata 合并，整合组件 `training_strategy_metadata()` 的 registry type、checkpoint、freeze policy、token metadata、reliability metadata 和 output dimension。
- [x] 2.7 实现 warning 生成，覆盖 incompatible encoder option、potential checkpoint download、unused parameter group、declared-vs-actual mismatch 和 unknown component role。
- [x] 2.8 实现 renderer，至少支持 JSON 对象、Markdown 表格和 CSV row/list 输出。

## 3. 配置与安全 build 路径

- [x] 3.1 实现 `summarize_model_config()`，从 resolved model config 构建模型或做 preflight summary，默认不构建 dataset、不创建 optimizer、不训练。
- [x] 3.2 默认禁止 checkpoint 下载或网络访问；对需要下载的 22k TinyViT 或其它 pretrained 配置输出 warning/error，除非用户显式允许。
- [x] 3.3 支持 config overrides，与现有 `load_config` override 语义一致；测试中所有 Python 命令使用 `conda run -n kd_mm_beam`。
- [x] 3.4 确认 summary 调用前后模型参数 `requires_grad` 状态不变，forward 输出契约不变。
- [x] 3.5 为无法 build 的配置提供 preflight-only summary 或清晰错误，错误中包含配置路径、组件名和可用选项。

## 4. Startup summary 和训练集成

- [x] 4.1 将 `src/kd_sensing/engine/debug_diagnostics.py` 的 `module_trainability_report()` 迁移为调用或包装新 architecture summary helper。
- [x] 4.2 保留 `startup_summary.json` 既有 `parameters.total_params`、`parameters.trainable_params` 和 `parameters.modules` 字段，保证现有 tests 和 TensorBoard scalars 不回归。
- [x] 4.3 在 startup summary 中新增 `architecture_summary` 或等价字段，包含统一 schema 的摘要。
- [x] 4.4 更新 TensorBoard startup scalar 写入测试，确认旧 tag 保留；如新增组件级 tag，确保 inactive/missing 值被跳过。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` 或相关 startup focused tests。

## 5. JEPA sweep summary 集成

- [x] 5.1 在 `src/kd_sensing/diagnostics/cnn_hybrid_jepa_visual_prior_sweep.py` 或相关 owner 模块中添加 architecture summary adapter，将现有 `params_metadata` 映射到统一 schema。
- [x] 5.2 保留 full results 中 missing、failed、skipped、unavailable 候选的参数摘要字段，不因 metrics 缺失删除候选。
- [x] 5.3 更新 Pareto、family best 和 Markdown summary，使用 total params、trainable params、image encoder params、visual/context encoder params、token count 和 compute proxy。
- [x] 5.4 添加 `patch14_stage1_gps_query`、`resnet18_layer4_tokens`、`resnet18_layer3_layer4_tokens` 的 summary fixture tests。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_cnn_hybrid_jepa_visual_prior_sweep.py -q`。

## 6. CLI 与文档

- [x] 6.1 新增包内薄 CLI，例如 `src/kd_sensing/cli/model_architecture_summary.py`，支持 `--config`、`-o/--override`、`--sweep-manifest`、`--variant-id`、`--startup-summary`、`--format` 和 `--output`。
- [x] 6.2 如新增 console script，更新 `pyproject.toml`，并同步 `docs/maintainer_context_index.yaml` entrypoint owner metadata、output boundary 和 validation commands。（本次未新增 console script，使用 `python -m kd_sensing.cli.model_architecture_summary` 包内入口。）
- [x] 6.3 更新 `docs/project_surface_inventory.md`，说明该能力是 current diagnostic/architecture summary 入口，不是运行时配置或第二套 registry。
- [x] 6.4 更新 README 或 `docs/extension_guide.md` 的短示例，展示如何查看当前配置、TinyViT override 和 JEPA sweep 候选参数量。
- [x] 6.5 添加 CLI help/format tests，运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` 或新增 CLI focused test。
- [x] 6.6 新增 `docs/model_architecture_inventory.md` 人类可读目录，集中展示当前整模型、encoder、projector、representation core、head、摘要字段口径和退役边界。

## 7. 验证与收口

- [x] 7.1 运行 `openspec validate add-model-architecture-summary --strict`。
- [x] 7.2 运行 `conda run -n kd_mm_beam pytest tests/test_model_architecture_summary.py -q`。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_tinyvit_image_encoder.py -q`。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_cnn_hybrid_jepa_visual_prior_sweep.py -q`。
- [x] 7.5 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
- [x] 7.6 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。（已运行；当前工作树既有无关治理漂移导致失败：未分类 `知乎问答下载.md`，以及 `jepa-visual-architecture-sweep`、`real-perturbation-forward-evaluation`、`safe-residual-beam-rerank-fusion` spec lifecycle/purpose 问题。）
- [x] 7.7 若 CLI 新增 console script，运行对应 `conda run -n kd_mm_beam kd-sensing-model-summary --help` 或最终命名的 `--help` smoke。（本次未新增 console script；已运行 `conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary --help` 和 sweep CSV smoke。）
- [x] 7.8 汇总最终实现说明，列出参数口径定义、patch14/ResNet token 校准表、TinyViT unused head 处理结果、验证命令结果和剩余 caveat。
