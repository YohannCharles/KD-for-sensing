## 1. 场景与 LOSO 数据基础

- [x] 1.1 扩展 `kd_sensing.data.scenes` 的 DeepSense6G 场景描述符，加入 Scenario 33/34 的别名、默认路径、legacy 路径和 removed dataset-type 迁移提示。
- [x] 1.2 增加 Scenario 33/34 场景解析、默认路径、显式 `data_root`/CSV 覆盖和未知场景拒绝测试，使用 `conda run -n kd_mm_beam pytest ...` 运行相关测试。
- [x] 1.3 新增 LOSO fold 解析窄模块，支持默认 31-34 四折、单 target scene 选择、显式 source scenes 校验和 fold metadata。
- [x] 1.4 新增 target `target_adapt`/`target_test` 拆分模块，默认 20%/80%，优先按完整 `seq_index` 拆分，并写出 split metadata。
- [x] 1.5 新增 multi-scene source dataset/dataloader 构建逻辑，复用 `DeepSense6GDataset` 和现有 normalizer/scaler artifact 传递规则，不复制 dataset 类。
- [x] 1.6 新增 few-shot target labeled subset sampler，支持 budgets `0/5/10/20/50`、seed 可复现、coarse-group stratified sampling 和 sampling manifest。

## 2. HiST-Beam 模型与 loss

- [x] 2.1 新增 HiST-Beam 配置解析与校验 helper，覆盖 `num_classes`、`group_size`、variant、loss 权重、adapter/prototype 开关和启用模态。
- [x] 2.2 实现并注册 HiST-Beam fusion 模型，输出 beam-level logits/log-probs、coarse logits、fine logits、shared/private representation、adapter representation 和 scene diagnostics。
- [x] 2.3 接入 image/radar/gps 三模态 encoder/projector，image 默认使用 RGB/ImageNet ResNet-18 能力，radar/GPS 复用现有输入契约。
- [x] 2.4 实现 hierarchical label helper、coarse/fine label 计算、不可整除 group size 拒绝和 horizon 维 shape 校验。
- [x] 2.5 实现 HiST-Beam loss helper，覆盖 hierarchical loss、flat auxiliary loss、orthogonality loss、shared scene confusion loss 和 private scene preservation loss。
- [x] 2.6 将 HiST loss 通过训练扩展或 objective hook 接入 trainer，确保非 HiST 配置继续使用既有 beam loss。
- [x] 2.7 增加模型 forward、variant 开关、loss 分量、GRL/scene logits 和普通配置不受影响的单元测试，使用 `conda run -n kd_mm_beam pytest ...` 验证。

## 3. Prototype 与 target adaptation

- [x] 3.1 实现 source prototype 生成器，在 source train split 上保存 shared/private coarse prototypes、count 和 metadata。
- [x] 3.2 实现 prototype artifact 加载、空 group 处理、confidence threshold 和 prototype coverage 统计。
- [x] 3.3 实现 bottleneck private adapter，初始化时保证 adapter 路径与 source model 等价。
- [x] 3.4 实现 adapter-only 和 adapter+prototype 冻结策略，记录 trainable/total parameter count 和 trainable ratio。
- [x] 3.5 实现 full fine-tuning baseline 适应策略，复用同一 V3 source checkpoint 并记录 100% 或等价全量 trainable ratio。
- [x] 3.6 实现 supervised、unlabeled 和 semi-supervised target adaptation loop，确保只消费 `target_adapt` 数据。
- [x] 3.7 增加 adapter 初始化等价、冻结参数、prototype loss 过滤、0-label 和 few-shot adaptation 的单元测试，使用 `conda run -n kd_mm_beam pytest ...` 验证。

## 4. 指标、产物与结果汇总

- [x] 4.1 增加 coarse group accuracy、fine offset accuracy、prototype coverage、trainable parameter ratio 和 adaptation time 指标。
- [x] 4.2 扩展 evaluation/prediction 导出，保存 sample id、scene、true/pred beam、top-k predictions、coarse true/pred 和 variant metadata。
- [x] 4.3 支持可选 beam power metrics；当 beam power vector 缺失时记录 unavailable reason，不伪造 power 指标。
- [x] 4.4 实现 source-only、few-shot adaptation 和 efficiency 三类 LOSO summary 输出，支持按 fold、target scene、variant、budget 和 seed 聚合。
- [x] 4.5 增加 metrics、predictions 和 summary writer 测试，使用 `conda run -n kd_mm_beam pytest ...` 验证。

## 5. 配置、CLI 与工作流

- [x] 5.1 新增 `configs/hist_beam/` base、V0-V6 variants、quick-smoke 和 LOSO matrix 配置，默认启用 `image`、`radar`、`gps`。
- [x] 5.2 新增包内 LOSO/adaptation CLI 或模块入口，并在 `pyproject.toml` 暴露 console script；不得新增长期维护的根目录训练脚本。
- [x] 5.3 支持单 fold 单 variant、默认 quick matrix、budgets/seeds/folds 缩小、已有 source checkpoint 复用和 overwrite/resume 行为记录。
- [x] 5.4 补充 README 简短入口和 docs/OpenSpec 指向，完整研究矩阵保留在 OpenSpec 或 docs 中。
- [x] 5.5 增加 CLI help smoke 测试，使用 `conda run -n kd_mm_beam <new-cli> --help` 和相关 pytest 验证。

## 6. 验收与回归

- [x] 6.1 运行 `openspec validate add-hist-beam-cross-scene-adaptation --strict`，修复所有 OpenSpec schema 或 requirement 问题。
- [x] 6.2 运行 `openspec status --change add-hist-beam-cross-scene-adaptation`，确认 proposal、design、specs 和 tasks 状态正确。
- [x] 6.3 运行架构边界快速检查：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 6.4 运行新能力相关测试和 CLI help smoke，确保不触发真实大规模训练。
- [x] 6.5 在实现完成后运行最终回归：`conda run -n kd_mm_beam pytest -q`。
