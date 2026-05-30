## 1. 引用盘点与边界确认

- [x] 1.1 使用 `rg -n "MARF|marf|CRAF|craf|G2D|g2d|multimodal[_-]?nf|Multimodal-NF|MultimodalNF" src configs tests scripts tools README.md docs openspec/specs pyproject.toml` 盘点所有 active 引用，并按源码、配置、测试、文档、OpenSpec 分类
- [x] 1.2 确认哪些 helper 属于 CRAF/MARF/G2D/Multimodal-NF 专属代码，哪些仍被当前保留的 DeepSense6G、MMW、Raymobtime、CSI、标准 fusion 或 viewer workflow 依赖
- [x] 1.3 确认 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和真实数据目录不纳入删除任务
- [x] 1.4 处理或记录空壳 active change `remove-craf-marf-architectures`，避免后续 OpenSpec 状态与实际方案重复或漂移

## 2. 源码与注册入口删除

- [x] 2.1 删除 CRAF/MARF 模型实现、训练 helper、teacher-prior gate/loader、counterfactual/reliability loss 和相关 active import
- [x] 2.2 删除 G2D distiller、SMP、teacher ensemble、diagnostics accumulator、训练接入和相关 active import
- [x] 2.3 删除 Multimodal-NF dataset、runtime metadata、preprocessing/audit/index/cache/codebook、profile/cache IO helper 和相关 active import
- [x] 2.4 从默认组件导入和 registry 中移除 `craf_fusion`、`marf_fusion`、`g2d` distiller、`multimodal_nf` dataset/preprocessor 等注册名
- [x] 2.5 清理 config loader、virtual/overlay recipe、artifact metadata、run index、profile 工具中对退役入口的专属分支，确保不保留同名 fallback alias
- [x] 2.6 保留当前核心 CLI、标准单模态、标准/模块化 fusion、MMW、DeepSense6G、Raymobtime、CSI 和 viewer manifest 的代码路径可导入

## 3. 配置与文档清理

- [x] 3.1 删除 `configs/multimodal_nf/` 和 `configs/preprocess/multimodal_nf_*.yaml`
- [x] 3.2 删除 CRAF、MARF、G2D 的 fusion 实体 YAML、overlay recipe、retired entity expectation 和示例配置入口
- [x] 3.3 清理 README 中 G2D、CRAF、MARF、Multimodal-NF 的推荐命令、实验矩阵和数据布局说明
- [x] 3.4 清理 `docs/`、脚本注释和帮助文本中对退役研究线的推荐入口；历史 archive 文档不改写
- [x] 3.5 更新 active `openspec/specs/`，归档后不再保留 G2D、teacher-prior CRAF、Multimodal-NF 及其近场 objective/cache/profile 正向承诺

## 4. 测试调整

- [x] 4.1 删除 CRAF、MARF、G2D、Multimodal-NF 的正向单元测试、fixture 和 smoke 训练测试
- [x] 4.2 补充或调整 registry/config failure 测试，验证 `craf_fusion`、`marf_fusion`、`distillation.type: g2d`、`data.dataset.type: multimodal_nf` 不再可构建
- [x] 4.3 更新架构边界测试，拒绝 active code path、registry、README/docs 和推荐配置中残留退役研究线入口
- [x] 4.4 更新现有配置加载测试，保留当前支持配置的覆盖，移除 CRAF/MARF/G2D/Multimodal-NF 成功加载断言

## 5. 验证

- [x] 5.1 运行 `openspec validate prune-legacy-code --strict`
- [x] 5.2 运行 `openspec status --change prune-legacy-code`
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 5.4 运行 `conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-preprocess --help`
- [x] 5.5 运行当前保留能力的 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_prediction_objectives.py -q`；如这些测试被任务调整拆分，则运行对应替代测试
- [x] 5.6 最终运行 `conda run -n kd_mm_beam pytest -q`，如耗时或环境阻塞则记录未运行原因和已完成的 focused 验证
