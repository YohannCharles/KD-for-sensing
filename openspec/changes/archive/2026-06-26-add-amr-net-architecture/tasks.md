## 1. 模型实现

- [x] 1.1 新增 `src/kd_sensing/models/amr_net.py`，实现 `amr_net` whole-model exception、三模态 snapshot shape 校验和 `supports_modality_kwargs=True`。
- [x] 1.2 实现 AMR image、LiDAR、GPS 私有 encoder，支持 paper-aligned `[B,1,1,224,224]`、`[B,1,216,2]`、`[B,1,2]` synthetic forward。
- [x] 1.3 实现 per-modality 概率嵌入、reparameterization、classifier 和 deterministic eval 路径。
- [x] 1.4 实现 CUAF helper，输出 fused logits、modality weights、entropy、cross-modal KL consistency、top-k margin 和 finite diagnostics。
- [x] 1.5 在默认模型组件导入中注册 `amr_net`，保持 `import kd_sensing.registries` 轻量导入边界。

## 2. Loss、metadata 和摘要

- [x] 2.1 新增 AMR loss helper，基于 `ModelOutput` diagnostics 计算 CE、KL、FEP 和可选 PRE，并处理无正样本 batch。
- [x] 2.2 将 AMR loss 接入现有训练/objective loss 路径，避免模型 forward 接收 labels。
- [x] 2.3 实现 `training_strategy_metadata()`，记录 whole-model exception、modalities、latent/CUAF/loss 配置、reliability metadata 消费和 `paper_approximation`。
- [x] 2.4 确认模型架构摘要能统计 AMR-Net total/trainable params，并保留 metadata。

## 3. 配置和文档

- [x] 3.1 新增最小 current AMR-Net 配置或 overlay，使用 `model.primary.type: amr_net`，不包含退役 token `amr_net_gps_image`。
- [x] 3.2 更新模型目录、实验协议或实验矩阵的最小条目，说明新 AMR-Net 是 current architecture baseline，不是旧 source-audit runner。
- [x] 3.3 确认旧 `amr_net_gps_image` runner/config token 仍被拒绝，不新增兼容 facade 或根目录脚本。

## 4. 测试和验证

- [x] 4.1 新增 AMR-Net focused tests，覆盖 registry build、synthetic forward、`adapt_model_output`、CUAF finite 输出和 metadata。
- [x] 4.2 新增 AMR loss tests，覆盖 KL/FEP/PRE、无正样本 batch 降级和 diagnostics。
- [x] 4.3 新增或扩展配置加载、架构摘要和旧入口隔离测试。
- [x] 4.4 运行 `openspec validate add-amr-net-architecture --strict`。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.6 运行 AMR focused pytest，例如 `conda run -n kd_mm_beam pytest tests/test_amr_net.py tests/test_config_load_characterization.py tests/test_model_architecture_summary.py -q`。
