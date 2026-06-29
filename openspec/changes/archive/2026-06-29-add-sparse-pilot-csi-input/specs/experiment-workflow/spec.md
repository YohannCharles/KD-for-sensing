## ADDED Requirements

### Requirement: Sparse-pilot physics-informed MMW config
系统 MUST 提供 sparse-pilot physics-informed MMW 配置，使用现有 `kd-sensing-train` 入口运行，不新增根脚本。该配置 MUST 启用 image + sparse pilot CSI 输入，并保持完整 CSI/path/beam power 作为训练监督或诊断。

#### Scenario: sparse pilot 配置加载
- **WHEN** 用户加载 `configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml`
- **THEN** 配置 MUST 设置 `data.csi_input_mode=sparse_pilot`
- **AND** `model.primary.modalities` MUST 包含 `image` 和 `csi`
- **AND** `oracle_full` 仍 MUST 只作为 upper-bound 配置
