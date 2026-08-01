## MODIFIED Requirements

### Requirement: MMW canonical surface 只包含 U0 和两个 baseline

MMW canonical recipes MUST 仅为 `configs/mmw/u0.yaml`、`amber_full.yaml` 和 `rmbp_mm.yaml`，共享 `_base.yaml`。MMW canonical launcher、evaluator 和 summary MUST 仅接受 `U0`、`amber_full`、`rmbp_mm`。隔离注册的 `pcpf_temporal_risk_fusion` MAY 仅通过 `tools/configs/pcpf/` 和本地研究工具运行，不得被描述为 current U0 或 canonical MMW recipe。

#### Scenario: 加载 MMW canonical recipe

- **WHEN** 用户加载任一 MMW canonical recipe
- **THEN** 配置 MUST 不包含 PCPF-T、BCACL、CMSBL、capacity/reference、nested capacity、recovery 或历史 ablation 字段

#### Scenario: 运行 PCPF-T 本地研究入口

- **WHEN** 用户显式加载 `tools/configs/pcpf/` 下的 PCPF-T recipe
- **THEN** 运行 MUST 复用显式绑定且审计通过的 MMW 数据契约；sparse-CSI 正式路线 MUST 使用 `mmw_id_stratified_block_v1`
- **AND** canonical MMW launcher 与 recipe 列表 MUST 保持不变

#### Scenario: PCPF-T 启用第五个 sparse CSI 专家

- **WHEN** 本地 PCPF-T recipe 显式声明 `use_sparse_csi=true`
- **THEN** 第五模态 MUST 只存在于该模型和 dataset sidecar
- **AND** 全局 canonical modality order、U0 config、U0 state dict 与公共 CLI MUST 保持不变

## ADDED Requirements

### Requirement: PCPF-T 不得改变 U0 数值路径

U0 在未声明 `pcpf_temporal_risk_fusion` 时 MUST 不创建、加载或执行 PCPF temporal、probability、risk、temperature、analytic fusion 或 control 参数。现有 supervised Router、prototype、checkpoint key 和默认 forward 输出 MUST 保持原契约。

#### Scenario: 构建或恢复 U0

- **WHEN** 配置的模型类型为现有 U0
- **THEN** state dict、optimizer 参数集合和 forward 图 MUST 不包含任何 PCPF owner
