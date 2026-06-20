## ADDED Requirements

### Requirement: 主线文档不得保留优先退役 claim 行
主线模型目录、实验协议表和结果 claim 账本 MUST 从 current 表中移除 AMR-Net_gps_image 和 JEPA-MSAC 的 pending、mock-smoke、blocked official 或 local-ready current 行。若保留历史背景，MUST 放入 retired/historical/tombstone 说明，不得作为当前可运行或待引用 claim 占位。

#### Scenario: 主线目录不列 AMR 和 JEPA-MSAC current 行
- **WHEN** 开发者阅读 `docs/mainline_model_catalog.md`
- **THEN** 文档 MUST 不把 AMR-Net_gps_image 列为 current model line
- **AND** 文档 MUST 不把 JEPA-MSAC Scenario 32 列为 current model line
- **AND** 如提到二者，MUST 标记为 retired、historical 或 blocked background

#### Scenario: claim 账本不保留 current pending 占位
- **WHEN** 开发者阅读 `docs/result_claims_registry.md`
- **THEN** 账本 MUST 不保留 AMR-Net_gps_image 或 JEPA-MSAC 的 current pending/mock-smoke claim 行
- **AND** 账本 MAY 保留一段退役说明，解释历史 blocked 原因和本地产物只作为 archive 背景

### Requirement: 当前文档只推荐保留入口
README、实验矩阵和协议表 MUST 只推荐仍维护的 current package CLI、config、diagnostic 或 shell runner。被退役入口的命令 MAY 出现在历史说明中，但 MUST 明确不可作为当前 quickstart 或正式复现实验。

#### Scenario: README quickstart 无退役命令
- **WHEN** 开发者阅读 README 的 quickstart、实验矩阵索引或 MMW 小节
- **THEN** README MUST 不提供 `kd-sensing-run-amr-net-gps-image` 或 `kd-sensing-run-jepa-msac` 作为当前命令
- **AND** README MUST 不提供被退役 shell orchestration 脚本作为当前命令

