## ADDED Requirements

### Requirement: JEPA temporal context fallback
JEPA downstream image encoder MUST 支持可配置 temporal context fallback，用历史 image latent 预测当前 degraded/missing image latent。Fallback MUST 只使用当前时间步之前的 image history，例如 `image_history[t-4:t-1]`，MUST 不读取未来帧或移动 target。

#### Scenario: 历史 latent 预测当前 latent
- **WHEN** 配置启用 JEPA temporal context fallback 且 batch 提供足够历史帧
- **THEN** encoder MUST 使用历史 image latent 生成 predicted `z_img[t]`
- **AND** metadata MUST 记录 source history range 和是否命中 fallback

#### Scenario: 不足历史可审计降级
- **WHEN** 历史长度不足以构造 `t-4:t-1`
- **THEN** 系统 MUST 使用配置声明的 clamp、zero、skip 或 raw latent fallback
- **AND** warnings MUST 记录受影响样本数和 fallback 策略

### Requirement: JEPA downstream 消费 observability metadata
JEPA downstream 模型 MUST 能消费 `image_valid_mask`、`image_observability_score` 和 Scenario D condition metadata，用于决定是否使用 raw current latent、temporal predicted latent 或二者的 gated mixture。

#### Scenario: 低 image observability 启用 predicted latent
- **WHEN** `image_observability_score` 低于配置阈值或 `image_valid_mask=false`
- **THEN** JEPA downstream MUST 能输出 temporal predicted latent 或 predicted/raw mixture
- **AND** downstream metadata MUST 记录 gating decision

#### Scenario: clean image 使用 current latent
- **WHEN** image condition 为 `D0_full_image` 且 `image_valid_mask=true`
- **THEN** JEPA downstream MUST 保持 current latent 作为默认输入
- **AND** mean-pooling 和 GPS-query baseline MUST 不因未启用 fallback 而改变行为

### Requirement: JEPA fallback 与 benchmark condition 对齐
JEPA downstream MUST 能接收 benchmark condition metadata，用于标记 `C3/C4 + D3/D4/D6/D7` advantage condition。该 metadata MUST 只影响 gating/fallback 选择和 diagnostics，不得改变 target、loss label 或 sample order。

#### Scenario: advantage condition metadata 传递
- **WHEN** Scenario D benchmark 评估 Image-JEPA+GPS 且 condition 为 `C4_severe_async + D6_burst_missing`
- **THEN** downstream MUST 能识别该 condition 为 JEPA advantage condition
- **AND** runtime metadata MUST 记录该 condition 与 fallback decision
