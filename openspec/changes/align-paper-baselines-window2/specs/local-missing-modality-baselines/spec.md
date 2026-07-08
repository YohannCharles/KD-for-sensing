## ADDED Requirements

### Requirement: Local paper baseline modality and window boundary
AMBER full、RMBP-MM 和其它本地缺失模态论文 baseline 在本 change 覆盖的可训练入口中 MUST 使用 `seq_len=2`、`num_pred=1`。除 AMR-Net 论文三模态子集外，本地缺失模态 baseline MUST 将 enabled modalities 限制为 `image`、`radar`、`gps`、`lidar`，不得启用 `mmwave`、`csi`、historical beam 或 beam measurement 输入。

#### Scenario: 缺失模态本地配置窗口一致
- **WHEN** 用户加载 AMBER full 或 RMBP-MM local substitute 配置
- **THEN** data 和 model 配置 MUST 声明 `seq_len=2` 与 `num_pred=1`
- **AND** enabled modalities MUST 不超出 `image/radar/gps/lidar`

#### Scenario: claim 边界保持本地状态
- **WHEN** 文档、summary 或 claim row 描述本 change 覆盖的 AMBER full、AMR-Net 或 RMBP-MM baseline
- **THEN** 它们 MUST 标记为 local architecture reproduction、local substitute 或 local experimental baseline
- **AND** 缺少官方源码、官方 checkpoint、官方 split 和真实严格可比 metrics 时 MUST NOT 声称 official reproduction
