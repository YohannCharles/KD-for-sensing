# image-only-legal-crossroad-probe Specification

## Purpose
定义 MMW Town10 crossroad 上 image-only few-shot probe 的合法输入边界、运行矩阵、feature cache 和 eligibility reporting，用于隔离图像模态能力并防止 GPS、radio/path、beam_power 或 target_test label oracle 泄漏到 adaptation。
## Requirements
### Requirement: Image-only Hist probe 已退役
Image-only legal crossroad probe 中依赖 `configs/hist_beam/`、HiST variants、V8/V9 Hist heads 或 `kd-sensing-hist-beam-loso` 的路径 MUST 从当前支持面退役。

#### Scenario: Image-only Hist probe 配置不可运行
- **WHEN** 用户引用 `configs/hist_beam/image_only_legal_crossroad_probe.yaml`
- **THEN** 系统 MUST 报告配置已退役或不存在
- **AND** 系统 MUST 不构建 image-only HiST probe model

