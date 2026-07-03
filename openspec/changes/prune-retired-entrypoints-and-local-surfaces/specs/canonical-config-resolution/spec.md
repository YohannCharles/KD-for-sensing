## ADDED Requirements

### Requirement: Generated experiment config families do not require entity YAML
规则化实验矩阵、seed sweep、night-grid、next-round queue 或本地 GPU 队列配置 MUST 优先由 base config、manifest 和 generator 表达。若实体 YAML 可由 generator 无损重建且不属于 canonical/current、paper/workflow reproduction 或 diagnostics manifest，它 MUST 从长期源码表面删除。

#### Scenario: Removable generated YAML
- **WHEN** generator 能重建实体 YAML 的 run name、seed、epoch、sampler、loss weights、missing pattern、dataset split、output boundary 和 critical overrides
- **THEN** 项目 MUST 保留 generator/manifest/base config
- **AND** 对应实体 YAML MAY 删除

#### Scenario: Retired path not regenerated
- **WHEN** generator 或 virtual config 解析规则处理实验矩阵
- **THEN** 它 MUST 不生成或接管 legacy KD、HiST/Hist、BGAM、viewer、Raymobtime、AMR-Net_gps_image 或 JEPA-MSAC retired path
- **AND** config migration guard MUST 继续 fail fast

### Requirement: Config generator has a small sanity check
保留的 config generator MUST 有 focused sanity check，覆盖 manifest 行、文件名/run name 和关键 resolved config 字段，且 MUST 不写入 `outputs/`、`logs/`、checkpoint 或真实训练结果。

#### Scenario: Generator sanity
- **WHEN** generator 更新 tracked config family 或 manifest
- **THEN** focused tests MUST 验证生成结果的核心语义
- **AND** 测试 MUST 使用临时目录或受控 config 输出路径
