## Context

当前项目已归档并启用了 `m2beamllm-modality-encoders` capability，代码中包含独立的 `m2beamllm_encoders.py`、四类单模态 M2BeamLLM 注册名、fusion 的 `encoder_profile: m2beamllm` 分支、`configs/m2beamllm/` 示例配置、README 说明，以及覆盖该实验路径的测试。用户明确反馈该模态编码器效果不好，希望删除相关内容以精简项目。

这次变更是一次退役与清理，不是替换为新的编码器方案。实现必须保证现有默认 image、radar、GPS、LiDAR、mmWave 和 fusion 模型仍按原注册名构建，并且训练、评估、KD 流程不需要新增迁移分支。

## Goals / Non-Goals

**Goals:**

- 删除 M2BeamLLM encoder 实现模块和所有直接引用，避免无效实验路径继续进入构建、测试和文档。
- 删除 `m2beamllm_*` 模型注册名与 fusion `encoder_profile: m2beamllm` 行为。
- 删除 M2BeamLLM 示例配置和 README 说明，防止用户继续引用已退役入口。
- 清理 M2BeamLLM 专用测试，并保留或补充默认模型构建回归验证。
- 如某些数据预处理分支只服务于该退役实验路径，则同步删除；如仍被其它正式能力依赖，则保持不动并移除 M2BeamLLM 命名入口。

**Non-Goals:**

- 不新增替代编码器或新的模态实验 profile。
- 不修改现有默认模型架构、GRU/KD/eval 输出契约。
- 不删除 mmWave、标准 LiDAR BEV、GPS-Rel-Polar、标准 radar/image 处理路径。
- 不做大规模配置体系重构；只移除与 M2BeamLLM encoder 相关的入口和引用。

## Decisions

1. 采用直接删除而不是隐藏开关。

   理由：用户目标是项目精简，隐藏开关仍会保留无效代码、依赖和测试维护成本。直接删除可以让未知注册名错误尽早暴露，提醒外部配置迁移回标准模型。

   备选方案：保留代码但标记 deprecated。该方案适合短期兼容，但不符合“删除相关内容和代码”的目标。

2. 单模态模型文件恢复到只包含标准注册名。

   `image.py`、`radar.py`、`gps.py`、`lidar.py` 中应移除 `M2BeamLLM*` import、`m2beamllm_*_teacher/student` 注册类、`m2beamllm_pretrained` 等只为该路径服务的构造参数。标准 `image_teacher/student`、`radar_teacher/student`、`gps_teacher/student`、`lidar_teacher/student` 行为保持不变。

3. Fusion 删除 profile 分支，保留默认多模态组合能力。

   `FusionModalityNet` 和 `StudentModalityNet` 应移除 `encoder_profile` 的 M2BeamLLM 选择逻辑，以及 image/radar/GPS/LiDAR 分支对 `M2BeamLLM*Encoder` 的依赖。若仍保留 `encoder_profile` 参数用于向后兼容，应拒绝非空或未知 profile，并确保 `m2beamllm` 不再可用；若没有其它 profile 使用，则优先删除该参数以精简接口。

4. 配置与文档同步删除。

   删除 `configs/m2beamllm/` 能避免示例配置继续误导用户。README 中 M2BeamLLM Encoder 对照章节应移除，避免文档承诺已经删除的注册名、profile、LiDAR histogram 和 GPS min-max 入口。

5. 测试从验证 M2BeamLLM 改为验证退役结果与默认路径。

   删除 `tests/test_m2beamllm_encoders.py` 中针对 M2BeamLLM shape、注册、fusion profile、示例配置的正向测试。根据现有测试布局补充轻量回归：默认单模态和 fusion 注册名仍能构建；引用 `m2beamllm_*` 或 `encoder_profile: m2beamllm` 时应失败。

## Risks / Trade-offs

- 外部实验配置引用 `m2beamllm_*` 会失败 → 在 README 清理或迁移说明中明确改用标准注册名和标准配置。
- M2BeamLLM 专用数据预处理函数可能与其它能力共享文件 → 实现时先用 `rg` 确认引用，只删除确认为专用且无正式 spec 依赖的分支。
- 删除 tests 后回归覆盖下降 → 用默认模型构建和未知注册名失败测试覆盖关键风险。
- `encoder_profile` 参数如果直接删除，旧 fusion 配置会报构造参数错误 → 这是符合退役目标的 breaking change，但错误应保持可诊断。

## Migration Plan

1. 搜索 `m2beamllm`、`M2BeamLLM`、`encoder_profile` 的所有引用，分类为模型、fusion、配置、测试、文档和数据预处理。
2. 删除模型实现和注册入口，保证 `src/kd_sensing/models/m2beamllm_encoders.py` 不再被 import。
3. 删除 fusion 的 M2BeamLLM profile 分支，确认 mmWave 和默认 feature extractor 路径不变。
4. 删除 `configs/m2beamllm/` 和 README 对照章节。
5. 删除或更新测试，运行 `conda run -n kd_mm_beam pytest` 中与模型构建相关的最小回归测试。
6. 如需回滚，可从版本控制恢复本变更删除的模块、配置和 README 章节；不需要数据迁移。
