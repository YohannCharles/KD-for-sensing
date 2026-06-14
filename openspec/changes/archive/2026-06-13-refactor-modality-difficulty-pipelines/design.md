## Context

当前仓库已经有几类“输入变难”的实现，但边界不统一：CSI degradation 在 `data/transform_ops/csi.py` 和 dataset 内部，LiDAR point dropout/jitter 在 LiDAR transform 路径，JEPA GPS shortcut benchmark 的 GPS/image perturbation、Scenario C preset、warnings 和 replay 字段集中在 `diagnostics/jepa_gps_shortcut_benchmark.py`。这些能力都服务于同一个研究问题：在 GPS 变 noisy、stale、low-rate 或 image 变弱时，验证 JEPA 视觉表征是否比 GPS shortcut 更稳。

后续如果继续把 GPS 和 image 难度写进各自 runner、dataset 或模型分支，会出现三个问题：训练与评估扰动不一致，benchmark 的 synthetic perturbation 无法复用到 supervised training，metadata 难以证明不同模型看到了同一 corrupted input。现有项目架构又明确要求职责拆分、轻量导入、配置解析与运行产物边界清晰，因此这个 change 应该抽出一个窄的 modality difficulty pipeline，而不是新增旧脚本或大型聚合层。

## Goals / Non-Goals

**Goals:**

- 建立统一 difficulty profile/schema/operator/pipeline，覆盖 GPS noisy、missing、delay、stride、async 和 image physical degradation。
- 让训练、验证、评估、JEPA benchmark 共用同一套 batch transform 与 seed/replay metadata。
- 保证 difficulty 只改变输入模态及输入可靠性 metadata，不移动 beam label、beam power、sample id 或 split。
- 保持现有 JEPA GPS shortcut benchmark manifest 语义兼容，同时把内部 perturbation 实现迁移到 shared pipeline。
- 保持 registry/config/dataset/runtime 轻量导入边界，不把 diagnostics 或模型重依赖带入配置加载。

**Non-Goals:**

- 不在本 change 中重新设计 JEPA 模型、pooler、loss 或 Stage 1 pretraining。
- 不引入真实物理渲染器或外部图像天气仿真依赖；第一版使用当前可测的 deterministic tensor transforms。
- 不把 difficulty profile 当成新模态，也不恢复 image motion profile、旧 KD、G2D、CRAF、MARF 或其它退役路径。
- 不提交真实 dataset、cache、checkpoint、训练输出或 benchmark 产物。

## Decisions

### 1. Difficulty 作为 batch-level pipeline，而不是 dataset 专属开关

实现上新增窄模块，例如 `kd_sensing.data.difficulty` 或 `kd_sensing.engine.difficulty`，提供 `DifficultyProfile`、`DifficultyOperatorConfig`、`DifficultyContext`、`DifficultyResult` 和 `apply_difficulty_pipeline(batch, profile, context)`。pipeline 在 dataset 已返回 flat sample、训练/评估提取 labels 之前运行；benchmark runner 也调用同一入口。

理由：GPS delay/stride/image degradation 都是“评估或训练条件”，不应该改写真实 dataset、split CSV 或 target provider。batch-level transform 更容易保证 target preservation，也能让 benchmark 与 training 共用同一实现。替代方案是把难度放进 dataset adapter，但会让 train/eval/benchmark 的 stage 选择、severity sweep 和 replay metadata 更难统一。

### 2. Operator 通过 registry 构建，但默认注册显式触发

新增 `DIFFICULTY_OPERATORS` registry 或难度专用窄 registry，内置 operator 分到 GPS、image、temporal 模块。`kd_sensing.registries` 仍保持轻量；配置加载或 pipeline 构建前显式调用默认 difficulty 注册函数。

理由：后续会不断添加新的 GPS 与 image 难度，registry 能让配置和 benchmark manifest 只引用 operator name，不需要主循环新增分支。替代方案是在一个大 `if suite_type` 中维护所有 operator；这正是当前 benchmark runner 的形态，扩展成本会继续上升。

### 3. Profile schema 同时服务 train/eval/benchmark

采用统一 profile 概念，包含 `id`、`operators`、`stage`、`splits`、`condition`、`severity/severities`、`seed`、`fallback`、`metadata`。训练配置可声明单个或少量 profile，evaluation/benchmark 可声明 severity sweep。现有 benchmark manifest 的 `perturbation_suites` 作为兼容输入，解析时映射到 difficulty profile。

理由：研究上需要 clean training、mild async training、GPS/image dropout training、evaluation-only sweep 成组比较。统一 schema 能保证 seed、severity、condition 命名一致。替代方案是训练配置和 benchmark manifest 各自维护一套字段，会导致同名 Scenario C 不一定产生同样输入。

### 4. Target preservation 是硬约束

pipeline 必须维护一个 target 字段 denylist/guard，禁止 operator 改写 `target_beam`、`beam_power`、soft target、auxiliary target、sample id 和 split metadata。GPS delay 的语义固定为 `G[k] -> historical G[src<=k]`，target 仍是当前 `y[k]` 或当前 prediction objective 的 target。

理由：用户要体现 JEPA 优势，关键是“当前视觉 sensing + 不可靠/滞后 GPS”下模型是否还能预测当前 beam；如果 target 跟着 GPS 移动，会变成另一个任务。替代方案是允许 operator 自由改 batch，但会引入隐性 label shift 和泄漏风险。

### 5. Metadata 采用可 replay 的 batch/operator provenance

每次应用 difficulty 时写出 profile digest、operator type、condition、severity、base seed、derived seed、sample ids、source index/timestamp、valid/stale/dropout mask summary 和 warnings。训练 run 写入 runtime metadata，benchmark 写入 `benchmark_manifest.json` 和表格列。

理由：论文图和模型可比性需要证明所有模型在同一 split、seed、severity 下看到了同一 corrupted input。替代方案只记录 human-readable condition 名称，无法复查 delay/stride/dropout 的实际采样。

## Risks / Trade-offs

- [Risk] batch-level transform 增加训练循环和评估循环的调用点。→ Mitigation：封装为单一 `apply_configured_difficulty(...)` helper，并在未配置 profile 时快速 no-op。
- [Risk] GPU/CPU tensor、numpy image 和不同 shape 的 batch 字段处理复杂。→ Mitigation：第一版只承诺当前 flat batch 中的 torch tensor 路径；其它格式必须显式跳过并写 warning。
- [Risk] difficulty registry 可能扩大轻量导入依赖。→ Mitigation：registry 对象轻量，默认 operator 注册函数延迟导入实现模块，并加 import-lightness 测试。
- [Risk] 旧 benchmark 输出列与新 provenance 字段并存会显得重复。→ Mitigation：保持现有表格列兼容，把新增 provenance 放到 manifest 和可选扩展列。
- [Risk] 训练时启用随机 difficulty 可能影响复现实验稳定性。→ Mitigation：seed 派生必须包含 profile/operator/condition/severity/split/sample id，训练 metadata 必须记录 digest 和 seed 规则。

## Migration Plan

1. 新增 difficulty schema、registry、内置 GPS/image operators 和 synthetic batch tests。
2. 在 config validation 中解析并校验 difficulty profiles，未配置时保持 no-op。
3. 在 training batch step 和 evaluation pass 中加入 no-op 默认的 difficulty hook，并记录 runtime metadata。
4. 将 `jepa_gps_shortcut_benchmark.py` 的 suite normalization 保留为兼容层，实际应用迁移到 shared difficulty pipeline。
5. 补充 benchmark manifest compatibility tests、Scenario C no-future-leak tests、metadata/digest tests 和 CLI smoke。

Rollback 策略：保留 benchmark manifest 的外部 schema 不变；如果 shared pipeline 回归失败，可在一个短期迁移窗口内让 benchmark runner 继续调用旧 helper，但不得新增新的专属 perturbation 分支。

## Open Questions

- difficulty 配置最终放在 top-level `difficulty`，还是 `data.difficulty` 加 `evaluation.difficulty` 两层，需要结合现有配置风格在实现时确定。
- 第一版是否让 training 支持 severity schedule，还是只支持固定 profile；建议先固定 profile，sweep 留给 evaluation/benchmark。
- image degradation 是否需要在 normalized tensor 空间还是 raw image 空间应用；第一版建议沿用当前 benchmark 的 tensor-space deterministic 变换，并在 metadata 记录输入空间。
