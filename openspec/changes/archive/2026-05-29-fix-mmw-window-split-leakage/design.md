## Context

当前 MMW Town10 数据准备会先从连续 frame 片段生成重叠滑窗，再对 `seq_index` 随机 80/20 切分。对 `seq_len=5`、`pred_len=6` 的 `l5p6` 产物，任意相邻两个窗口最多共享 10/11 帧；本地排查显示 test window 几乎都能在 train 中找到高度重叠窗口，并且大量未来标签序列已在 train 中出现。

这会把验证变成邻帧记忆测试，而不是严格的同场景泛化或跨场景验证。它也会污染 MMW quick validation summary：即使模型、日志和 CSV 路径都没有指错，曲线仍会因为 split 协议过宽而显得异常好、异常相似。

约束：

- 不修改 MMW 模型、loss 或指标定义。
- 不迁移或提交本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint。
- 保持现有公开训练/评估入口可用；新增协议通过数据准备/split utility 和 metadata 暴露。
- 旧随机窗口 split 不再作为准备流程或公开 split builder 支持的协议；已有产物只通过 metadata eligibility 保守处理，不作为主结论协议。

## Goals / Non-Goals

**Goals:**

- 让 MMW Town10 默认 sequence split 成为 group-safe 协议，避免 train/test 共享相邻滑窗上下文。
- 为 split 输出可机器读取的协议 metadata 和泄漏诊断，能解释当前曲线是否来自 strict split。
- 让训练、评估和 summary 产物记录 split 协议，并能过滤 unknown 或 strict-ineligible 结果。
- 为 scenario-LOSO 的 target_adapt/target_test 引入时间邻近隔离，避免 few-shot target 样本污染 target_test。
- 提供 focused tests，覆盖 group 不跨 split、guard band 生效、unsupported strategy 和运行产物记录。

**Non-Goals:**

- 不重新设计 HiST-Beam、单模态 backbone、fusion 结构或 KD/adaptation loss。
- 不改变 Top-K、ADBA、metric horizons 的数学定义。
- 不自动重跑或删除已有 `l5p6` 实验产物；缺失 metadata 或 `strict_validation_eligible=false` 的产物通过 summary 标记不可作为 strict 主结论。
- 不要求一次性支持所有 MMW condition/town 数据下载；本 change 只收紧已准备数据的 split 协议和记录。

## Decisions

### 1. 默认 split 从 random-window 改为 group-safe time-block

新的默认策略应按连续片段内的稳定 group 分配 train/test。推荐实现为：

1. 先按 `(scenario, agent, contiguous_segment_id)` 找连续 frame 片段。
2. 在每个片段内生成不重叠 time blocks，block 长度可配置，默认必须大于 `seq_len + pred_len`。
3. 在 train/test block 之间保留 guard band，默认至少 `seq_len + pred_len - 1` 帧。
4. 只生成或保留完全落在同一 split block 内的窗口。

这样做比“生成所有滑窗后随机分配”更直接，因为相邻窗口天然高度重叠；事后仅检查 sample id 无交集不够。

备选方案是按 agent 整体切分。它最干净，但只有三个 CAV agent 时样本量和标签分布容易不稳。time-block split 在防泄漏和样本覆盖之间更平衡。

### 2. metadata 必须记录协议和泄漏诊断

每个 split metadata 写出：

- `split_strategy`: 当前公开准备协议为 `group_safe_time_block`
- `split_protocol_version`
- `group_key_fields`
- `block_size_frames`
- `guard_band_frames`
- train/test window count、group count、frame range
- train/test frame overlap count
- test window 与 train window 的最大帧重叠分布
- test future label sequence 在 train 中复用的比例
- `strict_validation_eligible`
- `eligibility_reasons`

理由：光写 CSV 路径不能判断协议是否可比。把诊断写进 metadata 后，训练日志、评估报告和 summary 可以直接消费，不需要临时扫描 CSV。

### 3. 旧 random-window 不作为公开 split 协议

旧的随机滑窗 train/test 分配不再作为 MMW preparation 或公开 split builder 的兼容路径。实现只需要支持新的 group-safe 协议；其它 strategy 值按普通 unsupported strategy 处理，不为旧方法提供专门生成、专门警告或专门 metadata 分支。

已有本地产物如果缺少 split metadata，或 metadata 显示 `strict_validation_eligible=false`，训练、评估和 summary 继续按通用 unknown/ineligible split 处理，并给出重新生成 group-safe split 的修复提示。

理由：继续保留旧协议入口会让后续实验再次复用泄漏 split；但为某个历史 strategy 名称写专门报错/警告会扩大兼容面，不利于收敛协议边界。

### 4. scenario-LOSO target_adapt/target_test 也使用 group-safe 切分

跨场景协议中 sample id 无交集仍保留，但不再足够。target scenario 内部拆分 target_adapt/target_test 时同样必须以 group/time-block 为单位，并保留 guard band。few-shot labeled target subset 只能从 target_adapt 采样，且不得与 target_test 共享重叠窗口上下文。

理由：在滑窗数据上，只检查 sample id 会漏掉邻帧泄漏；few-shot target_adapt 如果和 target_test 相邻，会让 adaptation 指标偏乐观。

### 5. 运行产物消费 split metadata，而不是只靠文件名

训练和评估 runtime 应尽量读取 split metadata，并在 `final_config.yaml`、`train_log.json`、`metrics.json`、runtime metadata 或等价报告中记录核心字段。summary/conclusion 应把 `strict_validation_eligible=false` 的 run 排除出主结论或明确标记为 debug/sanity。

理由：文件名如 `l5p6` 只能表达窗口长度，不能表达 split 是否严格。结果可比性必须由 metadata 证明。

## Risks / Trade-offs

- 旧曲线数值会显著下降或收敛变慢 -> 这是协议修正后的预期现象；在 summary 中区分 strict、unknown 与 ineligible split，避免混比。
- group-safe split 可能造成标签分布不均 -> metadata 记录 beam label 分布，并在实现中允许 deterministic retry 或 stratified block assignment。
- guard band 会减少样本数 -> block 大小和 guard band 可配置，但默认必须满足最小隔离要求。
- 不同 scene 的 frame 编号范围和 agent 片段长度不同 -> group builder 使用实际连续片段和 agent，不依赖全局 frame id 对齐。
- 旧脚本仍可能复用已存在的随机滑窗 CSV -> preflight/prepare 阶段必须检查 split metadata；缺失或不满足 strict eligibility 时给出警告或生成新 split tag，而不是静默跳过。

## Migration Plan

1. 新增 group-safe split utility 与 leakage diagnostics helper，先通过小型 fixture 测试。
2. 更新公开 MMW split builder 和 Town10 preparation，使默认策略写出新的 strict split metadata。
3. 移除旧 random-window 生成兼容路径；旧 `l5p6` 文件不自动删除，但推荐以新 split tag 重新生成 group-safe split。
4. 更新 `run_mmw_sunny_modal15_l5p6_h246.sh` 或后续推荐脚本，默认使用 strict split tag，例如 `l5p6_group_safe`。
5. 更新训练/评估 runtime metadata 与 summary 过滤逻辑，确保 unknown 或 strict-ineligible run 不进入 strict 主结论。
6. 用 focused tests 验证协议，再手动选择是否重跑 MMW 实验。

Rollback 策略：如 group-safe split 生成出现阻塞，应修复 group-safe builder 或临时使用独立实验分支排查，不在主分支恢复旧随机滑窗协议入口。

## Open Questions

- 默认 block size 采用固定帧数还是按每个连续片段比例自适应，需要结合三个 Town10 scene 的有效 frame 数做一个小型统计后定值。
- 是否需要对 beam label 分布做 stratified block assignment，还是先记录分布并保持 deterministic random block assignment。
- 旧 `l5p6` tag 是否保留为历史产物目录，还是强制新 strict tag，避免目录名相同但协议变化。
