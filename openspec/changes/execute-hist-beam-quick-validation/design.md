## Context

`add-hist-beam-cross-scene-adaptation` 已归档并合入主规格后，仓库已经具备 HiST-Beam 模型、LOSO fold、target adapt/test split、few-shot sampling、adapter/full fine-tuning 策略、metrics/predictions writer 和 `kd-sensing-hist-beam-loso` CLI。当前缺口集中在执行闭环：`src/kd_sensing/cli/hist_beam_loso.py` 的 `--execute` 仍然只写 `loso_plan.json`，并返回 `execution.status: planned`，没有实际调用 source training、source-only evaluation、target adaptation、adapted evaluation 和 summary 生成。

本设计面向《跨场景自适应方案.md》的快速验证阶段，而不是重新实现 HiST-Beam 模型。上下文包括已归档 change `add-hist-beam-cross-scene-adaptation`、主规格 `cross-scene-loso-workflow` 和 `hist-beam-cross-scene-adaptation`。所有项目相关 Python 命令、测试和验证继续通过 `conda run -n kd_mm_beam ...` 执行。

## Goals / Non-Goals

**Goals:**

- 让 `kd-sensing-hist-beam-loso --execute` 真正执行 LOSO quick validation stages，而不是仅返回 planned。
- 支持最小执行矩阵：target scene 先 `34`，再扩展 `33/32/31`；variants 至少 `v0_flat`、`v3_decoupled`、`v4_adapter`、`v5_adapter_proto`、`v6_full_finetune`；budgets `0/10`；seed `0`。
- 将默认 CLI `quick_smoke` 与完整方法验证配置拆开：`quick_smoke` 必须是短 epoch、小矩阵、小数据比例的资源探针；完整 `quick_validation` 才使用 `data.dataset.portion: 1.0` 和 `training.epochs: 40`。
- 在执行前提供明确 preflight，缺少本地数据、CSV、启用模态资源或不可写输出目录时失败，并指出具体缺失项。
- 每个 run 产出 `metrics.json`、predictions、配置快照、fold/split/sampling metadata；adaptation run 记录 trainable ratio 和 adaptation time；长 stage 在开始时和每个 epoch 后写出进度 metadata。
- 矩阵结束后输出 LOSO summary CSV/JSON 和一份简短验证结论，直接回答 adapter/prototype 是否优于 source-only 和 full fine-tuning。
- 用户中断或 stage 失败时仍尽量写出 partial summary，并把未启动 run 标记为 missing/inconclusive，而不是让输出目录只有 plan 和空 stage 目录。
- 增加执行级测试，覆盖计划模式、执行模式、preflight failure 和最小数据 smoke。

**Non-Goals:**

- 不重写上一份 HiST-Beam 模型、loss、prototype 或 LOSO split 设计。
- 不把真实 DeepSense6G 数据、训练输出、日志、checkpoint 或临时验证产物纳入源码。
- 不新增根目录长期训练脚本；继续使用 `kd_sensing.cli` 下的 console script。
- 不要求单元测试跑完整 4 fold 大矩阵；测试必须使用最小 fixture、极小 epoch 或可注入 runner。
- 不把 budgets `5/20/50`、多 seed、LiDAR 主实验或论文完整对比纳入本次闭环。

## Decisions

### Decision 1: 将 CLI 分成 plan-only 与 execute 两条明确路径

`run_hist_beam_loso` 继续先构建 `loso_plan.json`。当未传 `--execute` 时，只写计划并返回 `mode: plan_only`。当传 `--execute` 时，必须进入 execution pipeline，并在返回 payload 中记录 `execution.status` 为 `completed`、`failed` 或 `partial_failed`，不得再使用 `planned` 表示已执行模式。

备选方案是把计划生成和执行拆成两个 CLI。放弃原因是现有入口和 specs 已经把 `--execute` 作为语义开关，保留一个入口更符合用户当前工作流。

### Decision 2: 增加窄的 LOSO execution runner，而不是把执行逻辑塞进 argparse 层

新增或抽出包内窄模块，例如 `kd_sensing.engine.hist_beam_loso_execution` 或 `kd_sensing.cli.hist_beam_loso_execution`，负责：

- 执行 preflight。
- 遍历 plan runs。
- 为每个 stage 调用训练、验证、adaptation 和 summary helper。
- 记录 stage result、artifact paths、duration、失败原因和 checkpoint 复用行为。

CLI 层只负责解析参数、加载配置、写计划和调用 runner。

备选方案是直接在 `hist_beam_loso.py` 中串联所有 stage。放弃原因是该文件已经承担 CLI 和 plan 构建，继续膨胀会让测试和后续 resume/partial failure 处理困难。

### Decision 3: 执行 stage 复用现有训练/验证/输出 helper

source training 应复用现有配置驱动训练入口或 trainer helper，生成标准 run directory、`metrics.json` 和 checkpoint。source-only 与 adapted evaluation 应复用现有 validator/evaluator 和 `write_hist_beam_predictions`。adapter/full fine-tuning 阶段复用 `hist_beam_adaptation`、prototype helper 和 HiST loss 训练能力。

备选方案是为 quick validation 写独立训练循环。放弃原因是会绕开现有 config、artifact、trainer 和 evaluation 契约，也会增加与单场景训练路径分叉的风险。

### Decision 4: preflight 在训练前集中失败

执行前先检查整张矩阵所需的 scene data root、CSV、启用模态资源、输出目录写权限和配置合法性。缺失数据时以结构化错误返回或抛出明确异常，错误必须包含 target scene、source scenes、variant、budget、seed 和缺失路径/资源。preflight 失败不得创建看似成功的 summary。

备选方案是让每个 stage 自然失败。放弃原因是训练失败信息通常较晚且不聚合，用户很难快速区分“数据没准备好”和“模型训练错误”。

### Decision 5: quick validation matrix 通过配置和 CLI 覆盖共同控制

默认 quick config 表达最小矩阵：先 `target_scene: 34`，variants `v0_flat/v3_decoupled/v4_adapter/v5_adapter_proto/v6_full_finetune`，budgets `0/10`，seed `0`。DeepSense6G 上默认 `data.dataset.portion: 1.0`，因为 `portion` 主要用于大数据集抽样，不应在本 quick validation 默认启用；默认 `training.epochs: 40`，避免 1 epoch smoke 造成模型塌缩并污染方法结论。CLI 的 `--target-scene`、`--variants`、`--budgets`、`--seeds` 继续允许缩小或扩展矩阵。扩展到 `33/32/31` 时使用同一个 runner，不增加新入口。

备选方案是在 CLI 中硬编码 target scene 34 和矩阵。放弃原因是会降低 smoke、resume 和后续完整 LOSO 的可配置性。

### Decision 6: summary 与 conclusion 由统一 writer 生成

矩阵完成后，runner 将每个 stage 的 run record 交给 summary writer，输出：

- `loso_summary.json`
- `loso_summary.csv`
- `quick_validation_conclusion.json` 或等价 Markdown/JSON

结论 writer 对同一 target scene、budget、seed 下的 `v4_adapter`、`v5_adapter_proto`、`v3_decoupled` source-only 和 `v6_full_finetune` 做横向比较，至少报告 Top-1、Top-3、Top-5、coarse accuracy、fine accuracy、trainable ratio、adaptation time。缺失 run 必须标记为 `missing` 或 `inconclusive`，不得用 0 补齐。

备选方案是只输出原始 summary，让人手动判断。放弃原因是本 change 的目标就是形成快速验证闭环，必须自动产出可读判断。

### Decision 7: 测试使用可注入 stage runner

执行级测试不应启动真实大规模训练。runner 应允许注入 fake stage executor 或使用最小 fixture 配置，让测试可以验证 `--execute` 的语义、preflight failure、artifact 写出和 summary 生成。真实训练路径由窄集成测试或人工 quick run 验证。

备选方案是在单测中跑真实模型训练。放弃原因是速度和本地数据依赖不可控，会让 CI 和开发迭代不稳定。

### Decision 8: 将 resource smoke 与 method quick validation 拆开

`configs/hist_beam/quick_smoke.yaml` 只保证真实数据路径、模型构建、训练循环和 metadata 写出能快速走通。它默认使用 target scene `34`、单个 `v0_flat` run、budget `0`、seed `0`、短 epoch、小 batch 和小数据比例，并可通过 `--max-runs` 进一步截断矩阵。CLI 默认配置继续指向 `quick_smoke`，避免用户无意中启动 10 个长训练 run。

完整方法验证使用 `configs/hist_beam/quick_validation.yaml`：target scene `34`、variants `v0_flat/v3_decoupled/v4_adapter/v5_adapter_proto/v6_full_finetune`、budgets `0/10`、seed `0`、`data.dataset.portion: 1.0`、`training.epochs: 40`。如果要扩展到 `33/32/31`，仍通过同一 CLI 的 target scene 参数或配置完成。

备选方案是继续让 `quick_smoke.yaml` 同时承担资源探针和方法验证。放弃原因是真实运行中 40 epoch × 10 run 会让用户长时间看不到任何结论，且手动中断时难以定位卡在哪个 stage。

### Decision 9: stage 开始即落盘，并在中断时写 partial summary

runner 必须在 run/stage 开始时立即写 `metadata.json`，stage 内部的 source training/adaptation 还应写 `progress.jsonl` 和 latest progress 快照。捕获 `KeyboardInterrupt` 时，当前 stage 标记为 failed，当前 run 标记为 failed，尚未启动的 planned runs 标记为 missing；随后写 `loso_summary.json`、`loso_summary.csv` 和 `quick_validation_conclusion.json`。无法捕获的系统级 `SIGKILL` 不保证写最终 summary，但 stage-start metadata 仍能说明最后进入的位置。

备选方案是只在矩阵结束后写 summary。放弃原因是人工 quick run 的资源和时长不可完全预测，结束后才落盘会让排障成本过高。

### Decision 10: few-shot 标签解析复用 DeepSense6G beam-power 语义

DeepSense6G 的 `beamN`/`future_beamN` CSV 字段在本仓库中通常不是整数标签，而是 64 维 beam-power 文件路径，真实 beam label 是该 power vector 的 `argmax`。few-shot sampler 必须优先读取 `future_beam_labelN`/`beam_labelN` 显式标签；若不存在，则对 `future_beamN`/`beamN` 路径读取 power vector 并取 `argmax`。sampling manifest 记录 `label_source`，用于追踪 few-shot 覆盖和失败定位。

备选方案是在 LOSO adapter 里直接从 dataset `__getitem__` 拉标签再采样。放弃原因是会提前触发图像/雷达/GPS 加载，采样阶段不应承担完整 batch I/O 成本。

### Decision 11: quick validation 显式声明 adaptation 超参

完整 `quick_validation` 配置声明 `hist_beam.adaptation.epochs`、`entropy_weight` 和 `prototype_weight`。这避免 0-label adaptation 因默认权重为 0 而不更新，也让 v4/v5/v6 的行为在配置中可审计。代码默认也采用方案中的轻量无标签设置：entropy `0.01`，prototype `0.1`。

备选方案是继续依赖代码默认。放弃原因是实验结论应能从运行配置复现，不能靠隐含默认解释。

## Risks / Trade-offs

- [Risk] 现有 trainer/evaluator API 不适合逐 stage 调用 → Mitigation: 先增加最小 adapter wrapper，只封装配置改写、run_dir 解析和 artifact 收集，不改默认训练路径。
- [Risk] 本地 DeepSense6G 31-34 数据目录命名或模态文件缺失 → Mitigation: preflight 明确列出 scene、CSV 和模态资源缺失项，并支持 CLI 缩小矩阵先跑 scene 34。
- [Risk] source checkpoint 复用和 overwrite/resume 语义混乱 → Mitigation: 每个 run record 记录 checkpoint path、reuse decision、overwrite flag、stage status 和 artifact paths。
- [Risk] 部分 stage 失败导致 summary 不完整 → Mitigation: summary 允许 `partial_failed`，但必须标记缺失 run 和失败原因；quick conclusion 对缺失比较输出 `inconclusive`。
- [Risk] adapter/prototype 与 full fine-tuning 初始 checkpoint 不一致 → Mitigation: V4/V5/V6 必须记录同一 V3 source checkpoint 来源；不一致时 comparison 标记不可比。
- [Risk] smoke fixture 与真实数据路径差异过大 → Mitigation: preflight 和 runner 测试分层，fixture 只证明执行闭环；真实路径检查由明确的 preflight error 和人工 quick run 覆盖。
- [Risk] 默认 `quick_smoke` 太重导致用户手动中断仍没有结论 → Mitigation: 默认配置改为资源探针，完整方法验证放到显式 `quick_validation` 配置；runner 写 running metadata 和 partial summary。

## Migration Plan

1. 保留现有 plan-only 行为，先为 `--execute` 增加 runner skeleton、stage status 和 preflight。
2. 串联 source training、source-only evaluation、target adaptation、adapted evaluation 和 artifact 收集。
3. 增加 quick smoke 与 quick validation 两层配置，确保 target scene 34 可单独执行，并能扩展 33/32/31。
4. 增加 summary CSV/JSON 和 quick validation conclusion writer。
5. 增加 running metadata、interrupt partial summary、`max_runs` 和资源探针配置 tests。
6. 增加 tests：`--execute` 语义、无数据 preflight failure、最小 fixture smoke summary。
7. 运行 OpenSpec strict validate 和相关 pytest：
   - `openspec validate execute-hist-beam-quick-validation --strict`
   - `openspec status --change execute-hist-beam-quick-validation`
   - `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py -q`
   - 必要时补跑 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`

回滚策略：`--execute` runner 是现有 CLI 的执行分支；如出现问题，可临时禁用 execution 分支并保留 plan-only，不影响既有单场景训练、评估和 HiST-Beam 模型注册。

## Open Questions

- source training、evaluation 和 adaptation stage 是直接调用现有 Python helper，还是先复用 console script 子进程以保持运行边界，需要在实现时根据当前 trainer API 决定。
- `quick_validation_conclusion` 最终采用 JSON、Markdown，还是二者都写；本 change 至少要求机器可读 JSON。
- smoke fixture 是否已有可复用最小 DeepSense6G 样本；若没有，需要构造只覆盖启用模态和 CSV 解析的最小临时数据。
