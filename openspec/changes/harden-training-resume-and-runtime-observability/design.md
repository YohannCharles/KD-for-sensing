## Context

当前训练阶段顺序是先创建/复用 run 目录、写入 running status、构建 dataloader、拟合并保存 normalization artifact、写 `resolved_config.yaml`/`final_config.yaml`，再构建模型与 optimizer，最后才调用 `resolve_resume_checkpoint()`。因此 `training.resume: true` 且 `last.pth` 缺失时会在已经改动运行目录后静默返回新训练；显式 checkpoint 缺 optimizer、scheduler 或 epoch 时，通用加载器也会跳过缺失字段。

现有 checkpoint payload 只覆盖模型、optimizer、scheduler 和部分选择状态，没有保存全局 RNG、loader generator、AMP GradScaler、training extension、history/epoch logs。`best*.pth` 直接写目标文件，sidecar 虽原子写出但不校验 checkpoint 摘要；每个 payload 又复用全局 selection 字段，无法证明某个具体文件因何被选中。最终测试固定加载 `best.pth`，共享 validator 会先写通用 `metrics.json`，随后调用方才补 split 标签，导致自定义 selection、跨 run 或零剩余 epoch 的实际 checkpoint 和最终测试证据不可靠。

训练热路径还会对多个标量逐个 `.item()`/`.cpu()`，按 batch 等权平均指标，每轮验证后主动销毁 persistent worker，默认开启未同步的 `perf_counter` GPU 阶段计时并逐行同步写 CSV。Synthetic dataset 则持有一个共享可变 generator，使 `dataset[i]` 取值依赖此前访问顺序。

本 change 必须保持现有 package CLI、objective、split、normalization 和 ignored runtime artifact 边界；历史 checkpoint 仍可通过明确的 legacy migration 分支读取，但不能继续污染新 schema。

## Goals / Non-Goals

**Goals:**

- 让声明 resume 的运行在任何缺失、损坏或不兼容条件下尽早失败，不产生“其实从头训练”的假恢复。
- 让 current schema 的中断恢复能重建下一 batch 所需的训练状态，并明确区分 exact resume 与 legacy best-effort resume。
- 让 checkpoint 文件、sidecar、选择策略、最终测试和 run status 对同一个实际文件形成可验证 provenance。
- 消除默认训练热路径中不必要的设备同步、batch 等权偏差、worker 反复重建和默认 profiling 开销。
- 用 synthetic/fixture 测试证明恢复等价性、指标加权、产物原子性和访问顺序确定性。

**Non-Goals:**

- 不承诺 bitwise 跨 PyTorch/CUDA/cuDNN 版本、不同设备拓扑或不同 DataLoader worker 配置恢复。
- 不迁移、重写或提交现有 `outputs/`、`logs/`、checkpoint、cache 或真实数据。
- 不新增训练 CLI、真实实验队列、第三方依赖、分布式训练协议或任意配置差异自动放行机制。
- 不改变 model selection 与 final test 隔离、validation loss 有效观测加权等已生效契约。

## Decisions

### 1. 恢复使用两阶段只读预检

引入窄的 `ResumePlan`（或等价不可变结构），由配置规范化后的训练入口在创建/覆盖 run artifact 和构建模型、optimizer 前解析。第一阶段只做以下工作：

1. 解析 `training.resume`、源 checkpoint、源 run 目录和目标 run 目录；
2. 使用安全 checkpoint loader 在 CPU 上读取 payload；
3. 校验 resume role schema 和配置 fingerprint；
4. 生成跨 run、同 run、legacy/current schema 和预期 next epoch 的计划。

若 `resume=true`，默认路径必须是既有目标 run 的 `checkpoints/last.pth`，缺失立即报错，不创建时间戳目录也不写 running status。显式 checkpoint 可以来自另一 run；目标输出身份可以不同，但来源和目标必须同时记录。

第二阶段只构建用于 split identity 的数据描述和只读加载源 normalization artifact，不把新 artifact 写回源或目标目录。split/normalization fingerprint 通过后，系统才原子发布目标 run 的初始配置与状态，再构建模型、optimizer、scheduler、GradScaler 和 extension 并恢复状态。

备选方案是保留现有顺序，在 `restore_if_needed()` 中补一个异常。该方案仍会在发现失败前拟合统计量并覆盖运行产物，不能满足 fail-closed，故拒绝。

### 2. Resume role 使用版本化 schema，legacy 走显式迁移函数

current checkpoint 顶层记录 `checkpoint_schema_version`、`checkpoint_role`、`state_dict`、`optimizer`、`scheduler`、`epoch`、`runtime_state`、`resume_contract` 和该文件的 selection provenance。resume role 的 `optimizer`、`scheduler` 与 `epoch` 字段必须存在；当前训练没有 scheduler 时 `scheduler` 可以显式为 `null`，但不能省略。当前运行存在 scheduler 时，checkpoint 中的 `null` 也必须拒绝。

没有 current schema version 的历史 payload 由独立 `migrate_legacy_resume_payload()`（名称可等价）识别、校验和标注，不能继续在通用状态代码中使用嵌套 `dict.get()` 猜测字段。legacy 核心 role 字段仍必须完整；缺少新 `runtime_state` 时允许 best-effort 恢复，但 checkpoint load 记录必须包含迁移版本、warning 和 `trajectory_equivalence: false`。历史 `test_loss` 只允许该迁移函数在缺少 `best_val_loss` 时解释为旧 validation-loss alias；新 payload 不写 `test_loss`。

这样保留“历史 checkpoint 缺少新 metadata 不被无条件拒绝”的现有要求，同时使 current schema 缺字段 fail-closed。

### 3. `runtime_state` 在 epoch 边界形成一次一致快照

每个可恢复 checkpoint 保存以下版本化状态：

- Python `random`、NumPy、Torch CPU 和所有可见 CUDA device 的 RNG；
- 按 split 命名的 DataLoader generator，以及有状态 sampler/generator 的状态；
- GradScaler state（关闭 AMP 时也记录 disabled schema）；
- 每个 extension 的稳定 id、state schema version 和 `state_dict`，无状态 extension 必须显式声明 stateless；
- `TrainingState` 中 early stopping、selection catalog、history、epoch logs 和其它下一 epoch 需要的计数。

状态使用 `weights_only=True` 可读取的 tensor、标量、列表和 mapping 形式，不引入任意对象 pickle。所有 epoch 日志、extension hook 和 scheduler 更新完成后先冻结快照，再用同一快照发布该 epoch 产生的 `last`/`best` checkpoint，避免文件之间的 epoch 状态不一致。

恢复时先构建并加载模型、optimizer、scheduler、GradScaler 与 extension state，最后恢复全局 RNG 和 loader/sampler generator，且必须发生在创建下一次 iterator 之前。extension 缺少所需状态或 CUDA RNG 拓扑不兼容时，current exact resume 明确失败；legacy 路径则标记非等价。

备选方案只调用 `set_seed(seed + epoch)`。它不能还原 batch shuffle、dropout、AMP scale 或 extension 内部进度，故拒绝。

### 4. 不可变兼容性使用三类 canonical SHA-256 fingerprint

`resume_contract` 分别记录：

- config fingerprint：模型、objective、loss、optimizer、scheduler、AMP、数据、dataloader/sampler、训练 seed 和选择策略的 canonical 配置；
- split fingerprint：dataset family/protocol、domain inventory、实际 train/validation/test sample identity 或 effective indices、样本数和目标 schema；
- normalization fingerprint：训练拟合 artifact 的模态、feature mode、domain policy、有效样本数和已有稳定摘要。

hash 使用标准库 SHA-256 和稳定 JSON 序列化，并同时保留足以生成结构化 diff 的 canonical metadata，错误不能只显示两个 hash。

允许差异采用代码内封闭 allowlist，不接受用户提供任意 glob。首批只允许：`training.resume` 本身、`training.epochs` 增大且不低于 next epoch、目标 `output.dir/run_name/overwrite`、进度显示、TensorBoard 开关/legacy tag、日志频率和 timing profile。模型、数据/split、normalization、seed、worker/sampler、optimizer/scheduler、AMP 和 selection policy 的变化必须拒绝。跨 run 只放行目标输出身份，不放行训练语义。

resume 时复用 checkpoint 引用的训练 normalization artifact；不得先对当前数据重新拟合再覆盖。源 artifact 缺失、摘要不符或与 split/feature mode 不匹配时 fail-closed。

### 5. 所有 checkpoint 通过统一发布器原子写出并验证 sidecar digest

统一 checkpoint publisher 对 `last.pth`、`best.pth`、`best_top1.pth`、`best_<selection>.pth` 和 registry 副本执行：

1. 在同目录临时文件写 payload，flush/fsync 后 `replace`；
2. 计算最终 checkpoint 的 SHA-256 与 size；
3. 将 digest、schema version、checkpoint role 和逐文件 provenance 写入临时 sidecar，再原子 `replace`；
4. reader 只有在 checkpoint 存在、sidecar 完整且 digest 匹配时才把 current schema 文件视为已发布。

文件系统无法把 checkpoint 与 sidecar 两个路径作为单事务替换。因此发布顺序选择“checkpoint 后 sidecar”，并把 sidecar 作为完成标记；崩溃留下的无 sidecar checkpoint会被 current reader 拒绝，而不是误选。临时文件在成功或异常后清理。

registry copy 也复用同一发布器，不再先 `copy2` 到最终路径。legacy 文件允许缺 digest，但必须在 load provenance 中标记 unverified。

### 6. 逐文件 selection catalog 是最终 checkpoint 的唯一来源

每个 sidecar 使用同一 schema 记录 `checkpoint_role` 和 `selection` 对象：metric、mode、value、selected epoch、source run、是否为 final-test candidate。含义按文件固定：

- `best.pth`：objective/early-stopping selection；
- `best_top1.pth`：Top-1 selection；
- `best_<selection>.pth`：显式自定义 selection；
- `last.pth`：last-epoch role，不伪装为 best selection。

checkpoint payload 中保存 selection catalog，路径以 source run 为基准并附 digest。恢复计划把 catalog 作为候选状态一并恢复；跨 run 不要求复制源 checkpoint，但必须验证引用仍存在且 digest 匹配。

统一 `SelectedCheckpoint` resolver 决定 final test、run status 和返回值使用的文件：关闭 model selection 时选 `last`；未显式配置自定义 selection 时选 objective `best.pth`；显式配置 selection 时选与该策略匹配的文件。若恢复后 `start_epoch >= training.epochs`，没有新 epoch 也必须从恢复 catalog 或 resume 文件解析实际候选，不能假设目标目录新生成了 `best.pth`/`last.pth`。候选缺失或歧义时清晰失败。

### 7. 最终测试采用独立、一次性发布的 metrics artifact

共享 evaluation pass 增加“不由 validator 写通用文件”的调用方式；final test 先在内存中完成指标计算，再补齐以下 provenance：`evaluation_split: test`、model-selection split、requested/resolved selection、checkpoint path/role/digest/source run 和 objective metadata。完整对象随后原子写入 `final_test_metrics.json`，并以同一对象嵌入 `train_log.json` 与 final config。

final test 不覆盖 validation 或 standalone evaluate 所有的 `metrics.json`。run status 同时引用实际 selected checkpoint 和 final-test artifact。写出或测试失败时，不得留下带 `complete` 状态的半成品。

### 8. 训练指标按各自有效分母在设备侧聚合

Batch step 返回 detached 的 metric numerator/denominator tensor bundle，而不是让 recorder 对 total/task/auxiliary loss 和 accuracy 分别调用 `.item()`。Recorder 在设备侧累计每个指标自己的有效 sample/token 数；epoch 结果等于 numerator sum / denominator sum，不能按 batch 等权。最后一个小 batch、masked token 和不同 auxiliary target 的分母分别测试。

进度显示只在配置的刷新间隔把一个紧凑标量向量批量搬到 CPU；epoch 结束再做一次最终物化。关闭进度和 timing 时，常规 batch 热路径不进行日志用途的逐标量设备同步。显式 debug extension 自身要求的同步不在本 change 中伪装为零开销，但必须与默认路径隔离。

### 9. 跳过 validation 表示 unavailable，不复用为当前 epoch 观测

validation interval 跳过某 epoch 时，该 epoch 的 `val_*`、`validation_metrics` 和 `val_primary_metric` 使用现有 null/NaN unavailable 语义，并记录 `validation_ran: false`。可以在独立的 `last_observed_validation` provenance 中引用最近一次验证 epoch，但不能把旧值写成本 epoch 指标，也不能据此更新 checkpoint、scheduler-on-validation、patience 或 early stopping。

`last.pth` 在跳过验证的 epoch 记录 `validation_loss: null`，同时保留 selection catalog 中此前真实验证产生的 best 候选。

### 10. Timing 是显式 profile，默认路径不测量也不逐行写盘

`training.timing.enabled` 默认 `false`。启用时必须显式选择 profile：host wall-clock 只声明 host/data/end-to-end 范围；CUDA stage timing 使用 CUDA events，并仅在采样 batch 同步。未启用时不调用阶段 `perf_counter`、CUDA event、显存/进程探针，也不创建 timing 文件。

Timing row 先在内存缓冲，在 epoch 边界和 finalization 批量写入当前 `run_dir` 下的专属 artifact，避免共享父级 CSV 的跨 run 混写和每 batch open/write。异常 finalization 仍尽力 flush，但不能覆盖原始异常。

### 11. DataLoader 与 synthetic 数据使用稳定、可恢复的随机身份

train/validation/test 分别拥有从 experiment seed、split 名称和 dataset fingerprint 域分离得到的 generator；不得共享一个可变 generator。generator 和有状态 sampler 的状态按 split 进入 `runtime_state`，恢复发生在下一 iterator 创建前。persistent worker 在跨 epoch 使用时保持存活，只在整个 run finalization、loader 替换或异常退出时关闭；普通非 persistent iterator 继续由 DataLoader 自身管理。

Synthetic dataset 不再把一个共享 generator 存在实例上。每次 `__getitem__(index)` 用 base seed、split、dataset schema identity 和 index 的稳定 SHA-256 派生局部 seed，因此相同 index 内容一致，访问顺序、worker 数和先访问其它 split 不会改变样本。这里不使用进程随机化的 Python `hash()`。

## Risks / Trade-offs

- [Risk] current checkpoint 体积因 history、epoch logs 和 RNG state 增加。→ 只保存 epoch 级紧凑记录与状态，不复制 prediction artifact；focused test 监控 payload schema，后续若体积成为实测瓶颈再提出独立压缩 change。
- [Risk] crash 发生在 checkpoint replace 与 sidecar replace 之间会留下孤立文件。→ sidecar 是 current schema 完成标记；reader fail-closed，下一次同名发布可原子覆盖，cleanup manifest 可识别孤立临时/未验证文件。
- [Risk] legacy checkpoint 无法达到轨迹等价。→ 单独迁移分支、warning 和 `trajectory_equivalence: false`，不得把 legacy smoke 结果宣称为 exact resume。
- [Risk] 严格 fingerprint 会拒绝过去被默许的配置漂移。→ 错误输出逐字段 diff；只对确定不影响训练轨迹的字段维护最小 allowlist，新增放行必须走代码与测试评审。
- [Risk] CUDA event profiling仍会在采样点同步并改变性能。→ 默认关闭，profile metadata 明确标记观测开销；正式吞吐比较必须使用相同 profile/interval。
- [Risk] extension state contract 暴露尚未实现序列化的扩展。→ extension 必须声明 stateless 或实现版本化 state；focused tests 覆盖所有 current extension，不能静默跳过。

## Migration Plan

1. 先增加 schema/fingerprint/selection 数据结构和纯函数测试，不改变训练写出。
2. 将 checkpoint 保存统一到原子 publisher，并为所有 role 生成 digest sidecar；保留 legacy reader。
3. 调整训练 phase 顺序，加入两阶段 `ResumePlan` 与 staged normalization/split compatibility gate。
4. 接入 runtime state capture/restore、extension/GradScaler/loader generator 状态，并增加 uninterrupted 与 interrupted-resume 等价测试。
5. 统一 final checkpoint resolver 和独立 `final_test_metrics.json` 写出，再移除新 payload 的 `test_loss`。
6. 最后切换设备侧加权 recorder、validation skip、worker 生命周期、timing profile 和 synthetic index 确定性；运行 focused、quick 与 full 回归。

实现阶段只在测试临时目录生成 checkpoint。若回归失败，回滚到上一步源码即可；不对历史本地产物做原地迁移。

## Open Questions

无阻塞问题。首版 allowlist、schema version 和 timing profile 名称在实现时固化为常量并由 characterization test 保护；不得通过宽松用户配置扩展。
