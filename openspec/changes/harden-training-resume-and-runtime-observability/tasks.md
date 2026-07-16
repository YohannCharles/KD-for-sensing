## 1. 基线与 characterization

- [x] 1.1 记录 `git status --short`、`openspec list --json`、本 change 的 apply instructions 和当前 focused baseline；运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_training_io_run_metadata.py tests/test_training_io_dataset_workflow.py tests/test_checkpoint_security.py -q`，只记录既有失败且不改动本地产物。
- [ ] 1.2 为 A01（`resume=true` 缺 `last.pth`）、A02（resume role 缺 optimizer/scheduler/epoch）、A03（完整 runtime state）和 A04（预检/fingerprint/allowlist）增加 fail-first synthetic/fixture characterization tests。
- [ ] 1.3 为 A05（跨 run/零剩余 epoch）、A06（best checkpoint 非原子）、A07（sidecar 无 digest）、A08（逐文件 selection provenance）、A09（过时 `test_loss` 字段拒绝）、A10（final test metrics 覆盖）和 A11（自定义 selection 未实际加载）增加临时目录 artifact tests。
- [ ] 1.4 为 A12（逐标量设备同步）、A13（validation worker 每轮关闭）、A14（默认 timing 与共享同步写）、A15（训练指标 batch 等权）、A16（跳过 validation 复制旧值）和 A17（synthetic 共享 generator）增加无真实数据的 focused tests。

## 2. Resume 预检与不可变契约

- [ ] 2.1 实现 A01 的只读 resume path preflight：在创建/覆盖 status、resolved/final config、normalization、model 或 optimizer 前解析同 run `last.pth`/显式路径，缺失时包含路径报错且不得返回 fresh-training sentinel。
- [ ] 2.2 实现 A02 的版本化 resume role validator：`optimizer`、`scheduler`、`epoch` 必须存在，scheduler 启用状态必须兼容，错误包含 role/path/字段，`training.start_epoch` 不得补 resume 缺失 epoch。
- [ ] 2.3 实现 A04 的 `ResumePlan` 或等价结构，区分 source/target run、current schema、next epoch 和 cross-run，并调整 training phases，确保校验通过前没有目标 run 的可变 artifact 写出。
- [ ] 2.4 实现 A04 的 canonical config/split/normalization SHA-256 fingerprint、结构化 diff 和封闭 allowlist；只允许合法增加总 epoch、目标 output identity 和 progress/TensorBoard/log/timing 等运行控制差异，拒绝用户自定义宽松通配。
- [ ] 2.5 调整 normalization/split 恢复顺序：resume 必须先验证并复用源训练 artifact，不得先从当前 train/validation/test 重拟合或覆盖；增加 artifact 缺失、摘要不符、sample identity 和 feature/domain policy 漂移测试。

## 3. Runtime state 等价恢复

- [ ] 3.1 实现 A03 的 safe-serialization runtime-state helper，捕获/恢复 Python、NumPy、Torch CPU 和所有可见 CUDA RNG，并验证 current schema 的字段与设备拓扑。
- [ ] 3.2 实现 A03 的按 split DataLoader generator 与有状态 sampler state 捕获/恢复；状态必须在下一 iterator 创建前恢复，runtime metadata 记录稳定 generator identity。
- [ ] 3.3 实现 A03 的 GradScaler、TrainingState history/epoch logs、early-stopping/selection state 序列化与恢复；关闭 AMP 时也写明确 disabled state。
- [ ] 3.4 为所有 current training extension 增加稳定 id、state schema 和 `state_dict`/`load_state_dict` 或显式 stateless 声明；current exact resume 遇到缺失/未知 extension state 必须失败。
- [ ] 3.5 将 epoch checkpoint 快照移动到 epoch log、extension hook 和 scheduler 状态完成之后，保证同一 epoch 的 last/best 文件使用一致 runtime state。
- [ ] 3.6 使用 `conda run -n kd_mm_beam pytest` 增加并运行连续 N epoch 与 K+resume 到 N 的 deterministic fixture 等价测试，比较下一 batch 顺序、模型/optimizer/scheduler/scaler/extension、history、epoch logs 和 selection provenance；非 current schema case 必须拒绝。

## 4. Checkpoint 原子性、摘要与 schema

- [ ] 4.1 实现 A06 的统一 checkpoint publisher，让 `last.pth`、`best.pth`、`best_top1.pth`、`best_<selection>.pth` 和 registry 副本都经同目录临时文件、flush/fsync 和 atomic replace 发布，删除 engine 内绕过 publisher 的直接 `torch.save`。
- [ ] 4.2 为 A06 增加序列化、replace 和 registry copy 故障注入测试，证明异常不会留下可选择的半文件、不会损坏既有 checkpoint/sidecar，且临时文件被清理或明确不可选。
- [ ] 4.3 实现 A07 的 SHA-256/size sidecar 完成标记与 reader 验证；current checkpoint 缺 sidecar、sidecar 未完成或摘要不符时，resume/registry/final test 全部 fail-closed。
- [ ] 4.4 实现 A08 的逐文件 `checkpoint_role`/selection provenance 和 selection catalog；分别验证 objective-best、Top-1-best、自定义 best、last 的 metric/mode/value/epoch/source/final-candidate 不互相冒充。
- [ ] 4.5 实现 A09 的 current checkpoint schema：只写真实 `validation_loss`，删除新 payload 的 `test_loss`；包含旧 alias 或缺少 current schema 的 payload 必须被 resume reader 拒绝。

## 5. 实际选模与最终测试产物

- [ ] 5.1 实现 A11 的统一 selected-checkpoint resolver：默认 objective selection、显式 Top-1/自定义 selection 和 fixed-epoch last 使用各自已验证候选，final test、run status、返回值和 final config 共享同一解析结果。
- [ ] 5.2 实现 A05 的跨 run selection catalog 和零剩余 epoch 路径；无新 epoch/无新 best 时验证源候选 path/digest/source run，候选缺失、摘要漂移或策略歧义时明确失败。
- [ ] 5.3 实现 A10 的 final-test evaluation 非持久化共享 pass：先在内存补 `evaluation_split`、model-selection split 与实际 checkpoint provenance，再原子发布 `final_test_metrics.json`。
- [ ] 5.4 保证 A10 的 final test 不覆盖 validation/通用 `metrics.json`，并让 `train_log.json`、final config、run status 和返回对象引用同一 final-test 内容；故障路径不得写 `complete`。
- [ ] 5.5 更新 run index/artifact summary 的只读解析，使其优先使用已验证逐文件 sidecar与独立 final-test artifact，缺失 provenance 时输出 warning 而不是从文件名猜测。

## 6. 指标、worker、timing 与 synthetic 确定性

- [ ] 6.1 实现 A15 的训练 metric numerator/denominator bundle，按每项有效 sample/token 加权 total/task/auxiliary loss 和 accuracy；覆盖最后小 batch、masked token、不同分母与零有效计数。
- [ ] 6.2 实现 A12 的 detached device-side 累计和紧凑批量主机物化；progress 按刷新间隔采样，默认 batch 热路径不得为 total/task/auxiliary/accuracy 分别 `.item()`/`.cpu()`，并用 focused spy/characterization test 验证。
- [ ] 6.3 实现 A16 的 skipped-validation unavailable 语义：当前 epoch 的 `val_*`/`validation_metrics`/primary metric 为 null/NaN，最近观测只作独立 provenance，且不更新 best、scheduler-on-validation、patience 或 early stopping。
- [ ] 6.4 实现 A13 的 worker 生命周期：删除每轮 validation 后的私有 shutdown，persistent worker 跨 epoch 复用；正常/异常 run finalization 统一关闭所有可关闭 loader 且不覆盖原始异常。
- [ ] 6.5 实现 A14 的 timing config 与 recorder：默认关闭且零 timer/probe/file 开销；显式 host/CUDA-event profile 才采样，CUDA 阶段用 event，row 缓冲后写当前 `run_dir` 专属 artifact，不逐 batch 或跨 run 共享同步写。
- [ ] 6.6 实现 A17 的 synthetic per-index 稳定 seed，使用 base seed、split、schema identity 和 index 的稳定摘要构造局部 generator；验证重复 index、不同访问顺序、不同 worker 数一致，seed/split 域分离。
- [ ] 6.7 统一 train/validation/test generator 的稳定域分离和 runtime metadata，验证构建或迭代其它 split 不改变当前 split 顺序，且不使用 Python 进程随机化 `hash()`。

## 7. 回归与收口

- [ ] 7.1 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_training_io_run_metadata.py tests/test_training_io_dataset_workflow.py tests/test_training_io_label_workflow.py tests/test_training_io_cache_workflow.py tests/test_checkpoint_security.py tests/test_evaluation_pass.py -q`，确认 A01-A17 focused regression 全部通过。
- [ ] 7.2 运行 `make verify-quick`、`make verify-cli-config` 和 `make verify-compile`，确认训练 phase、配置 characterization、三个 retained CLI 与导入边界未回归。
- [ ] 7.3 运行 `openspec validate harden-training-resume-and-runtime-observability --strict` 和 `openspec validate --all --strict`，修正所有 artifact/spec 一致性问题。
- [ ] 7.4 运行 `make verify-full` 和 `conda run -n kd_mm_beam pytest -q` 完成最终回归；不得启动真实训练、读取真实 `dataset/` 或生成需提交的 checkpoint。
- [ ] 7.5 审计 `git status --short` 与 tracked 文件，确认没有纳入 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard 或临时验证产物，并更新本清单实际完成状态。
