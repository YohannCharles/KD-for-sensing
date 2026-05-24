## 1. Run Index Core

- [x] 1.1 新增只读 run discovery helper，扫描 `outputs/` 下包含运行 artifact 的目录并生成候选 run records。
- [x] 1.2 实现 artifact presence、timestamp、config、metrics、checkpoint 和 TensorBoard 摘要提取逻辑。
- [x] 1.3 实现日志关联与失败模式解析，覆盖 `Killed`、Traceback、conda failed、waiting checkpoint 等常见模式。
- [x] 1.4 实现 run state 推断规则，输出 `running`、`complete`、`started_no_metrics`、`partial`、`failed`、`killed`、`waiting`、`stale` 和 `unknown`。

## 2. Resource Snapshot

- [x] 2.1 实现当前 Python/训练进程发现与命令行 run_name/config 匹配。
- [x] 2.2 实现可选 GPU 快照读取，缺少 `nvidia-smi` 时安全降级。
- [x] 2.3 实现系统内存、swap 和进程 RSS 摘要，并与 run records 关联。

## 3. CLI And Output

- [x] 3.1 新增 run index CLI 或研究脚本入口，支持 `--outputs`、`--logs`、`--format`、`--state` 和 `--output`。
- [x] 3.2 支持 JSON 输出，并提供简洁表格输出；可选支持 CSV 输出。
- [x] 3.3 如采用正式 console script，更新 `pyproject.toml` 和 CLI help 测试。

## 4. Training/Evaluation Status Sidecar

- [x] 4.1 在训练入口创建 run_dir 后写出 `running` 状态 sidecar 或等价 runtime status。
- [x] 4.2 在训练正常结束后写出 `complete` 状态、duration、primary metric、metrics path 和 best checkpoint。
- [x] 4.3 在可捕获 Python 异常路径写出 `failed` 状态和异常摘要，不改变异常抛出语义。
- [x] 4.4 为评估入口补齐同等的 started/complete/failed 状态记录。

## 5. Documentation And Tests

- [x] 5.1 更新 README 或 docs，说明 run index 入口、状态分类和只读边界。
- [x] 5.2 添加纯函数测试，覆盖完整 run、partial run、started/no metrics、Killed log、waiting checkpoint 和 running process。
- [x] 5.3 使用 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q` 或等价测试验证 CLI help。
- [x] 5.4 使用 `conda run -n kd_mm_beam pytest <focused run-index tests> -q` 验证 run index 输出 schema。
- [x] 5.5 视影响范围运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
