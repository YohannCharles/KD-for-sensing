## Context

项目已经完成从原始脚本到 `src/kd_sensing` 包、配置矩阵、诊断工具和 Gradio viewer 的主要迁移，但最近一次健康检查暴露出几个需要集中处理的问题：

- `conda run -n kd_mm_beam pytest -q` 当前为 `535 passed, 1 failed`，唯一失败来自 Phase 1.5 汇总入口在 checkpoint matrix 未完成时过早给出 `complete` 决策。
- `project-architecture` 已经定义轻量导入和职责拆分要求，但实际 `engine/data_factory.py`、`engine/optim.py` 等模块仍主要从 `_builders_impl.py` 转发；`data.transform_ops` 也仍大量依赖 `_legacy.py`。
- 包级 `__init__.py` 会导致导入单个轻量子模块时牵出重依赖，例如 `matplotlib`、`pandas`、`scipy`、dataset 和训练构建模块。
- `pyproject.toml` 声明的 console scripts 与当前 editable install 生成的 entry points 不一致，README 中建议的部分命令不可直接调用。
- `.gitignore` 已忽略 `*.pth`，但 `All_models/*.pth` 仍被 Git 跟踪；这可能是内置复现权重，也可能是历史遗留，需要明确策略而不是继续混在普通本地产物中。

本变更目标是把这些“项目健康度”问题收敛为一个可验证的稳定化批次，为后续模型实验和诊断功能降低回归风险。

## Goals / Non-Goals

**Goals:**

- 恢复 `kd_mm_beam` 环境下全量 pytest 绿灯。
- 修正 Phase 1.5 final decision gate，避免缺失 checkpoint / audit / baseline 产物时输出 final 结论。
- 收紧 `engine`、`diagnostics`、`distillation` 的包级导入边界，降低最小环境和测试收集脆弱性。
- 将 `_builders_impl.py` 和 `_legacy.py` 中仍在使用的实现逐步迁移到对应职责模块，同时保留兼容 facade。
- 修复 console script entry points 与安装元数据不一致的问题。
- 明确 `All_models` 与新生成 checkpoint 的版本控制边界。
- 增加轻量 smoke / help / import regression，让这些问题后续能被快速发现。

**Non-Goals:**

- 不改变模型结构、训练 loss、router、数据集切分、指标定义或当前实验结论。
- 不重新设计 Phase 1.5 统计方法，只修正决策 gate 与状态表达。
- 不删除用户本地数据、`outputs/`、`logs/` 或已有 checkpoint。
- 不强制迁移真实权重到远程 artifact 服务；本变更只要求明确当前仓库策略和校验方式。
- 不重写 Gradio viewer 交互，只保证相关 entry point 和导入边界稳定。

## Decisions

### 1. Phase 1.5 使用三路完成状态作为 final gate

Phase 1.5 的最终 `decision.status=complete` 只有在 bootstrap、dedicated baseline matrix 和 checkpoint matrix 都完成时才允许产生。缺失 audit 产物、缺失 checkpoint、或 baseline seed 未齐时，summary 仍可输出 bootstrap / baseline 的探索性结果，但总 `decision` 必须保持 `pending`，`evidence_level` 必须保持 `exploratory`。

替代方案是只把缺失项从 final 计算中剔除，但这会让报告在证据矩阵不完整时仍给出 final 标签，和当前 spec 中“缺失条目不得纳入 final decision gate”的语义冲突。

### 2. 用 lazy package exports 替代包级 eager import

`kd_sensing.engine.__init__`、`kd_sensing.diagnostics.__init__`、`kd_sensing.distillation.__init__` 应只声明 `__all__` 和按需 `__getattr__`，避免导入任一子模块时执行重依赖导入链。公共 API 名称保持不变，用户仍可 `from kd_sensing.engine import train`，但只有访问该符号时才导入训练器。

替代方案是要求用户永远导入深层模块，但这会破坏现有公开 API，也不能解决测试收集中由包初始化触发的重依赖问题。

### 3. 先迁移实现，再保留 facade

`engine._builders_impl` 中的 cache policy、modality resolution、data factory、normalization artifacts、run metadata、optim/device 构建逻辑应移动到已经存在的窄模块中；`engine.builders` 继续作为兼容 facade。`data.transform_ops._legacy` 也按 image、gps、lidar、mmwave、radar、io、normalization 迁移实现，`data.transforms` 继续导出旧公共符号。

替代方案是保留 `_builders_impl` / `_legacy` 并只增加测试，但这会让“职责拆分”长期停留在转发层，后续改动仍会集中冲突。

### 4. Console scripts 以 `pyproject.toml` 为唯一声明源

安装后的 entry points 必须与 `pyproject.toml` 一致。实现时应校正目标函数，例如 `kd-sensing-export-viewer-manifest` 指向真正的 manifest export CLI，而不是复用静态可视化兼容入口。验证时用 `conda run -n kd_mm_beam <script> --help` 检查。

替代方案是只在 README 中推荐 `python tools/...`，但项目已经声明包内 CLI，安装入口不可用会继续造成使用歧义。

### 5. 权重文件采用“显式内置资产”策略

`All_models/*.pth` 如果保留在 Git 中，必须被文档标记为内置复现权重，并给出用途、来源、大小和加载路径；如果决定移出 Git，应单独执行迁移并保证配置 fallback 仍有清晰错误。新生成 checkpoint 继续由 `.gitignore` 忽略。

替代方案是直接从 Git 移除所有 `.pth`，但这些文件当前可能支撑 legacy image / image+radar KD 复现，贸然移除会破坏现有配置。

### 6. 健康检查分层运行

保留全量 `conda run -n kd_mm_beam pytest -q` 作为最终验收，同时新增快速 smoke：

- 轻量导入 probe：验证指定导入不加载重依赖。
- CLI help probe：验证 console scripts 可调用。
- Phase 1.5 runner 单测：覆盖 pending gate。
- 互补分析核心测试：防止大型诊断文件拆分后语义漂移。

这能把“项目结构问题”和“模型训练问题”分层暴露，减少每次都依赖完整实验环境才能发现基础回归。

## Risks / Trade-offs

- 公开 API lazy export 可能漏掉某个历史导出符号 → 使用现有 tests 加上显式 import compatibility 测试兜底。
- 拆分 `_builders_impl` 和 `_legacy` 可能造成循环导入 → 先移动纯函数和低层工具，再处理构建入口；每一步运行窄测试。
- Console script 修复可能要求重新 editable install → 文档和验收命令统一使用 `conda run -n kd_mm_beam python -m pip install -e .` 后验证。
- `All_models` 策略如果改为移出 Git，可能影响离线复现 → 本变更先要求明确和测试边界，不把删除权重作为默认任务。
- 轻量导入测试可能与现有 “导入核心子模块暴露公共入口” 要求有张力 → 用 lazy `__getattr__` 同时满足公共入口可访问和不 eager 加载。
