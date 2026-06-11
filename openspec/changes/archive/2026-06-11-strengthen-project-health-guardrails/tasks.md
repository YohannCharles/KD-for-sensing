## 1. 健康基线与 inventory

- [x] 1.1 生成当前源码/测试/配置健康基线，记录源码行数、测试行数、配置数量、最长函数/类和已知热点清单，作为 `docs/project_surface_inventory.md` 的更新依据。
- [x] 1.2 更新 `docs/project_surface_inventory.md`，新增项目健康护栏章节，列出已知热点、推荐拆分方向、暂缓原因和后续优先级。
- [x] 1.3 为 `configs/` 实验子目录和 root/docs 文档补充生命周期分类，确保长期入口、实验复现、debug/smoke、dataset preparation、diagnostics、历史记录边界清晰。

## 2. 测试基础设施收敛

- [x] 2.1 新增或更新 `tests/conftest.py`，集中管理普通测试导入 `src/` 的 bootstrap，并避免 eager import 重依赖 runtime 模块。
- [x] 2.2 在 `pyproject.toml` 或等价 pytest 配置中声明 `testpaths`、必要 markers 和 warning 约束，不新增 mandatory runtime dependency。
- [x] 2.3 分批移除普通测试文件中的重复 `ROOT/SRC/sys.path.insert` 文件级启动片段；保留架构边界子进程 import probe 的局部显式路径控制。

## 3. 架构与健康 guardrail

- [x] 3.1 扩展 `tests/test_architecture_boundaries.py`，检查 shared pytest bootstrap、健康 inventory、热点清单和分层健康检查命令存在且互相一致。
- [x] 3.2 增加 AST 静态检查 helper，识别新增或未登记的超长函数/类、兼容 facade 回流和热点 inventory 漏项，并输出文件/符号级失败信息。
- [x] 3.3 扩展配置引用检查，扫描 README、docs、scripts 和当前 OpenSpec specs 中的当前支持面配置路径，拒绝指向不存在文件的未分类引用。
- [x] 3.4 扩展文档支持面检查，确保 root 文档和 `docs/` 研究/复现文档在 inventory 中有生命周期分类，且当前长期文档不推荐退役入口。

## 4. 文档与健康检查说明

- [x] 4.1 更新 README 的快速健康检查段落，加入本 change 定义的分层验证命令，并保持所有 Python 命令使用 `conda run -n kd_mm_beam`。
- [x] 4.2 如需要，新增或更新开发者说明，解释新增热点时如何更新 inventory、何时拆分窄模块、何时允许 import-boundary probe 例外。
- [x] 4.3 确认文档不要求提交真实数据、训练输出、日志、cache、checkpoint 或本地临时验证产物。

## 5. 验证

- [x] 5.1 运行 `openspec validate strengthen-project-health-guardrails --strict` 并修复所有 OpenSpec 问题。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 并修复架构边界问题。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`，确认 CLI help 和配置加载 smoke 没有回归。
- [x] 5.4 如果实现过程中触碰具体训练、数据集、诊断或模型 forward 文件，追加对应 focused tests，并在最终说明中列出未运行的检查及原因。
