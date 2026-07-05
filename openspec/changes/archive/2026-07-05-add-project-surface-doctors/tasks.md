## 1. Doctor 范围定义

- [x] 1.1 定义 scripts/configs/hotspots doctor 的输入、输出格式和失败级别。
- [x] 1.2 确认 doctor 只读 tracked files、OpenSpec、pyproject、inventory 和 docs，不读取真实 `dataset/`。
- [x] 1.3 决定 doctor 暴露方式：package CLI、开发脚本或 verify 子命令。

## 2. Scripts / entrypoint doctor

- [x] 2.1 扫描 tracked `scripts/` 和 `tools/analysis/`。
- [x] 2.2 从 inventory、README/docs 或 OpenSpec 推导 lifecycle。
- [x] 2.3 报告未分类脚本、重复 thin wrapper、失效默认 config 和输出边界缺失。

## 3. Config doctor

- [x] 3.1 列出 tracked YAML 和 virtual config route，按 family/lifecycle 分类。
- [x] 3.2 报告失效引用、退役 token 回流和缺少文档分类的 config。
- [x] 3.3 输出 recipe migration candidates，但不自动删除实体 YAML。

## 4. Hotspot doctor

- [x] 4.1 从 inventory 和代码统计生成 hotspot next-touch 报告。
- [x] 4.2 对每个热点输出 split/merge/keep/monitor/accepted/hard-budget 建议和 focused tests。
- [x] 4.3 确保行数只作为趋势信号，不作为唯一决策。

## 5. 验证

- [x] 5.1 运行 `openspec validate add-project-surface-doctors --strict`。
- [x] 5.2 运行 doctor smoke，确认只读且输出可定位。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
