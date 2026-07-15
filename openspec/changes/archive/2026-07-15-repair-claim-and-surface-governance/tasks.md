## 1. Lifecycle 完整性

- [x] 1.1 将 current spec lifecycle guard 改为 actual 与 inventory 集合相等，并报告 missing/extra/duplicate/非法分类。
- [x] 1.2 补齐 3 个 current spec 与 root/current document lifecycle，增加未分类 root 文档和 agent context 失效入口测试。
- [x] 1.3 建立 on-disk `scripts/` 与 script lifecycle 的完整覆盖检查，补齐 owner、保留原因、推荐关系、输出边界、验证和删除条件。
- [x] 1.4 对 9 个当前 MMW 脚本逐项判定 retained/local-manual/delete；删除项先迁移必要结论，保留项纳入 inventory。

## 2. Claim 与 paper gate

- [x] 2.1 将 result claim registry 迁移到固定完整 schema，保持 pending/not_comparable 状态且不填未验证数值。
- [x] 2.2 增加 catalog claim id 唯一外键检查，修复 `CLAIM-OVERNIGHT-BRANCH-ROUTER-V2` 断链。
- [x] 2.3 将 paper exporter 改为 reviewed allowlist 与必填字段 gate，输出明确 excluded reasons。
- [x] 2.4 使用 `conda run -n kd_mm_beam` 增加并运行真实 registry、外键、pending/unknown/完整 reviewed row focused tests。

## 3. 文档与 active change 收口

- [x] 3.1 清理 AGENTS、agent context、research notes、root 复现/环境文档中的退役 CLI、缺失配置、虚假 CI 和“当前硬件”描述。
- [x] 3.2 README 增加最短 conda/editable install 与健康检查，并统一 final C2 主线、MMW supporting campaign 术语。
- [x] 3.3 将 Beam active delta 重基为 current temporal requirement 与 LG/CLS 场景并集，完成 18/18 最后验证任务。
- [x] 3.4 使用 `openspec validate validate-t2-beam-geometry-and-head --strict` 确认归档不会覆盖 current S1 契约；符合条件时按 archive 流程收口。

## 4. 真实验证链

- [x] 4.1 让 `verify_compile.py` 覆盖受控 on-disk owner roots，并排除 dataset/outputs/logs/cache；补未跟踪语法错误测试。
- [x] 4.2 让 `make verify-full` 真正执行 `conda run -n kd_mm_beam pytest -q`，保持 quick/CLI/compile 前置。
- [x] 4.3 新增单一最小 CI，复用 OpenSpec strict、quick、CLI/config、compile 和 full 入口；同步环境文档。

## 5. 回归

- [x] 5.1 运行 `openspec validate repair-claim-and-surface-governance --strict`、`openspec validate --all --strict`、`make verify-quick`、`make verify-cli-config` 和 `make verify-compile`。
- [x] 5.2 运行 `make verify-full`，修复本 change 引入的回归并记录非本 change 阻塞。
- [x] 5.3 复核 `git status --short`，确认没有纳入 PDF、dataset、outputs、logs、cache、checkpoint、generated paper artifact 或其它本地产物。
