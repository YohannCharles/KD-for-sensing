## Context

当前仓库的主线文档仍以 final C2/DeepSense6G 为默认，实际正在推进的 T2 MMW change 则以 `u_mask_beam_jepa` 为主模型，并以 S1、AMBER-Full、RMBP-MM 为比较行。三条基线 recipe 目前从 ignored `outputs/` 读取，且共享 registry、dataset factory、训练扩展和诊断模块仍无条件暴露 CSI、物理 MMW、C2 和历史路线。

本变更是用户明确授权的 breaking cleanup：不保留旧命令、旧 YAML、迁移 guard 或兼容 facade。已归档 OpenSpec 和集中历史说明承担追溯职责；`dataset/`、`outputs/`、`logs/`、cache 与 checkpoint 不在删除范围。

## Goals / Non-Goals

**Goals:**

- 让干净 clone 仅凭 tracked 文件即可构建并启动 T2、S1、AMBER-Full、RMBP-MM 的 MMW workflow。
- 删除所有不在四方法传递闭包内的 current source/config/script/test/CLI，并收缩导入和 registry。
- 删除外部 teacher artifact guidance、checkpoint/full-to-partial KD 与其兼容字段；保留 T2 同一模型 no-grad full/superset consistency、BPA 与 active CMA ablation。
- 以集中历史说明替代运行代码的 tombstone、alias 和 migration guard。
- 停用 GitHub Actions，并删除仅供 GitHub、Cursor 或 Kiro 使用的协作适配；Codex 使用本地文档与验证命令。

**Non-Goals:**

- 不运行训练、不改写本地数据或删除 runtime artifacts。
- 不取消或重写 active `tune-t2-mmw-hyperparameters`、`validate-t2-mmw-bpa-cma-ablation` 的 T2 需求。
- 不把 AMBER-Full 或 RMBP-MM 伪装成官方原始复现，也不改变其 baseline caveat。
- 不为了保留旧 API 而增加 adapter、fallback 或 compatibility config。

## Decisions

### 1. 用四方法和一个共享 MMW base 定义唯一闭包

保留 T2、S1、AMBER-Full、RMBP-MM。S1 是 T2 的 superset-consistency-off 对照，不新增独立实现；AMBER-Full、RMBP-MM 保留为现有比较所需 baseline。新增 tracked MMW shared base 和小 overlay，替换三处 `outputs/...yaml` 输入。相比提交 resolved output，这避免把运行元数据、绝对路径和本地产物带入源码。

### 2. 先收缩入口和共享导入，再删除叶子

第一波让 `pyproject`、CLI surface、launcher、registry、data/model factories 只导入四方法闭包；第二波才删不再可达的 C2/DeepSense6G/CSI/physics/诊断实现、配置和测试。这样每波都可用 import、CLI help 与 focused tests 证明没有残留引用，而不是根据目录名批量删除。

### 3. 按行为而不是 token 判断蒸馏

删除 `teacher_guidance`、外部 teacher tensors/checkpoints、`full_to_partial_kd` 和 disabled legacy fields。保留 T2 同一 primary model 产生的 no-grad full/superset logits、BPA/prototype 和 router supervision；它们由 active T2 protocol 消费。相比按 `teacher`、`KD` 字符串删除，这不会破坏 T2。

### 4. 历史只保留集中说明

新增/更新一份简短 T2-era historical note，列出退役族、原用途和可追溯的 dated OpenSpec archive。删除原 route 的实体配置、guard、测试、CLI/script 和 compatibility code；不保留“已删除”占位模块。

### 5. Active T2 artifacts 优先于旧默认主线

以 active T2 changes 定义 protected source/config/script/test 输入；旧 final-C2 contract 被本 change 显式退役。若实现发现某个 active T2 task 仍引用待删文件，先把它改到 canonical T2/baseline owner，再删除旧文件。

### 6. 本地 Codex 协作替代 GitHub 专属配置

`AGENTS.md`、导航、scoped context 与 `docs/agentic_collaboration_guardrails.md` 保留 Codex 的操作规则、任务字段和人工审查提示。删除 `.github/` 的 Actions、Copilot、Issue/PR 模板，以及 `.cursor/`、`.kiro/` 和仅供 Actions 创建环境的 `envs/smoke-dev.yml`。Git 同步不依赖这些文件；如远端分支保护仍把 `verify` 设为必需检查，维护者需在 GitHub 仓库设置中单独移除该规则。

## Risks / Trade-offs

- [四方法的真实依赖比静态名称更宽] → 每次删除前反向搜索 caller，并运行最小 import/config/CLI tests。
- [T2/S1/RMBP recipe 当前只在 ignored outputs] → 从已验证 resolved inputs 提炼无运行元数据的 tracked base/overlays，并用 config load test 固定。
- [MMWDataset 继承 DeepSense6G owner] → 先抽出或保留 T2 所需通用 loader，不直接删除父类。
- [现有工作树包含用户未提交 T2 修改] → 不回退、不重格式化；只删除经反向追踪确认不被 T2/baseline 或 active T2 change 使用的路径。
- [范围很大导致回归成本高] → 分为入口、核心、叶子、文档四波，每波保留一个可运行检查；最终再跑完整验证。
- [远端分支保护仍要求 `verify`] → 代码删除 workflow 后不再触发 Actions；维护者需在远端设置中移除 required check，不能由仓库文件代替。

## Migration Plan

1. 建立 tracked MMW base、T2/S1/RMBP overlays，改 launcher 与 active T2 helper，先验证配置不再读取 `outputs/`。
2. 删除外部 teacher/KD 支线及其配置字段、测试和 compatibility guard，保留 T2 same-model consistency。
3. 收缩 registry、factory、CLI 和 package exports，再删除失去 caller 的 source/config/script/test 家族。
4. 更新 specs、README、inventory、研究/claim 文档和集中历史说明；加入只验证 T2/baseline current surface 的 guard，并移除 GitHub/多工具专属协作配置。
5. 若需回滚，只回滚本 change 的源码与 tracked recipes；不触碰本地训练产物或 archived OpenSpec。

## Open Questions

无。baseline 集合按 active MMW protocol 固定为 S1、AMBER-Full、RMBP-MM；不包含可选 GPS-only 行。
