# AGENTS.md

本文件约束 AI agent 和协作者在本仓库中的操作方式。项目结构和架构契约以 OpenSpec 与 README 为准，本文件只记录工作规则，不维护完整目录清单。

## 基本原则

- 使用简体中文撰写说明、计划、OpenSpec 产物和提交相关描述。
- 先阅读已有实现、README 和相关 OpenSpec，再改代码；优先沿用现有模块边界和配置风格。
- 非平凡功能、架构调整、训练流程变化、数据契约变化和兼容性变化应先走 OpenSpec change。
- 不要把本地数据、训练输出、日志、缓存、checkpoint 或临时验证产物纳入源码变更。
- 不要新增旧入口、兼容聚合层或绕过当前 `src/kd_sensing` 包结构的运行方式。

## OpenSpec

- 当前架构和需求权威在 `openspec/specs/`，其中项目结构与模块边界主要看 `openspec/specs/project-architecture/spec.md`。
- 有 active change 时，先读对应 `openspec/changes/<change>/proposal.md`、`design.md`、`tasks.md` 和 specs。
- 实现过程中发现范围、需求或设计决策变化时，先更新对应 OpenSpec artifact，再继续落代码。
- 没有 active change 且只是窄修复、文档小改或测试修复时，可以直接改，但仍需遵守现有 specs。

## 命令环境

所有项目相关 Python 命令、测试、训练、验证和依赖安装都必须使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam <command>
```

常用入口：

```bash
conda run -n kd_mm_beam python scripts/train.py --help
conda run -n kd_mm_beam python scripts/evaluate.py --help
conda run -n kd_mm_beam python scripts/preprocess.py --help
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help
```

## 系统配置与启动项安全

- 禁止修改、重写或把训练命令写入容器启动/认证配置文件，例如 `/root/.container_env`、`/etc/profile`、`/etc/environment`、SSH 配置、系统账号密码配置等；除非用户明确点名要求修改具体文件，并且先说明风险、备份原文件、展示拟修改内容。
- `/root/.container_env` 只能保留容器平台写入的运行状态、用户名、密码、IP、网关等字段；严禁把 `cd ...`、`CUDA_VISIBLE_DEVICES=...`、`python scripts/train.py ...`、`nohup ...`、`tmux ...` 等命令写入 `USERNAME`、`PASSWD` 或其他凭证字段。
- 需要长时间运行训练时，优先在当前 shell 中使用 `conda run -n kd_mm_beam ...`、`tmux`、`nohup`、项目脚本或平台提供的任务系统；不要通过污染系统凭证文件实现“开机自启”。
- 如确实需要配置自启动，必须先征得用户明确确认，并使用专门的启动机制或进程管理工具；不要自行猜测容器平台的内部配置文件用途。

## 验证

窄改动优先运行相关测试；涉及架构、导入边界、CLI 或公共工作流时，至少考虑以下快速检查：

```bash
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help
conda run -n kd_mm_beam kd-sensing-visualize-modalities --help
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_modality_visual_diagnostics.py -q
```

最终回归命令：

```bash
conda run -n kd_mm_beam pytest -q
```

OpenSpec 相关变更还应运行对应校验，例如：

```bash
openspec validate <change> --strict
openspec status --change <change>
```

## 文档边界

- README 保留安装、快速上手、简短目录结构和主要工作流说明。
- OpenSpec 记录需求、架构约束、设计决策和变更历史。
- 本文件记录 agent 操作规则，不重复维护完整项目结构。
- 不要使用 hook 自动重写 README、OpenSpec 或本文件；如需防漂移，优先添加检查型脚本或测试，由人决定文档如何更新。

## 产物边界

- `dataset/` 是本地数据输入，默认不提交。
- `outputs/`、`outputs/cache/`、`logs/`、legacy 根 `cache/`、TensorBoard 产物和新生成 checkpoint 是本地运行产物，默认不提交；新可再生成 cache 默认归入 `outputs/cache/`。
- `All_models/` 中已跟踪权重是历史复现实验资料；新生成的 `.pth`、`.pt`、`.ckpt` 不应进入源码变更。
