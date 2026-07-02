# GitHub 迁移与 Codex 新服务器配置

本页只记录通过 GitHub 迁移源码、在新服务器恢复 Python/Codex 工作环境的最小流程。真实数据、训练输出、日志、cache、checkpoint、`.env` 和 CodeGraph 数据库不走 Git。

## 迁移边界

会随 Git 迁移：

- 源码、配置、测试、OpenSpec、README 和 `docs/`
- 项目级 Codex 配置：`.codex/config.toml`、`.codex/hooks.json`、`.codex/scripts/update-codegraph-index.sh`
- 项目级 Codex/OpenSpec skills：`.codex/skills/`
- CodeGraph 忽略规则：`.codegraph/.gitignore`

不会随 Git 迁移：

- `dataset/`、`outputs/`、`logs/`、`cache/`、`.pytest_cache/`、`.codegraph/*.db`
- 新生成的 `.pth`、`.pt`、`.ckpt`、`.npy`、`.npz` 等大产物
- `.env`、SSH key、GitHub token、Codex 登录态和服务器级配置

## 旧服务器提交

先确认只有要迁移的源码/文档改动进入 commit：

```bash
git status --short
git ls-files --others --exclude-standard
git status --short --ignored=matching
```

`git ls-files --others --exclude-standard` 应该为空或只包含你明确要新增的源码/文档。`git status --short --ignored=matching` 里看到的 `dataset/`、`outputs/`、`logs/`、`.codegraph/`、cache 和 `__pycache__/` 是本地状态，不要 `git add -f`。

提交并推送当前分支：

```bash
branch="$(git branch --show-current)"
git add README.md docs/project_surface_inventory.md docs/server_migration_github_codex.md
git status --short
git commit -m "docs: add server migration runbook"
git push -u origin "$branch"
```

如果 `git status --short` 里还有别人的改动，先确认是否一起提交；不要为了迁移顺手回退。

## 新服务器拉取源码

```bash
git clone <github-repo-url> KD-for-sensing
cd KD-for-sensing
git checkout <branch>
```

GitHub 鉴权用新服务器自己的 SSH key 或 token 配置，不写入仓库。

## Python 环境

本项目所有 Python 命令统一使用 `kd_mm_beam`：

```bash
conda create -n kd_mm_beam python=3.11 -y
conda run -n kd_mm_beam python -m pip install --upgrade pip
```

按新服务器 CUDA/驱动安装匹配的 `torch` 和 `torchvision` 后，再安装本仓库：

```bash
conda run -n kd_mm_beam python -m pip install -e ".[dev,hdf5,lmdb]"
conda run -n kd_mm_beam python -c "import kd_sensing"
conda run -n kd_mm_beam kd-sensing-train --help
```

若只做轻量代码检查，`.[dev]` 足够；需要 HDF5 或 LMDB 数据路径时再保留 `hdf5,lmdb`。

## Codex 与 CodeGraph

仓库已经跟踪项目级 Codex hook。新服务器上的 Codex 账号、CLI 安装和登录态是机器级状态，需要在新服务器单独配置，不能通过 Git 迁移。

Codex 打开本仓库后会读取 `AGENTS.md`。如果机器上有 `codegraph` 命令，Stop hook 会自动初始化或同步 `.codegraph/`；也可以手动执行：

```bash
codegraph init -i .
codegraph sync .
```

没有 `codegraph` 命令时 hook 会安静跳过，Codex 仍可工作，只是少了结构化索引。不要提交 `.codegraph/codegraph.db`、pid、socket 或日志。

给新服务器 Codex 的第一条提示可以直接用：

```text
请先阅读 AGENTS.md、docs/agent_navigation.md 和 docs/server_migration_github_codex.md。
本仓库所有 Python 命令使用 conda run -n kd_mm_beam，不要把 dataset/、outputs/、logs/、cache、checkpoint 或 .codegraph 数据加入 Git。
```

## 数据与运行产物

源码迁移完成后，如果新服务器也要训练或复现实验，单独同步本地数据和必要权重：

```bash
rsync -a --info=progress2 <old-server>:/root/projects/KD-for-sensing/dataset/ ./dataset/
rsync -a --info=progress2 <old-server>:/root/projects/KD-for-sensing/All_models/ ./All_models/
```

`outputs/` 和 `logs/` 默认不迁移。只有需要保留历史结果、诊断表或 checkpoint 时才单独同步到新服务器，仍不要通过 GitHub。

## 最小验证

```bash
openspec list --json
conda run -n kd_mm_beam python -c "import kd_sensing"
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_cli_help.py -q
```

如果新服务器还没有安装 `openspec` 或 `codegraph` CLI，对应命令可以先跳过，但要在后续开发前补齐。
