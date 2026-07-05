## Why

项目的 scripts、configs 和热点模块已经有详尽 inventory，但日常维护仍需要人工阅读大段表格判断“这个入口/配置/热点是否应该保留”。需要把这部分治理做成可执行 doctor，让 agent 和维护者在改动前快速得到可定位反馈。

## What Changes

- 增加 scripts surface doctor，检查脚本生命周期、默认 config 引用、输出边界和重复 thin wrapper。
- 增加 configs list/doctor，按 family、lifecycle、formal/smoke/local/manual、真实数据需求和验证命令列出配置。
- 增加 hotspot next-touch doctor，对长文件、长函数、facade 和 helper 给出 split/keep/merge/monitor 建议。
- 为可生成的实验 YAML 规划 recipe/manifest 化迁移，不无损时不得强行删除实体 YAML。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-health-guardrails`: 增加 scripts/configs/hotspot doctor 的健康护栏要求。
- `project-entrypoint-lifecycle`: 增加脚本生命周期 doctor 与默认路径检查。
- `canonical-config-resolution`: 增加 config doctor/list 和 recipe migration 边界。
- `project-hotspot-governance`: 增加 next-touch hotspot decision 输出要求。

## Impact

- 可能新增 package CLI 或开发脚本，但不得成为训练入口。
- 可能更新 inventory、architecture boundary tests、config characterization tests。
- 不删除真实输出、不移动数据、不重写 checkpoint。
