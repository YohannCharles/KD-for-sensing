# maintainer-context-index Specification

## Purpose

定义面向 MMW T2/baseline 的机器可读维护路由，使 agent 能按任务定位最小 authority、owner module 和 focused validation，而不复制完整项目表面。

## Requirements

### Requirement: 索引只描述最小 current 路由

`docs/maintainer_context_index.yaml` MUST 提供 model、data、config、cli、runtime、openspec、documentation 和 claims 路由，并为每项给出 authority、owner 与 focused validation。

#### Scenario: agent 选择上下文

- **WHEN** agent 接到非平凡任务
- **THEN** 它 MUST 能从索引定位一个 scoped context 和最小验证
- **AND** 索引不得复制完整源码树或历史路线清单
