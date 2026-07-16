# Result Claims Registry

| claim_id | 方法集合 | 状态 | 证据门槛 |
| --- | --- | --- | --- |
| `CLAIM-MMW-T2-BASELINES-PENDING` | T2、S1、AMBER-Full、RMBP-MM | `pending` | tracked recipe、三 seed、40 epoch `last.pth`、一致 split/mask identity 与 MMW summary |
| `CLAIM-MMW-T2-BPA-CMA-PENDING` | T2 BPA/CMA 配对消融 | `pending` | active BPA/CMA change 规定的 matched runs、paired summary 与 caveat |

development screening、历史路线和无完整 provenance 的本地结果不得升级为 current claim。历史用途仅见 `docs/retired_routes.md`。
