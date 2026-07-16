import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingState:
    start_epoch: int = 0
    checkpoint_loads: list[dict[str, Any] | None] = field(default_factory=list)
    history: dict[str, list] = field(default_factory=dict)
    epoch_logs: list[dict[str, Any]] = field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_epoch": int(self.start_epoch),
            "checkpoint_loads": copy.deepcopy(self.checkpoint_loads),
            "history": copy.deepcopy(self.history),
            "epoch_logs": copy.deepcopy(self.epoch_logs),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.start_epoch = int(payload.get("start_epoch", self.start_epoch))
        if isinstance(payload.get("checkpoint_loads"), list):
            self.checkpoint_loads[:] = copy.deepcopy(payload["checkpoint_loads"])
        if isinstance(payload.get("history"), dict):
            self.history.clear()
            self.history.update(copy.deepcopy(payload["history"]))
        if isinstance(payload.get("epoch_logs"), list):
            self.epoch_logs[:] = copy.deepcopy(payload["epoch_logs"])
