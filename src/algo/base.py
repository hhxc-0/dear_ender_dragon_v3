# interface definition

from typing import Protocol, Mapping
from ..types import Batch  # your batch dataclass


class Learner(Protocol):
    def update(self, batch: Batch) -> Mapping[str, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
