# interface definition

from typing import Protocol, Mapping
from src.types import MiniBatch  # your batch dataclass


class Learner(Protocol):
    def update(self, mini_batch: MiniBatch) -> Mapping[str, float]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
