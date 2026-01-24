# TensorBoard logger + config dump + run dir creation

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union
import json

Number = Union[int, float]


class Logger(Protocol):
    """Minimal logging interface. Training code should only depend on this."""

    def log_scalar(self, name: str, value: Number, step: int) -> None: ...
    def log_scalars(
        self, scalars: Mapping[str, Number], step: int, prefix: str = ""
    ) -> None: ...
    def log_text(self, name: str, text: str, step: int) -> None: ...
    def log_config(self, config: Mapping[str, Any]) -> None: ...
    def log_artifact(
        self, path: Union[str, Path], name: Optional[str] = None
    ) -> None: ...
    def close(self) -> None: ...


@dataclass
class NoOpLogger:
    """Logger that does nothing; useful for profiling or disabling logging."""

    def log_scalar(self, name: str, value: Number, step: int) -> None:
        return None

    def log_scalars(
        self, scalars: Mapping[str, Number], step: int, prefix: str = ""
    ) -> None:
        return None

    def log_text(self, name: str, text: str, step: int) -> None:
        return None

    def log_config(self, config: Mapping[str, Any]) -> None:
        return None

    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        return None

    def close(self) -> None:
        return None


class TensorboardLogger:
    def __init__(self, run_dir: Union[str, Path]) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.run_dir = Path(run_dir)
        self.writer = SummaryWriter(run_dir)

    def log_scalar(self, name: str, value: Number, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def log_scalars(
        self, scalars: Mapping[str, Number], step: int, prefix: str = ""
    ) -> None:
        # log scalars independently
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        for k, v in scalars.items():
            self.writer.add_scalar(prefix + k, v, step)

    def log_text(self, name: str, text: str, step: int) -> None:
        self.writer.add_text(name, text, step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        # Human-readable + exact repro
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True)
        )
        # Also show in TB (optional)
        self.writer.add_text(
            "config", f"```json\n{json.dumps(config, indent=2)}\n```", 0
        )

    def log_artifact(self, path: Union[str, Path], name: Optional[str] = None) -> None:
        # TensorBoard doesn’t have a universal artifact concept; copy into run_dir.
        src = Path(path)
        dst = self.run_dir / (name or src.name)
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def make_logger(backend: Optional[str], run_dir: Union[str, Path]):
    if backend is None:
        return NoOpLogger()
    if backend == "tensorboard":
        return TensorboardLogger(run_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}")
