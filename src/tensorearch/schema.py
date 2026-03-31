from dataclasses import dataclass, field
from typing import Any


@dataclass
class SliceState:
    slice_id: str
    kind: str
    flops: float = 0.0
    activation_bytes: float = 0.0
    memory_bytes: float = 0.0
    comm_bytes: float = 0.0
    sync_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceEdge:
    src: str
    dst: str
    weight: float = 0.0
    kind: str = "dataflow"
    metadata: dict[str, Any] = field(default_factory=dict)
