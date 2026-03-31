from dataclasses import dataclass, field

from .schema import SliceEdge, SliceState


@dataclass
class ArchitectureGraph:
    slices: list[SliceState] = field(default_factory=list)
    edges: list[SliceEdge] = field(default_factory=list)

    def add_slice(self, slice_state: SliceState) -> None:
        self.slices.append(slice_state)

    def add_edge(self, edge: SliceEdge) -> None:
        self.edges.append(edge)
