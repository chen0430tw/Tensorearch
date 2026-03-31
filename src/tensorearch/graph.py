from dataclasses import dataclass, field

from .schema import SliceEdge, SliceState, SystemTrace


@dataclass
class ArchitectureGraph:
    system: SystemTrace | None = None
    slices: list[SliceState] = field(default_factory=list)
    edges: list[SliceEdge] = field(default_factory=list)

    def add_slice(self, slice_state: SliceState) -> None:
        self.slices.append(slice_state)

    def add_edge(self, edge: SliceEdge) -> None:
        self.edges.append(edge)

    def slice_map(self) -> dict[str, SliceState]:
        return {slice_state.slice_id: slice_state for slice_state in self.slices}
