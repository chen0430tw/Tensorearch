from .schema import SliceState


def base_cost(
    slice_state: SliceState,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    return (
        alpha * slice_state.flops
        + beta * slice_state.memory_bytes
        + gamma * slice_state.comm_bytes
        + delta * slice_state.sync_cost
    )
