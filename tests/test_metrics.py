from tensorearch.metrics import base_cost
from tensorearch.schema import SliceState


def test_base_cost_adds_all_terms():
    s = SliceState(
        slice_id="blk0.attn",
        kind="attention",
        flops=1.0,
        memory_bytes=2.0,
        comm_bytes=3.0,
        sync_cost=4.0,
    )
    assert base_cost(s) == 10.0
