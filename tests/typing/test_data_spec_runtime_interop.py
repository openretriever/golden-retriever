from __future__ import annotations

import pytest

from retriever_typing.data import (
    from_runtime_event_buffer,
    is_runtime_event_buffer,
    to_runtime_event_buffer,
)


def test_runtime_buffer_roundtrip() -> None:
    runtime = [(1.0, "a"), (2.5, "b")]
    typed = from_runtime_event_buffer(runtime, stream_id="cam")

    assert len(typed) == 2
    assert typed[0].event_time_ns == 1_000_000_000
    assert typed[1].value == "b"

    back = to_runtime_event_buffer(typed)
    assert back == runtime


def test_runtime_buffer_preserves_optional_metadata() -> None:
    runtime = [(1.0, "a"), (2.0, "b")]
    typed = from_runtime_event_buffer(
        runtime,
        stream_id="cam",
        frame_id="camera",
        units="score",
        ingest_offset_ns=25,
    )

    assert typed[0].frame_id == "camera"
    assert typed[0].units == "score"
    assert typed[0].ingest_time_ns == typed[0].event_time_ns + 25
    assert typed[1].ingest_time_ns == typed[1].event_time_ns + 25


def test_runtime_buffer_rejects_negative_ingest_offset() -> None:
    with pytest.raises(ValueError, match="ingest_offset_ns"):
        from_runtime_event_buffer([(1.0, "a")], stream_id="cam", ingest_offset_ns=-1)


def test_runtime_buffer_shape_detection() -> None:
    assert is_runtime_event_buffer([(1.0, 1), (2.0, 2)])
    assert is_runtime_event_buffer([])
    assert not is_runtime_event_buffer({"ts": 1.0})


def test_interop_with_runtime_eventbuffer_class_if_available() -> None:
    flow_types = pytest.importorskip("retriever.flow.types")
    event_buffer_cls = getattr(flow_types, "EventBuffer", None)
    if event_buffer_cls is None:
        pytest.skip("retriever runtime does not expose retriever.flow.types.EventBuffer")
    runtime_buffer = event_buffer_cls([(3.0, 33), (4.0, 44)])

    typed = from_runtime_event_buffer(runtime_buffer, stream_id="imu")
    back = to_runtime_event_buffer(typed)

    assert back == [(3.0, 33), (4.0, 44)]
