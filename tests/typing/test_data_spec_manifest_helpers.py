from __future__ import annotations

import pytest

from retriever_typing.data import (
    DataSpec,
    Event,
    EventBuffer,
    SchemaRef,
    StreamId,
    StreamSpec,
    build_dataset_manifest,
    build_episode_manifest,
    validate_dataset_manifest,
)


def _evt(stream: str, t: int, seq: int, value: int) -> Event[int]:
    return Event(
        stream_id=StreamId(stream),
        event_time_ns=t,
        ingest_time_ns=t + 5,
        seq=seq,
        value=value,
        type_name="int",
    )


def test_episode_and_dataset_manifest_helpers_roundtrip() -> None:
    spec = DataSpec(
        name="tabletop",
        version="1.0",
        streams=(
            StreamSpec(stream_id=StreamId("cam"), schema=SchemaRef(name="RGBImage")),
            StreamSpec(stream_id=StreamId("joint"), schema=SchemaRef(name="JointState")),
        ),
    )
    episode = build_episode_manifest(
        "ep-1",
        {
            "cam": EventBuffer((_evt("cam", 100, 0, 1), _evt("cam", 200, 1, 2))),
            "joint": EventBuffer((_evt("joint", 150, 0, 3),)),
        },
        artifacts=("episode.mcap",),
        metadata={"scene": "demo"},
    )
    manifest = build_dataset_manifest(
        "dataset-1",
        spec=spec,
        episodes=(episode,),
        source="unit-test",
    )

    validate_dataset_manifest(manifest)

    assert episode.stream_ids == ("cam", "joint")
    assert episode.start_event_time_ns == 100
    assert episode.end_event_time_ns == 200
    assert episode.event_count == 3
    assert manifest.episodes[0].episode_id == "ep-1"


def test_validate_dataset_manifest_rejects_unknown_episode_stream() -> None:
    spec = DataSpec(
        name="tabletop",
        version="1.0",
        streams=(StreamSpec(stream_id=StreamId("cam"), schema=SchemaRef(name="RGBImage")),),
    )
    episode = build_episode_manifest(
        "ep-1",
        {"joint": EventBuffer((_evt("joint", 100, 0, 1),))},
    )
    manifest = build_dataset_manifest(
        "dataset-1",
        spec=spec,
        episodes=(episode,),
        source="unit-test",
    )

    with pytest.raises(ValueError, match="not present in data spec"):
        validate_dataset_manifest(manifest)
