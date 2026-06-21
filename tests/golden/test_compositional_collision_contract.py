from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from retriever_typing import Header, PoseStamped, Quaternion, SE3Pose, Twist, TwistStamped, Vector3


class AmbiguousFieldError(RuntimeError):
    pass


class FieldNotFoundError(RuntimeError):
    pass


class _AliasView:
    def __init__(self, data: Any):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._data, name):
            return getattr(self._data, name)
        raise FieldNotFoundError(name)


class CompositeIOView:
    def __init__(self, payloads: dict[str, Any]):
        self._payloads = payloads
        self._field_to_aliases: dict[str, list[str]] = {}
        for alias, obj in payloads.items():
            for field in vars(obj).keys():
                self._field_to_aliases.setdefault(field, []).append(alias)

    def __getattr__(self, name: str) -> Any:
        if name in self._payloads:
            return _AliasView(self._payloads[name])

        aliases = self._field_to_aliases.get(name)
        if not aliases:
            raise FieldNotFoundError(name)
        if len(aliases) > 1:
            raise AmbiguousFieldError(name)
        return getattr(self._payloads[aliases[0]], name)

    def _get_signal(self, key: str) -> Any:
        if "." in key:
            alias, field = key.split(".", 1)
            if alias not in self._payloads:
                raise FieldNotFoundError(key)
            payload = self._payloads[alias]
            if not hasattr(payload, field):
                raise FieldNotFoundError(key)
            return getattr(payload, field)
        return getattr(self, key)

    def _set_signal(self, key: str, value: Any) -> None:
        if "." in key:
            alias, field = key.split(".", 1)
            if alias not in self._payloads:
                raise FieldNotFoundError(key)
            payload = self._payloads[alias]
            if not hasattr(payload, field):
                raise FieldNotFoundError(key)
            setattr(payload, field, value)
            return
        aliases = self._field_to_aliases.get(key)
        if not aliases:
            raise FieldNotFoundError(key)
        if len(aliases) > 1:
            raise AmbiguousFieldError(key)
        setattr(self._payloads[aliases[0]], key, value)

    def _has_signal(self, key: str) -> bool:
        if "." in key:
            alias, field = key.split(".", 1)
            return alias in self._payloads and hasattr(self._payloads[alias], field)
        aliases = self._field_to_aliases.get(key)
        if not aliases:
            return False
        if len(aliases) > 1:
            raise AmbiguousFieldError(key)
        return True


@dataclass
class A:
    arg1: int
    only_a: int


@dataclass
class B:
    arg1: int
    only_b: int


def test_unique_unqualified_access_succeeds() -> None:
    view = CompositeIOView({"A": A(arg1=1, only_a=10), "B": B(arg1=2, only_b=20)})
    assert view.only_a == 10
    assert view.only_b == 20


def test_ambiguous_unqualified_access_raises() -> None:
    view = CompositeIOView({"A": A(arg1=1, only_a=10), "B": B(arg1=2, only_b=20)})
    with pytest.raises(AmbiguousFieldError):
        _ = view.arg1
    with pytest.raises(AmbiguousFieldError):
        view._get_signal("arg1")
    with pytest.raises(AmbiguousFieldError):
        view._set_signal("arg1", 9)
    with pytest.raises(AmbiguousFieldError):
        view._has_signal("arg1")


def test_qualified_access_succeeds() -> None:
    view = CompositeIOView({"A": A(arg1=1, only_a=10), "B": B(arg1=2, only_b=20)})
    assert view.A.arg1 == 1
    assert view.B.arg1 == 2
    assert view._get_signal("A.arg1") == 1
    assert view._has_signal("B.arg1")
    view._set_signal("A.arg1", 77)
    assert view.A.arg1 == 77


def test_shared_spatial_payloads_follow_same_collision_rules() -> None:
    header = Header(stamp_ns=100, frame_id="map", source="unit-test")
    pose = PoseStamped(
        header=header,
        pose=SE3Pose(
            position=Vector3(0.0, 0.0, 0.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
    )
    twist = TwistStamped(
        header=header,
        twist=Twist(
            linear=Vector3(1.0, 0.0, 0.0),
            angular=Vector3(0.0, 0.0, 1.0),
        ),
    )

    view = CompositeIOView({"pose": pose, "twist": twist})

    assert view.pose.pose.position.x == 0.0
    assert view.twist.twist.linear.x == 1.0
    with pytest.raises(AmbiguousFieldError):
        _ = view.header
    assert view._get_signal("pose.header") == header
