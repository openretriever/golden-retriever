"""
Effectful object tracking with a deterministic detection simulator.

Run:
  pixi run python examples/advanced/state_management/object_tracking_eff.py --steps 30 --dt 0.1
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

from retriever.flow import Flow, Pipeline, Rate, Latest, io
from retriever.types import Eff


@dataclass(frozen=True)
class Detection:
    label: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Track:
    track_id: int
    label: str
    x: float
    y: float
    vx: float
    vy: float
    age: int
    missed: int
    confidence: float


@dataclass(frozen=True)
class TrackerState:
    tracks: dict[int, Track]
    next_id: int


@io
class DetectionsOut:
    t_sim: float | None = None
    dt: float | None = None
    detections: list[Detection] | None = None


@io
class TrackingOut:
    t_sim: float | None = None
    tracks: list[Track] | None = None
    created: int | None = None
    dropped: int | None = None
    active: int | None = None


class DetectionsSim(Flow[None, DetectionsOut]):
    def __init__(self, *, dt: float, drop_period: int):
        super().__init__()
        self.dt = float(dt)
        self.drop_period = int(drop_period)

    def init_config(self) -> dict:
        return {"dt": self.dt, "drop_period": self.drop_period}

    def init(self) -> None:
        self.step = 0
        self.t_sim = 0.0

    def step(self, _):  # type: ignore[override]
        self.step += 1
        self.t_sim += self.dt

        dets: list[Detection] = []

        # Object A: gentle drift
        x_a = 0.5 + 0.15 * self.step
        y_a = 0.4 + 0.1 * math.sin(0.3 * self.step)
        dets.append(Detection(label="alpha", x=x_a, y=y_a, confidence=0.9))

        # Object B: circular path with periodic dropout.
        if self.drop_period <= 0 or (self.step % self.drop_period) != 0:
            x_b = 2.0 + 0.4 * math.cos(0.25 * self.step)
            y_b = 0.7 + 0.4 * math.sin(0.25 * self.step)
            dets.append(Detection(label="beta", x=x_b, y=y_b, confidence=0.85))

        return DetectionsOut(t_sim=self.t_sim, dt=self.dt, detections=dets)


def predict_tracks(state: TrackerState, dt: float) -> dict[int, Track]:
    predicted: dict[int, Track] = {}
    for track_id, track in state.tracks.items():
        predicted[track_id] = replace(
            track,
            x=track.x + track.vx * dt,
            y=track.y + track.vy * dt,
            age=track.age + 1,
            missed=track.missed + 1,
            confidence=max(0.0, track.confidence - 0.02),
        )
    return predicted


def greedy_match(
    tracks: dict[int, Track],
    detections: list[Detection],
    max_dist: float,
) -> tuple[dict[int, Detection], list[Detection], list[int]]:
    matches: dict[int, Detection] = {}
    remaining = detections[:]

    for track_id, track in tracks.items():
        best_det = None
        best_dist = max_dist
        for det in remaining:
            if det.label != track.label:
                continue
            dist = math.hypot(det.x - track.x, det.y - track.y)
            if dist < best_dist:
                best_dist = dist
                best_det = det
        if best_det is not None:
            matches[track_id] = best_det
            remaining.remove(best_det)

    unmatched_tracks = [tid for tid in tracks if tid not in matches]
    return matches, remaining, unmatched_tracks


def update_with_detection(track: Track, det: Detection, dt: float) -> Track:
    dt = max(dt, 1e-6)
    vx = (det.x - track.x) / dt
    vy = (det.y - track.y) / dt
    new_vx = 0.7 * track.vx + 0.3 * vx
    new_vy = 0.7 * track.vy + 0.3 * vy
    return replace(
        track,
        x=det.x,
        y=det.y,
        vx=new_vx,
        vy=new_vy,
        missed=0,
        confidence=min(1.0, track.confidence + 0.1 * det.confidence),
    )


def init_track(det: Detection, track_id: int) -> Track:
    return Track(
        track_id=track_id,
        label=det.label,
        x=det.x,
        y=det.y,
        vx=0.0,
        vy=0.0,
        age=1,
        missed=0,
        confidence=det.confidence,
    )


def tracking_step(
    detections: list[Detection],
    dt: float,
    max_dist: float,
    max_missed: int,
) -> Eff[TrackerState, TrackingOut]:
    def op(state: TrackerState) -> tuple[TrackingOut, TrackerState]:
        predicted = predict_tracks(state, dt)
        matches, unmatched_dets, unmatched_tracks = greedy_match(
            predicted, detections, max_dist
        )

        updated_tracks: dict[int, Track] = {}
        dropped = 0

        for track_id, track in predicted.items():
            if track_id in matches:
                updated_tracks[track_id] = update_with_detection(
                    track, matches[track_id], dt
                )
            else:
                if track.missed <= max_missed:
                    updated_tracks[track_id] = track
                else:
                    dropped += 1

        next_id = state.next_id
        created = 0
        for det in unmatched_dets:
            updated_tracks[next_id] = init_track(det, next_id)
            next_id += 1
            created += 1

        summary = TrackingOut(
            tracks=sorted(updated_tracks.values(), key=lambda t: t.track_id),
            created=created,
            dropped=dropped,
            active=len(updated_tracks),
        )
        return summary, TrackerState(tracks=updated_tracks, next_id=next_id)

    return Eff(op)


class TrackerFlow(Flow[DetectionsOut, TrackingOut]):
    def __init__(self, *, max_dist: float, max_missed: int):
        super().__init__()
        self.max_dist = float(max_dist)
        self.max_missed = int(max_missed)

    def init_config(self) -> dict:
        return {"max_dist": self.max_dist, "max_missed": self.max_missed}

    def init(self) -> None:
        self.state = TrackerState(tracks={}, next_id=1)

    def reset(self) -> None:
        self.state = TrackerState(tracks={}, next_id=1)

    def step(self, input: DetectionsOut) -> TrackingOut:
        if input.detections is None or input.dt is None or input.t_sim is None:
            return TrackingOut()

        program = tracking_step(
            detections=input.detections,
            dt=float(input.dt),
            max_dist=self.max_dist,
            max_missed=self.max_missed,
        )
        summary, new_state = program.run(self.state)
        self.state = new_state
        summary.t_sim = float(input.t_sim)
        return summary


class Printer(Flow[TrackingOut, None]):
    def __init__(self, *, print_every: int):
        super().__init__()
        self.print_every = int(print_every)

    def init_config(self) -> dict:
        return {"print_every": self.print_every}

    def init(self) -> None:
        self.step = 0

    def step(self, input: TrackingOut) -> None:
        if input.tracks is None or input.t_sim is None:
            return None
        self.step += 1
        if self.print_every <= 0 or self.step % self.print_every != 0:
            return None
        ids = [f"{t.track_id}:{t.label}" for t in input.tracks]
        print(
            f"[t={input.t_sim:4.1f}s] active={input.active} "
            f"created={input.created} dropped={input.dropped} tracks={ids}"
        )
        return None


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    hz = 1.0 / max(args.dt, 1e-6)
    pipe = Pipeline("object_tracking_eff")

    with pipe:
        sim = DetectionsSim(dt=args.dt, drop_period=args.drop_period) @ Rate(hz=hz)
        tracker = TrackerFlow(max_dist=args.max_dist, max_missed=args.max_missed) @ Rate(
            hz=hz
        )
        printer = Printer(print_every=args.print_every) @ Rate(hz=hz)

        pipe.connect(sim, tracker, sync=Latest())
        pipe.connect(tracker, printer, sync=Latest())

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Effectful object tracking demo.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--drop-period", type=int, default=6)
    p.add_argument("--max-dist", type=float, default=0.6)
    p.add_argument("--max-missed", type=int, default=3)
    p.add_argument("--print-every", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)

    for _ in range(args.steps):
        pipe.step(dt=args.dt)

    pipe.close_stepper()


if __name__ == "__main__":
    main()
