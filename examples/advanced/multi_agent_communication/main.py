"""
Integrated Multi-Agent Communication Example (Advanced)

Run:
  pixi run python examples/advanced/multi_agent_communication/app.py --mode step --steps 60 --dt 0.1
  pixi run python examples/advanced/multi_agent_communication/app.py --mode run --backend multiprocessing --duration 6
  pixi run python examples/advanced/multi_agent_communication/app.py --mode run --backend dora --duration 6
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from retriever import Flow
from retriever.flow import Pipeline, Rate, io
from retriever.flow.adapter import Latest


@io
class TaskAnnouncement:
    task_id: int | None = None
    required_skill: str | None = None
    zone: str | None = None
    priority: int | None = None


@io
class TaskSignal:
    packet: TaskAnnouncement | None = None


@io
class BidPacket:
    agent_id: str | None = None
    task_id: int | None = None
    can_execute: bool | None = None
    eta_s: float | None = None
    energy_cost: float | None = None
    score: float | None = None
    note: str | None = None


@io
class BidSignal:
    packet: BidPacket | None = None


@io
class AuctionInput:
    task: TaskAnnouncement | None = None
    scout_bid: BidPacket | None = None
    carrier_bid: BidPacket | None = None


@io
class Assignment:
    task_id: int | None = None
    assignee: str | None = None
    reason: str | None = None
    scout_score: float | None = None
    carrier_score: float | None = None


@io
class AssignmentSignal:
    packet: Assignment | None = None


@io
class AgentInput:
    task: TaskAnnouncement | None = None
    assignment: Assignment | None = None


@io
class AgentReport:
    agent_id: str | None = None
    task_id: int | None = None
    status: str | None = None
    progress: float | None = None
    note: str | None = None


@io
class ReportSignal:
    packet: AgentReport | None = None


@io
class MonitorInput:
    task: TaskAnnouncement | None = None
    assignment: Assignment | None = None
    scout_report: AgentReport | None = None
    carrier_report: AgentReport | None = None


class TaskGenerator(Flow[None, TaskSignal]):
    """Periodically announces tasks as atomic packets."""

    def __init__(self) -> None:
        super().__init__()
        self._schedule = [
            {"at_step": 5, "task_id": 101, "required_skill": "inspect", "zone": "A", "priority": 2},
            {"at_step": 20, "task_id": 102, "required_skill": "lift", "zone": "C", "priority": 3},
            {"at_step": 35, "task_id": 103, "required_skill": "transport", "zone": "B", "priority": 1},
        ]

    def init(self) -> None:
        self._step = 0
        self._idx = 0

    def reset(self) -> None:
        self._step = 0
        self._idx = 0

    def run(self, _):  # type: ignore[override]
        self._step += 1
        if self._idx >= len(self._schedule):
            return TaskSignal()

        slot = self._schedule[self._idx]
        if self._step < int(slot["at_step"]):
            return TaskSignal()

        self._idx += 1
        return TaskSignal(
            packet=TaskAnnouncement(
                task_id=int(slot["task_id"]),
                required_skill=str(slot["required_skill"]),
                zone=str(slot["zone"]),
                priority=int(slot["priority"]),
            )
        )


class BiddingAgent(Flow[TaskSignal, BidSignal]):
    """Converts task packets into one bid packet per task."""

    def __init__(
        self,
        *,
        agent_id: str,
        skills: set[str],
        speed: float,
        cost_bias: float,
    ) -> None:
        super().__init__()
        self.agent_id = str(agent_id)
        self.skills = set(skills)
        self.speed = float(speed)
        self.cost_bias = float(cost_bias)

    def init_config(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "skills": sorted(self.skills),
            "speed": self.speed,
            "cost_bias": self.cost_bias,
        }

    def init(self) -> None:
        self._last_task_id: int | None = None

    def reset(self) -> None:
        self._last_task_id = None

    def run(self, task_signal: TaskSignal) -> BidSignal:
        task = task_signal.packet
        if (
            task is None
            or task.task_id is None
            or task.required_skill is None
            or task.priority is None
            or task.zone is None
        ):
            return BidSignal()

        task_id = int(task.task_id)
        if self._last_task_id == task_id:
            return BidSignal()
        self._last_task_id = task_id

        zone_factor = {"A": 1.0, "B": 1.4, "C": 2.0}.get(task.zone, 1.5)
        can_execute = task.required_skill in self.skills
        eta_s = zone_factor / max(self.speed, 1e-6)
        energy_cost = zone_factor * self.cost_bias

        if can_execute:
            score = float(task.priority) * 10.0 - (2.0 * eta_s + energy_cost)
            note = "eligible"
        else:
            score = -1.0
            note = "skill_mismatch"

        return BidSignal(
            packet=BidPacket(
                agent_id=self.agent_id,
                task_id=task_id,
                can_execute=bool(can_execute),
                eta_s=float(eta_s),
                energy_cost=float(energy_cost),
                score=float(score),
                note=note,
            )
        )


class Auctioneer(Flow[AuctionInput, AssignmentSignal]):
    """Fan-in bids and emit one assignment packet per task."""

    def init(self) -> None:
        self._assigned_task_ids: set[int] = set()

    def reset(self) -> None:
        self._assigned_task_ids = set()

    def run(self, input: AuctionInput) -> AssignmentSignal:
        task = input.task
        if task is None or task.task_id is None:
            return AssignmentSignal()

        task_id = int(task.task_id)
        if task_id in self._assigned_task_ids:
            return AssignmentSignal()

        scout = input.scout_bid
        carrier = input.carrier_bid
        if scout is None or carrier is None:
            return AssignmentSignal()
        if scout.task_id != task_id or carrier.task_id != task_id:
            return AssignmentSignal()

        bids = [scout, carrier]
        eligible = [b for b in bids if b.can_execute and b.score is not None]

        if not eligible:
            self._assigned_task_ids.add(task_id)
            return AssignmentSignal(
                packet=Assignment(
                    task_id=task_id,
                    assignee="none",
                    reason="no_eligible_bid",
                    scout_score=float(scout.score or -1.0),
                    carrier_score=float(carrier.score or -1.0),
                )
            )

        winner = sorted(
            eligible,
            key=lambda b: (float(b.score), -float(b.eta_s or 1e9)),
            reverse=True,
        )[0]

        self._assigned_task_ids.add(task_id)
        return AssignmentSignal(
            packet=Assignment(
                task_id=task_id,
                assignee=str(winner.agent_id),
                reason="max_score",
                scout_score=float(scout.score or -1.0),
                carrier_score=float(carrier.score or -1.0),
            )
        )


class ExecutionAgent(Flow[AgentInput, ReportSignal]):
    """Executes assigned task and streams progress reports."""

    def __init__(self, *, agent_id: str) -> None:
        super().__init__()
        self.agent_id = str(agent_id)

    def init_config(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    def init(self) -> None:
        self._active_task_id: int | None = None
        self._remaining_ticks = 0
        self._total_ticks = 1

    def reset(self) -> None:
        self._active_task_id = None
        self._remaining_ticks = 0
        self._total_ticks = 1

    def _duration_ticks(self, skill: str | None, zone: str | None) -> int:
        skill_ticks = {"inspect": 6, "lift": 10, "transport": 8}
        zone_ticks = {"A": 0, "B": 2, "C": 4}
        return int(skill_ticks.get(skill or "", 7) + zone_ticks.get(zone or "", 1))

    def run(self, input: AgentInput) -> ReportSignal:
        task = input.task
        assignment = input.assignment

        if self._active_task_id is not None:
            self._remaining_ticks = max(0, self._remaining_ticks - 1)
            progress = 1.0 - (self._remaining_ticks / max(self._total_ticks, 1))
            status = "done" if self._remaining_ticks == 0 else "working"

            packet = AgentReport(
                agent_id=self.agent_id,
                task_id=self._active_task_id,
                status=status,
                progress=float(progress),
                note="completed" if status == "done" else "in_progress",
            )

            if status == "done":
                self._active_task_id = None
                self._remaining_ticks = 0
                self._total_ticks = 1

            return ReportSignal(packet=packet)

        if (
            task is None
            or task.task_id is None
            or assignment is None
            or assignment.assignee is None
            or assignment.task_id is None
        ):
            return ReportSignal()

        task_id = int(task.task_id)
        assignee = str(assignment.assignee)
        assignment_task_id = int(assignment.task_id)

        # Guard against stale assignment packets from previous tasks.
        if assignment_task_id != task_id:
            return ReportSignal()

        if assignee != self.agent_id:
            return ReportSignal()

        self._active_task_id = task_id
        self._total_ticks = self._duration_ticks(task.required_skill, task.zone)
        self._remaining_ticks = self._total_ticks
        return ReportSignal(
            packet=AgentReport(
                agent_id=self.agent_id,
                task_id=task_id,
                status="accepted",
                progress=0.0,
                note=f"start_{task.required_skill}",
            )
        )

        return ReportSignal()


class MissionMonitor(Flow[MonitorInput, None]):
    """Collect and print deduplicated communication events."""

    def init(self) -> None:
        self._seen_tasks: set[int] = set()
        self._seen_assignments: set[tuple[int, str]] = set()
        self._seen_reports: set[tuple[str, int, str, int]] = set()

    def reset(self) -> None:
        self._seen_tasks = set()
        self._seen_assignments = set()
        self._seen_reports = set()

    def _report_key(self, report: AgentReport) -> tuple[str, int, str, int]:
        progress_bucket = int((report.progress or 0.0) * 10)
        return (
            str(report.agent_id),
            int(report.task_id or -1),
            str(report.status),
            progress_bucket,
        )

    def run(self, input: MonitorInput) -> None:
        task = input.task
        assignment = input.assignment

        if task is not None and task.task_id is not None and task.task_id not in self._seen_tasks:
            self._seen_tasks.add(int(task.task_id))
            print(
                f"[task] id={task.task_id} skill={task.required_skill} "
                f"zone={task.zone} priority={task.priority}"
            )

        if assignment is not None and assignment.task_id is not None and assignment.assignee is not None:
            key = (int(assignment.task_id), str(assignment.assignee))
            if key not in self._seen_assignments:
                self._seen_assignments.add(key)
                print(
                    f"[assign] task={assignment.task_id} -> {assignment.assignee} "
                    f"reason={assignment.reason}"
                )

        for report in [input.scout_report, input.carrier_report]:
            if report is None or report.task_id is None or report.status is None:
                continue
            key = self._report_key(report)
            if key in self._seen_reports:
                continue
            self._seen_reports.add(key)
            print(
                f"[report] agent={report.agent_id} task={report.task_id} "
                f"status={report.status} progress={report.progress:.2f}"
            )

        return None


def build_pipeline() -> Pipeline:
    pipe = Pipeline("multi_agent_communication_demo")

    tasks = TaskGenerator() @ Rate(hz=10.0)
    scout_bidder = BiddingAgent(
        agent_id="scout",
        skills={"inspect", "map"},
        speed=1.7,
        cost_bias=1.4,
    ) @ Rate(hz=20.0)
    carrier_bidder = BiddingAgent(
        agent_id="carrier",
        skills={"lift", "transport"},
        speed=1.1,
        cost_bias=0.8,
    ) @ Rate(hz=20.0)
    auctioneer = Auctioneer() @ Rate(hz=20.0)
    scout_exec = ExecutionAgent(agent_id="scout") @ Rate(hz=10.0)
    carrier_exec = ExecutionAgent(agent_id="carrier") @ Rate(hz=10.0)
    monitor = MissionMonitor() @ Rate(hz=20.0)

    pipe.connect(tasks, scout_bidder, map={"packet": "packet"}, sync=Latest())
    pipe.connect(tasks, carrier_bidder, map={"packet": "packet"}, sync=Latest())
    pipe.connect(tasks, auctioneer, map={"packet": "task"}, sync=Latest())

    pipe.connect(scout_bidder, auctioneer, map={"packet": "scout_bid"}, sync=Latest())
    pipe.connect(carrier_bidder, auctioneer, map={"packet": "carrier_bid"}, sync=Latest())

    pipe.connect(tasks, scout_exec, map={"packet": "task"}, sync=Latest())
    pipe.connect(tasks, carrier_exec, map={"packet": "task"}, sync=Latest())
    pipe.connect(auctioneer, scout_exec, map={"packet": "assignment"}, sync=Latest())
    pipe.connect(auctioneer, carrier_exec, map={"packet": "assignment"}, sync=Latest())

    pipe.connect(tasks, monitor, map={"packet": "task"}, sync=Latest())
    pipe.connect(auctioneer, monitor, map={"packet": "assignment"}, sync=Latest())
    pipe.connect(scout_exec, monitor, map={"packet": "scout_report"}, sync=Latest())
    pipe.connect(carrier_exec, monitor, map={"packet": "carrier_report"}, sync=Latest())

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated multi-agent communication demo.")
    p.add_argument("--mode", choices=["step", "run"], default="step")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--backend", choices=["multiprocessing", "dora"], default="multiprocessing")
    p.add_argument("--duration", type=float, default=6.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline()

    print("=" * 66)
    print("Integrated Multi-Agent Communication Demo")
    print(f"mode={args.mode} backend={args.backend} steps={args.steps} dt={args.dt}")
    print("=" * 66)

    if args.mode == "step":
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
        pipe.close_stepper()
    else:
        pipe.run(backend=args.backend, duration=args.duration, blocking=True)


if __name__ == "__main__":
    main()
