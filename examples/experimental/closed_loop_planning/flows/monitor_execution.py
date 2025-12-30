from retriever.flow import Flow

from ..types.flow_types import MonitorInput, MonitorOutput
from ..types.options import ReplanConfig


class ExecutionMonitorFlow(Flow[MonitorInput, MonitorOutput]):
    """Monitors execution and decides when to trigger replanning."""

    def __init__(self, name: str = "monitor"):
        self.name = name
        self._last_state_id = None

    def step(self, inp: MonitorInput) -> MonitorOutput:
        should_replan = False
        reason = ""

        # 1. Replan on executor failure
        if inp.executor_status == "failure":
            should_replan = True
            reason = "executor_failure"

        # 2. Track state changes (for future use)
        current_id = id(inp.state) if inp.state else 0
        self._last_state_id = current_id

        # 3. Default: trigger periodic replan check
        if not should_replan:
            should_replan = True
            reason = "periodic_check"

        return MonitorOutput(
            replan_config=ReplanConfig(should_replan=should_replan, reason=reason)
        )
