from dataclasses import dataclass
from typing import Optional

from retriever.flow import Flow, io

from ..types.belief import BeliefState


@io
@dataclass
class DebugInput:
    data: Optional[BeliefState]

@io
@dataclass
class DebugOutput:
    pass

class DebugFlow(Flow[DebugInput, DebugOutput]):
    def __init__(self, name="DebugFlow"):
        self.name = name

    def step(self, inp: DebugInput) -> DebugOutput:
        print(f"[{self.name}] RECEIVED DATA: {type(inp.data)}")
        try:
            print(f"[{self.name}] Data content: {inp.data}")
        except:
            print(f"[{self.name}] Could not print content.")
        return DebugOutput()
