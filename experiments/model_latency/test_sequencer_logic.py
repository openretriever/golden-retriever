# import pytest
from experiments.closed_loop_planning.flows.sequencer_policy import PolicySequencerFlow, SequencerState, SequencerInput
from experiments.closed_loop_planning.types.flow_types import PlannerOutput, MonitorOutput, PolicyFeedback, PlannerResult

# @pytest.fixture
def sequencer():
    return PolicySequencerFlow(name="TestSequencer")

def test_initial_state(sequencer):
    assert sequencer.state == SequencerState.IDLE
    assert sequencer.current_step_idx == 0

def test_plan_ingestion(sequencer):
    # Mock Planner Output
    plan_steps = ["Pick up red block", "Place in bowl"]
    
    class MockOption:
        def __init__(self, desc):
            self.description = desc
        def __repr__(self):
            return self.description
            
    mock_plan = [MockOption(s) for s in plan_steps]
    planner_output = PlannerOutput(result=PlannerResult(plan=mock_plan, reasoning="Test", belief_update="Test"))
    
    # Run Step 1: Ingest Plan
    input_data = SequencerInput(plan=planner_output)
    output = sequencer.run(input=input_data)
    
    assert sequencer.state == SequencerState.EXECUTING
    assert sequencer.plan == plan_steps
    assert sequencer.current_step_idx == 0
    assert output.prompt == "Pick up red block"

def test_execution_flow_hybrid(sequencer):
    # Setup state
    plan_steps = ["Step 1", "Step 2"]
    sequencer.plan = plan_steps
    sequencer.state = SequencerState.EXECUTING
    sequencer.current_step_idx = 0
    
    # 1. Policy Running, Low Progress -> Stay Executing
    feedback = PolicyFeedback(task_progress=0.5)
    output = sequencer.run(input=SequencerInput(feedback=feedback))
    assert sequencer.state == SequencerState.EXECUTING
    
    # 2. Policy Done (Progress ~1.0) -> Verify
    feedback = PolicyFeedback(task_progress=1.0)
    output = sequencer.run(input=SequencerInput(feedback=feedback))
    assert sequencer.state == SequencerState.VERIFYING
    
    # 3. Vision Unknown -> Optimistic Advance
    monitor = MonitorOutput(visual_status="UNKNOWN")
    output = sequencer.run(input=SequencerInput(monitor=monitor))
    assert sequencer.state == SequencerState.EXECUTING
    assert sequencer.current_step_idx == 1
    assert output.prompt == "Step 2"

def test_execution_failure_recovery(sequencer):
    # Setup state
    plan_steps = ["Step 1"]
    sequencer.plan = plan_steps
    sequencer.state = SequencerState.VERIFYING
    sequencer.current_step_idx = 0
    
    # Vision says NOT DONE -> Retry (Back to Executing step 0)
    monitor = MonitorOutput(visual_status="IN_PROGRESS")
    output = sequencer.run(input=SequencerInput(monitor=monitor))
    
    # Should revert to executing the same step
    assert sequencer.state == SequencerState.EXECUTING
    assert sequencer.current_step_idx == 0
    assert output.prompt == "Step 1"

def test_completion(sequencer):
    plan_steps = ["Step 1"]
    sequencer.plan = plan_steps
    sequencer.state = SequencerState.EXECUTING
    sequencer.current_step_idx = 1 # Past end
    
    output = sequencer.run(input=SequencerInput())
    assert sequencer.state == SequencerState.COMPLETED

if __name__ == "__main__":
    print("Running tests...")
    s = sequencer()
    test_initial_state(s)
    print("test_initial_state passed")
    
    s = sequencer()
    test_plan_ingestion(s)
    print("test_plan_ingestion passed")
    
    s = sequencer()
    test_execution_flow_hybrid(s)
    print("test_execution_flow_hybrid passed")
    
    s = sequencer()
    test_execution_failure_recovery(s)
    print("test_execution_failure_recovery passed")
    
    s = sequencer()
    test_completion(s)
    print("test_completion passed")
    print("All tests passed!")
