from typing import Optional

from retriever.flow import Flow

from ..types.belief import (
    BeliefState,
    BeliefUpdateInput,
    BeliefUpdateOutput,
    EpistemicState,
    EpistemicValue,
)


class BeliefUpdaterFlow(Flow[BeliefUpdateInput, BeliefUpdateOutput]):
    """State estimation flow with epistemic tracking."""

    def __init__(self, max_history: int = 10, name: str = "BeliefUpdater"):
        self.name = name
        self.max_history = max_history
        self.current_belief: Optional[BeliefState] = None
        # Rerun initialized globally

    def step(self, inp: BeliefUpdateInput) -> BeliefUpdateOutput:
        import rerun as rr

        obs = inp.observation
        # Debug logging
        # print(f"[{self.name}] Input: Obs={'Present' if obs else 'None'}, Atoms={len(inp.visible_atoms)}")

        prev = inp.prev_belief or self.current_belief
        action = inp.action
        observed_atoms = inp.visible_atoms

        # If no observation, return previous or empty belief
        if obs is None:
            if prev is not None:
                return BeliefUpdateOutput(belief=prev)
            # Create empty belief
            return BeliefUpdateOutput(belief=BeliefState(data={}))

        # Build new belief from observation
        epistemic = EpistemicState()

        # Initialize from CURRENT observation (Assume observable = TRUE)
        for atom in observed_atoms:
            epistemic.update(atom, EpistemicValue.TRUE)

        new_belief = BeliefState(
            data=obs.data.copy(),
            visual_atoms={},
            epistemic=epistemic,
            action_history=[],
            raw_observation=inp.raw_observation,
        )

        # Carry forward from previous belief
        if prev is not None:
            # Action history
            new_belief.action_history = prev.action_history.copy()
            if action is not None:
                new_belief.action_history.append(action)
            if len(new_belief.action_history) > self.max_history:
                new_belief.action_history = new_belief.action_history[-self.max_history:]

            # Epistemic state: merge with prior
            for atom in prev.epistemic.known_true:
                if new_belief.is_unknown(atom):
                    new_belief.epistemic.update(atom, EpistemicValue.TRUE)

            for atom in prev.epistemic.known_false:
                if new_belief.is_unknown(atom):
                    new_belief.epistemic.update(atom, EpistemicValue.FALSE)

            # Visual atoms: carry forward
            new_belief.visual_atoms = prev.visual_atoms.copy()


        if len(new_belief.epistemic.known_true) > 0:
            # Log to Rerun
            belief_text = "\n".join([str(atom) for atom in new_belief.epistemic.known_true])
            rr.log("belief/known_atoms", rr.TextDocument(belief_text))

        self.current_belief = new_belief
        return BeliefUpdateOutput(belief=new_belief)
