"""Concise language/grounding flows for advanced examples."""

from __future__ import annotations

from retriever.flow import Flow
from retriever.types.language import Caption, GroundedPhrase, PlanStepText, PlanText, ReferringExpression
from retriever.types.perception import DetectionBatch


class CaptionSource(Flow[None, Caption]):
    def __init__(self, *, captions: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.captions = tuple(captions or (
            'red cube on the left near the blue cylinder',
            'pick the red cube and place it at the goal',
            'inspect the blue object before moving',
        ))

    def init_config(self) -> dict:
        return {'captions': self.captions}

    def reset(self) -> None:
        self._index = 0

    def step(self, _):  # type: ignore[override]
        text = self.captions[self._index % len(self.captions)]
        self._index += 1
        return Caption(text=text, source='golden.language_examples')


class CaptionPlanner(Flow[Caption, PlanText]):
    def step(self, caption: Caption) -> PlanText:
        text = caption.text.lower()
        steps: list[PlanStepText] = [PlanStepText(index=0, text=f'inspect scene: {caption.text}', action_label='inspect')]
        next_index = 1
        if 'red' in text:
            steps.append(PlanStepText(index=next_index, text='focus on the red object', action_label='focus'))
            next_index += 1
        if 'pick' in text or 'grasp' in text:
            steps.append(PlanStepText(index=next_index, text='pick the target object', action_label='pick'))
            next_index += 1
        if 'place' in text or 'goal' in text:
            steps.append(PlanStepText(index=next_index, text='place the target at the goal', action_label='place'))
        return PlanText(steps=tuple(steps), summary=caption.text, source='caption_planner')


class PlanTextPrinter(Flow[PlanText, None]):
    def step(self, plan: PlanText) -> None:
        summary = [f'{step.index}:{step.action_label or "step"}:{step.text}' for step in plan.steps]
        print(f'plan={summary}')
        return None


class ReferringExpressionSource(Flow[None, ReferringExpression]):
    def __init__(self, *, expressions: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self.expressions = tuple(expressions or ('the red object', 'the blue object', 'the left target'))

    def init_config(self) -> dict:
        return {'expressions': self.expressions}

    def reset(self) -> None:
        self._index = 0

    def step(self, _):  # type: ignore[override]
        text = self.expressions[self._index % len(self.expressions)]
        self._index += 1
        return ReferringExpression(text=text)


class DetectionGrounder(Flow[(ReferringExpression, DetectionBatch), GroundedPhrase]):
    def step(self, inp):  # type: ignore[override]
        if isinstance(inp, tuple):
            expression, batch = inp
        else:
            expression = inp.ReferringExpression
            batch = inp.DetectionBatch
        text = expression.text.lower()
        chosen = None
        for det in batch.detections:
            if det.label.lower() in text:
                chosen = det
                break
        if chosen is None and batch.detections:
            chosen = batch.detections[0]
        return GroundedPhrase(
            text=expression.text,
            referent_label=None if chosen is None else chosen.label,
            confidence=None if chosen is None else chosen.confidence,
            frame_index=batch.frame_index,
            span=expression.span,
        )


class GroundedPhrasePrinter(Flow[GroundedPhrase, None]):
    def step(self, phrase: GroundedPhrase) -> None:
        print(
            f'grounded=text={phrase.text!r} referent={phrase.referent_label!r} frame={phrase.frame_index!r} confidence={phrase.confidence!r}'
        )
        return None
