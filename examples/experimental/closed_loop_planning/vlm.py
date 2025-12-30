import logging
import os
from typing import Any, Sequence

from retriever.types.symbolic import Object, Predicate, State

logger = logging.getLogger(__name__)

# Check for Gemini
try:
    from google import genai
    HAS_GENAI = True
except ImportError as e:
    HAS_GENAI = False
    logger.warning(f"google-genai import failed: {e}. VLM will fallback to mock.")
    import traceback
    traceback.print_exc()

class VisualPredicate(Predicate):
    """A predicate that uses a VLM to classify the state from an image."""

    def __init__(self, name: str, types: Sequence[Any], prompt: str):
        super().__init__(name, types, self.classifier)
        self.prompt = prompt
        self._client = None
        # Client initialized lazily to avoid pickling issues

    def classifier(self, state: State, objects: Sequence[Object]) -> bool:
        """
        Classifies the predicate using the VLM.
        Expects 'image' to be present in the state (as a global feature or attached to an object).
        For this demo, we assume the 'image' is in state.data['global_image'].
        """
        # 0. Initialize Client if needed (Lazy Init for Multiprocessing)
        if HAS_GENAI and self._client is None:
             api_key = os.environ.get("GEMINI_API_KEY")
             if api_key:
                 self._client = genai.Client(api_key=api_key)
             else:
                 logger.warning("GEMINI_API_KEY not found in environment.")

        # 1. Get Image
        # In a real system, we might crop based on object bbox
        # Access underlying data dict since State doesn't have .get()
        image = state.data.get("global_image")

        if image is None:
            # Fallback to symbolic/geometric check if image is missing
            # (e.g. simulation might not render images)
            return self._geometric_fallback(state, objects)

        # 2. VLM Call
        if not HAS_GENAI or not self._client:
            reason = []
            if not HAS_GENAI: reason.append("HAS_GENAI=False")
            if not self._client: reason.append("Client=None")
            logger.info(f"[{self.name}] VLM not available ({', '.join(reason)}). Using fallback.")
            return self._geometric_fallback(state, objects)

        try:
            # Prepare prompt
            full_prompt = f"{self.prompt} Answer only 'yes' or 'no'."

            # Generate Answer
            # New SDK usage
            response = self._client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=[full_prompt, image]
            )
            answer = response.text.strip().lower()

            # Log to Rerun
            import rerun as rr
            try:
                # Log the image being analyzed
                # Assuming image is PIL Image or compatible
                # If PIL, convert to numpy
                import numpy as np
                if hasattr(image, 'convert'):
                     img_np = np.array(image.convert("RGB"))
                     rr.log(f"vlm/{self.name}/image", rr.Image(img_np))

                rr.log(f"vlm/{self.name}/prompt", rr.TextDocument(f"Q: {self.prompt}\nA: {answer}"))
            except Exception as e_rr:
                logger.warning(f"Rerun logging failed: {e_rr}")

            logger.info(f"[{self.name}] VLM Response: {answer}")
            return "yes" in answer

        except Exception as e:
            logger.error(f"[{self.name}] VLM Error: {e}")
            return self._geometric_fallback(state, objects)

    def _geometric_fallback(self, state: State, objects: Sequence[Object]) -> bool:
        """Fallback logic if VLM fails or no image."""
        # By default, return False or try to find a legacy classifier?
        # For IsOpen on the door object (index 0), we can look at state vector
        if self.name == "IsOpen":
            door = objects[0]
            if door in state:
                return state[door][2] > 0.5
        return False
