"""
VLM Utility for Robot Planning and Perception
"""

import os
import re
import logging
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

# Suppress Google GenAI AFC (Automatic Function Calling) INFO logs
logging.getLogger('google.genai.models').setLevel(logging.INFO)


class VLMPlanner:
    """VLM Planner with lazy client initialization for multiprocessing compatibility."""

    def __init__(self, model_name: str = "gemini-robotics-er-1.5-preview"):
        self.model_name = model_name
        self.is_busy = False
        self.system_prompt: Optional[str] = None
        # Lazy initialization - clients are created on first use
        self._google_client = None
        self._openai_client = None
        self._initialized = False

    def _ensure_initialized(self):
        """Initialize clients on first use (after multiprocessing fork)."""
        if self._initialized:
            return

        if "gemini" in self.model_name:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if api_key:
                # Note: Remove api_version to use default (stable)
                self._google_client = genai.Client(api_key=api_key)

        if "gpt" in self.model_name or os.getenv("OPENAI_API_BASE"):
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._openai_client = OpenAI(
                    api_key=api_key,
                    base_url=os.getenv("OPENAI_API_BASE")
                )

        self._initialized = True

    def _get_system_prompt(self, mode: str) -> str:
        if self.system_prompt:
            return self.system_prompt
        if mode == "perception":
            return """You are a robotic vision assistant. Describe the scene concisely. 
Identify main objects, their positions, and potential actions (affordances). 
Keep it brief and factual."""
        
        elif mode == "seasoning":
            return """You are a robotic chef assistant. 
1. Identify spice containers (salt, pepper, spices).
2. Identify target food items.
3. Plan a seasoning action.
Output your reasoning followed by the action in JSON:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "pick_up_spice_container"}]
Coordinates are normalized 0-1000."""

        elif mode == "chess":
            return """You are an expert Chess/Gomoku strategist.
Analyze the board state.
Output your reasoning followed by the move in JSON:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "place_piece"}]
Coordinates are normalized 0-1000."""

        # Default: detect top 3 salient objects
        # Default: detect top 3 salient objects
        return """You are a robotic vision assistant.
Analyze the image and identify the TOP 3 most salient/interesting objects.

STEP 1: Provide a brief 1-sentence reasoning about what you see and why these objects are salient.
STEP 2: Output a JSON array with bounding boxes.

Format:
[Reasoning text here]

[
  {"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"},
  ...
]

Coordinates are normalized 0-1000 (0=top/left, 1000=bottom/right).
ALWAYS output exactly 3 objects with bounding boxes."""

    def _extract_reasoning_and_json(self, text: str) -> Tuple[str, str]:
        """Extracts reasoning (raw text) and the first JSON block found."""
        json_pattern = r"\[\s*\{.*\}\s*\]|\{.*\}"
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group()
            reasoning = text[:match.start()].strip()
            return reasoning, json_str
        return text.strip(), "[]"

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to max dimension 768px for optimal VLM latency."""
        max_dim = 768
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))
        return image

    def _process_image_for_openai(self, image: Image.Image) -> str:
        """Compress image for efficient OpenAI Base64 transfer."""
        # Resize first
        image = self._resize_image(image)
        
        # Compress: JPEG quality 70 is a good balance
        from io import BytesIO
        import base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=70)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def plan(self, image: Image.Image, instruction: str, mode: str = "general") -> Dict[str, Any]:
        if self.is_busy:
            return {"status": "busy", "reasoning": "Inference in progress..."}

        self._ensure_initialized()
        self.is_busy = True
        try:
            system_prompt = self._get_system_prompt(mode)
            full_prompt = f"{instruction}\n\nMode: {mode.upper()}"

            if self._google_client and "gemini" in self.model_name:
                # Resize image for faster upload
                image = self._resize_image(image)

                # Configure generation
                config_params = {
                    "system_instruction": system_prompt,
                    "temperature": 0.4,
                }

                # Use thinking config for Robotics-ER models or if explicitly requested
                if "gemini-robotics" in self.model_name:
                    config_params["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)

                response = self._google_client.models.generate_content(
                    model=self.model_name,
                    contents=[full_prompt, image],
                    config=genai_types.GenerateContentConfig(**config_params)
                )
                text = response.text

            elif self._openai_client:
                # OpenAI Vision API with optimized base64 image
                img_base64 = self._process_image_for_openai(image)
                
                # Prepare params for OpenAI
                params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": full_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}",
                                        "detail": "auto"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_completion_tokens": 1024,
                }

                # Only add temperature if NOT using o1/o3/gpt-5-mini which enforce default
                # (Simple heuristic: if "o1" or "o3" or "mini" in name, skip temp)
                if not any(x in self.model_name for x in ["o1", "o3", "gpt-5"]):
                     params["temperature"] = 0.4

                response = self._openai_client.chat.completions.create(**params)
                text = response.choices[0].message.content

            else:
                return {"status": "error", "message": "No VLM client initialized."}

            reasoning, coordinates_json = self._extract_reasoning_and_json(text)

            return {
                "status": "success",
                "reasoning": reasoning,
                "coordinates": coordinates_json,
                "raw_text": text
            }
        except Exception as e:
            if "429" in str(e):
                return {"status": "rate_limited", "reasoning": "Rate limit hit (429). Skipping."}
            return {"status": "error", "message": str(e)}
        finally:
            self.is_busy = False

    def describe(self, image: Image.Image) -> Dict[str, Any]:
        """Convenience method for scene description."""
        return self.plan(image, "Describe this scene.", mode="perception")


class BeliefVLMPlanner(VLMPlanner):
    """
    Extensions to VLMPlanner that maintains a 'current_belief' state
    updates it with every observation.
    """
    def __init__(self, model_name: str = "gemini-robotics-er-1.5-preview"):
        super().__init__(model_name)
        # State: Belief History
        self.belief_history = []
        self.current_belief = "No objects detected yet."

    def _get_system_prompt(self, mode: str) -> str:
        if self.system_prompt:
            return self.system_prompt
        if mode and mode != "general":
            return super()._get_system_prompt(mode)
        # Override default to force Stateful Belief Update
        # (Could also support other modes if wrapped in belief logic)
        
        return """You are a robotic vision assistant with persistent memory.
Analyze the image and the CURRENT BELIEF to update your understanding of the scene.

STEP 1: Update your Belief.
- Combine the 'Current Belief' with the 'New Image'.
- If new objects appear, add them.
- If objects moved, update their location.
- If objects disappeared, note that they are 'not currently visible' but remember them.

STEP 2: Output a JSON array with bounding boxes for relevant objects.

Format:
[Updated Belief Summary]

[
  {"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"},
  ...
]

Coordinates are normalized 0-1000 (0=top/left, 1000=bottom/right)."""

    def plan(self, image: Image.Image, instruction: str, mode: str = "general") -> Dict[str, Any]:
        if self.is_busy:
             return {"status": "busy", "reasoning": "Inference in progress..."}

        # Inject Belief State into prompt
        # We need to hook into the prompt generation. 
        # Since 'plan' in base class does heavy lifting of API calls, 
        # let's shadow the prompt construction logic by overriding plan,
        # OR we can just duplicate the 'plan' method since it's the core loop.
        
        # Duplicating 'plan' logic for simplicity with State injection:
        self._ensure_initialized()
        self.is_busy = True
        try:
            system_prompt = self._get_system_prompt(mode)
            
            # Inject Belief State
            full_prompt = (
                f"{instruction}\n\n"
                f"--- CURRENT BELIEF ---\n{self.current_belief}\n"
                f"----------------------\n\n"
                f"Mode: {mode.upper()}"
            )
            
            # --- START COPY FROM BASE (API CALLS) ---
            # Ideally we refactor base to helper _call_api(full_prompt, image, system_prompt)
            # but for now direct implementation to ensure correctness.
            
            text = ""
            if self._google_client and "gemini" in self.model_name:
                image = self._resize_image(image)
                config_params = {
                    "system_instruction": system_prompt,
                    "temperature": 0.4,
                }
                if "gemini-robotics" in self.model_name:
                    config_params["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)

                response = self._google_client.models.generate_content(
                    model=self.model_name,
                    contents=[full_prompt, image],
                    config=genai_types.GenerateContentConfig(**config_params)
                )
                text = response.text
            elif self._openai_client:
                 img_base64 = self._process_image_for_openai(image)
                 params = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                                {"type": "text", "text": full_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]}
                    ],
                    "max_completion_tokens": 1024,
                 }
                 if not any(x in self.model_name for x in ["o1", "o3", "gpt-5"]):
                      params["temperature"] = 0.4
                 response = self._openai_client.chat.completions.create(**params)
                 text = response.choices[0].message.content
            else:
                 return {"status": "error", "message": "No VLM client initialized."}
            # --- END COPY FROM BASE ---

            reasoning, coordinates_json = self._extract_reasoning_and_json(text)

            # Update Belief State
            if reasoning.strip():
                 self.current_belief = reasoning
                 self.belief_history.append(reasoning)
                 if len(self.belief_history) > 10:
                     self.belief_history.pop(0)

            return {
                "status": "success",
                "reasoning": reasoning,
                "coordinates": coordinates_json,
                "raw_text": text
            }
        except Exception as e:
            if "429" in str(e):
                 return {"status": "rate_limited", "reasoning": "Rate limit hit (429). Skipping."}
            return {"status": "error", "message": str(e)}
        finally:
            self.is_busy = False
