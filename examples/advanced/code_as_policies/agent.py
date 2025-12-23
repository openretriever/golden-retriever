"""
LLM Agent to generate Code Policies.
"""

import os
import logging
import re
from typing import Optional
from .prompts import CAP_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class CodeGenAgent:
    """
    Generates Python code from natural language instructions.
    """

    def __init__(self, model: str = None, mock: bool = False):
        # Allow override via env var, default to a widespread model if not set
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.client = None
        self.mock = mock
        if not self.mock:
            self._init_client()

    def _init_client(self):
        try:
            # We default to Gemini for this example per implementation plan
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not set. Switching to Mock mode.")
                self.mock = True
                return
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Initialized Gemini Agent: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to init Gemini client: {e}")
            self.mock = True

    def generate_code(self, instruction: str, object_names: list[str]) -> str:
        """
        Generate code for the given instruction and context.
        """
        if self.mock:
            logger.info("[Agent] Generating MOCK code.")
            return self._mock_generation(instruction)

        if not self.client:
            return "print('Error: LLM client not initialized.')"

        context_str = f"Objects available: {object_names}"
        full_prompt = f"{CAP_SYSTEM_PROMPT}\n\nUser: {instruction}\nContext: {context_str}\n\nResponse:"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            text = response.text
            return self._extract_code(text)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            safe_error = str(e).replace("'", "").replace('"', "")
            return f"print('Generation Error: {safe_error}')"

    def _mock_generation(self, instruction: str) -> str:
        """
        Return a hardcoded script for testing.
        """
        return """
# Mock Script
say("Starting mock sequence for: " + "'''""" + instruction + """'''")
red_pos = get_object_position("red_block")
# pick("red_block") # Skipped for speed in mock
move_to(red_pos)
say("Simulating pick...")
blue_pos = get_object_position("blue_block")
# place(blue_pos)
move_to(blue_pos)
say("Simulating place...")
say("Task complete!")
"""

    def _extract_code(self, text: str) -> str:
        """
        Extract code from markdown block.
        """
        # Regex for ```python ... ```
        pattern = r"```python(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: maybe just ``` ... ```
        pattern_generic = r"```(.*?)```"
        match_generic = re.search(pattern_generic, text, re.DOTALL)
        if match_generic:
            return match_generic.group(1).strip()
            
        # Fallback: assume raw text is code
        return text.strip()
