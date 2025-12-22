"""
Text-Based RL Environments for LLM Agents.

This module provides text-based environments suitable for
testing LLM agents in multi-turn interactive settings.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvState:
    """State of a text environment."""
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)  # [(query, response), ...]
    step_count: int = 0
    total_reward: float = 0.0


class TwentyQuestionsEnv:
    """
    A text-based "20 Questions" guessing game environment.
    
    The environment thinks of a secret word.
    The LLM agent asks yes/no questions to guess it.
    
    Rewards:
        +10.0  Correct guess
        +0.1   Good question (narrows down possibilities)
        -0.05  Vague or repeated question
        -1.0   Wrong final guess
        -5.0   Running out of questions
    
    Actions:
        - "ask: <question>" - Ask a yes/no question
        - "guess: <word>" - Make a final guess
    """
    
    # Word categories for variety
    WORD_CATEGORIES = {
        "animals": ["elephant", "penguin", "giraffe", "dolphin", "butterfly", "octopus"],
        "fruits": ["apple", "banana", "strawberry", "mango", "pineapple", "watermelon"],
        "objects": ["laptop", "umbrella", "telescope", "bicycle", "piano", "lighthouse"],
        "places": ["mountain", "ocean", "desert", "forest", "island", "volcano"],
        "vehicles": ["helicopter", "submarine", "motorcycle", "sailboat", "rocket", "train"],
    }
    
    # Flatten for random selection
    ALL_WORDS = [word for words in WORD_CATEGORIES.values() for word in words]
    
    # Simple knowledge base for answering questions
    WORD_PROPERTIES = {
        "elephant": {"alive": True, "big": True, "has_legs": True, "can_fly": False, "in_water": False, "gray": True},
        "penguin": {"alive": True, "big": False, "has_legs": True, "can_fly": False, "in_water": True, "black_white": True},
        "giraffe": {"alive": True, "big": True, "has_legs": True, "can_fly": False, "in_water": False, "spotted": True},
        "dolphin": {"alive": True, "big": False, "has_legs": False, "can_fly": False, "in_water": True, "gray": True},
        "butterfly": {"alive": True, "big": False, "has_legs": True, "can_fly": True, "in_water": False, "colorful": True},
        "octopus": {"alive": True, "big": False, "has_legs": True, "can_fly": False, "in_water": True, "tentacles": True},
        "apple": {"alive": False, "edible": True, "round": True, "red": True, "sweet": True, "fruit": True},
        "banana": {"alive": False, "edible": True, "round": False, "yellow": True, "sweet": True, "fruit": True},
        "strawberry": {"alive": False, "edible": True, "round": False, "red": True, "sweet": True, "fruit": True},
        "mango": {"alive": False, "edible": True, "round": False, "orange": True, "sweet": True, "fruit": True},
        "pineapple": {"alive": False, "edible": True, "round": False, "yellow": True, "sweet": True, "fruit": True},
        "watermelon": {"alive": False, "edible": True, "round": True, "green": True, "sweet": True, "fruit": True},
        "laptop": {"alive": False, "electronic": True, "portable": True, "has_screen": True, "metal": True},
        "umbrella": {"alive": False, "electronic": False, "portable": True, "has_screen": False, "for_rain": True},
        "telescope": {"alive": False, "electronic": False, "portable": True, "has_screen": False, "for_seeing": True},
        "bicycle": {"alive": False, "electronic": False, "portable": False, "has_wheels": True, "vehicle": True},
        "piano": {"alive": False, "electronic": False, "portable": False, "musical": True, "has_keys": True},
        "lighthouse": {"alive": False, "electronic": True, "portable": False, "tall": True, "near_water": True},
        "mountain": {"alive": False, "natural": True, "big": True, "tall": True, "has_snow": True},
        "ocean": {"alive": False, "natural": True, "big": True, "water": True, "salty": True},
        "desert": {"alive": False, "natural": True, "big": True, "hot": True, "sandy": True},
        "forest": {"alive": False, "natural": True, "big": True, "green": True, "has_trees": True},
        "island": {"alive": False, "natural": True, "big": False, "water": True, "surrounded": True},
        "volcano": {"alive": False, "natural": True, "big": True, "hot": True, "dangerous": True},
        "helicopter": {"alive": False, "electronic": True, "can_fly": True, "vehicle": True, "has_blades": True},
        "submarine": {"alive": False, "electronic": True, "can_fly": False, "vehicle": True, "in_water": True},
        "motorcycle": {"alive": False, "electronic": False, "can_fly": False, "vehicle": True, "has_wheels": True},
        "sailboat": {"alive": False, "electronic": False, "can_fly": False, "vehicle": True, "in_water": True},
        "rocket": {"alive": False, "electronic": True, "can_fly": True, "vehicle": True, "goes_to_space": True},
        "train": {"alive": False, "electronic": True, "can_fly": False, "vehicle": True, "has_wheels": True},
    }
    
    def __init__(
        self,
        max_questions: int = 20,
        seed: Optional[int] = None,
    ):
        """
        Initialize the 20 Questions environment.
        
        Args:
            max_questions: Maximum number of questions allowed
            seed: Random seed for reproducibility
        """
        self.max_questions = max_questions
        self.rng = random.Random(seed)
        
        # State
        self.secret: Optional[str] = None
        self.category: Optional[str] = None
        self.question_count = 0
        self.history: List[Tuple[str, str]] = []
        self.done = False
        self.total_reward = 0.0
        self.asked_questions: set = set()
        
    def reset(self) -> EnvState:
        """Reset the environment with a new secret word."""
        # Pick random category and word
        self.category = self.rng.choice(list(self.WORD_CATEGORIES.keys()))
        self.secret = self.rng.choice(self.WORD_CATEGORIES[self.category])
        
        self.question_count = 0
        self.history = []
        self.done = False
        self.total_reward = 0.0
        self.asked_questions = set()
        
        observation = self._get_initial_observation()
        
        return EnvState(
            observation=observation,
            reward=0.0,
            done=False,
            info={"category": self.category, "secret": "[HIDDEN]"},
            history=[],
            step_count=0,
            total_reward=0.0,
        )
    
    def _get_initial_observation(self) -> str:
        """Generate the initial game prompt."""
        return f"""🎮 Welcome to 20 Questions!

I'm thinking of a word from the category: **{self.category}**

Rules:
- Ask yes/no questions to narrow down the answer
- You have {self.max_questions} questions maximum
- Format your questions as: "ask: <your question>"
- When ready to guess, say: "guess: <your answer>"

Good luck! What's your first question?"""
    
    def step(self, action: str) -> EnvState:
        """
        Process an action (question or guess) and return the result.
        
        Args:
            action: Either "ask: <question>" or "guess: <word>"
            
        Returns:
            EnvState with observation, reward, and done status
        """
        if self.done:
            return self._get_state("Game is already over. Call reset() to play again.", 0.0, True)
        
        action = action.strip().lower()
        
        # Parse action
        if action.startswith("guess:"):
            return self._handle_guess(action[6:].strip())
        elif action.startswith("ask:"):
            return self._handle_question(action[4:].strip())
        else:
            # Try to infer intent
            if "?" in action:
                return self._handle_question(action)
            else:
                return self._get_state(
                    "I didn't understand that. Please use 'ask: <question>' or 'guess: <word>'.",
                    -0.1,
                    False,
                )
    
    def _handle_question(self, question: str) -> EnvState:
        """Handle a yes/no question."""
        self.question_count += 1
        
        # Check for repeated questions
        q_normalized = question.lower().strip()
        if q_normalized in self.asked_questions:
            response = "You already asked that! Try a different question."
            reward = -0.1
        else:
            self.asked_questions.add(q_normalized)
            response, reward = self._answer_question(question)
        
        # Add to history
        self.history.append((question, response))
        
        # Check if out of questions
        remaining = self.max_questions - self.question_count
        if remaining <= 0:
            self.done = True
            observation = f"❌ Out of questions! The word was: **{self.secret}**"
            reward = -5.0
        else:
            observation = f"Q{self.question_count}: {question}\nA: {response}\n\n[{remaining} questions remaining]"
        
        self.total_reward += reward
        
        return self._get_state(observation, reward, self.done)
    
    def _answer_question(self, question: str) -> Tuple[str, float]:
        """
        Generate an answer to a yes/no question.
        
        Uses simple keyword matching against known properties.
        """
        question_lower = question.lower()
        properties = self.WORD_PROPERTIES.get(self.secret, {})
        
        # Check various question patterns
        if "alive" in question_lower or "living" in question_lower:
            answer = "Yes" if properties.get("alive", False) else "No"
        elif "big" in question_lower or "large" in question_lower:
            answer = "Yes" if properties.get("big", False) else "No"
        elif "fly" in question_lower:
            answer = "Yes" if properties.get("can_fly", False) else "No"
        elif "water" in question_lower or "swim" in question_lower:
            answer = "Yes" if properties.get("in_water", False) else "No"
        elif "eat" in question_lower or "edible" in question_lower or "food" in question_lower:
            answer = "Yes" if properties.get("edible", False) else "No"
        elif "electronic" in question_lower or "electric" in question_lower:
            answer = "Yes" if properties.get("electronic", False) else "No"
        elif "wheel" in question_lower:
            answer = "Yes" if properties.get("has_wheels", False) else "No"
        elif "vehicle" in question_lower or "transport" in question_lower:
            answer = "Yes" if properties.get("vehicle", False) else "No"
        elif "natural" in question_lower or "nature" in question_lower:
            answer = "Yes" if properties.get("natural", False) else "No"
        elif "fruit" in question_lower:
            answer = "Yes" if properties.get("fruit", False) else "No"
        elif "animal" in question_lower:
            answer = "Yes" if properties.get("alive", False) and not properties.get("fruit", False) else "No"
        elif "leg" in question_lower:
            answer = "Yes" if properties.get("has_legs", False) else "No"
        elif "round" in question_lower or "circular" in question_lower:
            answer = "Yes" if properties.get("round", False) else "No"
        elif "red" in question_lower:
            answer = "Yes" if properties.get("red", False) else "No"
        elif "green" in question_lower:
            answer = "Yes" if properties.get("green", False) else "No"
        elif "yellow" in question_lower:
            answer = "Yes" if properties.get("yellow", False) else "No"
        elif "hot" in question_lower or "warm" in question_lower:
            answer = "Yes" if properties.get("hot", False) else "No"
        elif "cold" in question_lower:
            answer = "Yes" if properties.get("has_snow", False) or self.secret == "penguin" else "No"
        else:
            # Generic fallback - try to be helpful
            answer = "I can only answer yes/no questions about the word."
            return answer, 0.0
        
        # Reward for good questions
        reward = 0.1 if answer in ["Yes", "No"] else 0.0
        
        return answer, reward
    
    def _handle_guess(self, guess: str) -> EnvState:
        """Handle a final guess."""
        guess_normalized = guess.lower().strip()
        
        self.done = True
        
        if guess_normalized == self.secret:
            # Correct!
            bonus = max(0, (self.max_questions - self.question_count) * 0.5)  # Bonus for fewer questions
            reward = 10.0 + bonus
            observation = f"🎉 Correct! The word was **{self.secret}**!\n\nYou solved it in {self.question_count} questions. (Bonus: +{bonus:.1f})"
        else:
            reward = -1.0
            observation = f"❌ Wrong! You guessed '{guess}', but the word was **{self.secret}**."
        
        self.total_reward += reward
        return self._get_state(observation, reward, True)
    
    def _get_state(self, observation: str, reward: float, done: bool) -> EnvState:
        """Create an EnvState object."""
        return EnvState(
            observation=observation,
            reward=reward,
            done=done,
            info={
                "category": self.category,
                "secret": self.secret if done else "[HIDDEN]",
                "questions_asked": self.question_count,
                "questions_remaining": max(0, self.max_questions - self.question_count),
            },
            history=self.history.copy(),
            step_count=self.question_count,
            total_reward=self.total_reward,
        )


class WordGuessingEnv:
    """
    Simple word guessing game (like Wordle but text-based).
    
    The agent guesses a secret word, receiving feedback on each guess.
    Simpler than 20 Questions for quick testing.
    """
    
    WORDS = ["robot", "apple", "ocean", "piano", "cloud", "bread", "dream", "stone"]
    
    def __init__(self, max_guesses: int = 6, seed: Optional[int] = None):
        self.max_guesses = max_guesses
        self.rng = random.Random(seed)
        self.secret: Optional[str] = None
        self.guesses: List[str] = []
        self.done = False
        self.total_reward = 0.0
        
    def reset(self) -> EnvState:
        self.secret = self.rng.choice(self.WORDS)
        self.guesses = []
        self.done = False
        self.total_reward = 0.0
        
        return EnvState(
            observation=f"Guess the {len(self.secret)}-letter word! You have {self.max_guesses} guesses.",
            reward=0.0,
            done=False,
            info={"word_length": len(self.secret)},
            history=[],
            step_count=0,
            total_reward=0.0,
        )
    
    def step(self, guess: str) -> EnvState:
        if self.done:
            return EnvState("Game over!", 0.0, True, {}, self.guesses, len(self.guesses), self.total_reward)
        
        guess = guess.lower().strip()
        self.guesses.append(guess)
        
        if guess == self.secret:
            reward = 10.0 + (self.max_guesses - len(self.guesses))
            self.done = True
            obs = f"🎉 Correct! The word was {self.secret}!"
        elif len(self.guesses) >= self.max_guesses:
            reward = -5.0
            self.done = True
            obs = f"❌ Out of guesses! The word was {self.secret}."
        else:
            # Give feedback
            feedback = self._get_feedback(guess)
            reward = 0.1 * feedback.count("✓")
            remaining = self.max_guesses - len(self.guesses)
            obs = f"Guess: {guess} → {feedback}\n[{remaining} guesses remaining]"
        
        self.total_reward += reward
        return EnvState(obs, reward, self.done, {"secret": self.secret if self.done else None}, 
                       list(zip(self.guesses, [""] * len(self.guesses))), len(self.guesses), self.total_reward)
    
    def _get_feedback(self, guess: str) -> str:
        """Generate Wordle-style feedback."""
        feedback = []
        for i, c in enumerate(guess):
            if i < len(self.secret) and c == self.secret[i]:
                feedback.append("✓")  # Correct position
            elif c in self.secret:
                feedback.append("~")  # Wrong position
            else:
                feedback.append("✗")  # Not in word
        return "".join(feedback)


if __name__ == "__main__":
    # Quick test
    env = TwentyQuestionsEnv(max_questions=10, seed=42)
    state = env.reset()
    print(state.observation)
    print(f"\n[Debug] Secret word: {env.secret}\n")
    
    # Simulate a few questions
    questions = [
        "ask: Is it alive?",
        "ask: Is it big?",
        "ask: Can it fly?",
        "ask: Does it live in water?",
        "guess: dolphin",
    ]
    
    for q in questions:
        state = env.step(q)
        print(f"> {q}")
        print(state.observation)
        print(f"[Reward: {state.reward:+.2f}, Total: {state.total_reward:.2f}]")
        print()
        if state.done:
            break
