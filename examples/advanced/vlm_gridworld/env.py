"""
GridWorld Environment with Visual Rendering.

A simple navigation environment where an agent moves on a grid
towards a goal position. The environment renders observations as
RGB images suitable for VLM processing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class GridState:
    """State of the GridWorld environment."""
    agent_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    size: int
    step_count: int
    done: bool
    reward: float
    total_reward: float


class GridWorld:
    """
    Minimal GridWorld environment with image rendering.
    
    The agent starts at (0, 0) and must navigate to the goal.
    Actions: "up", "down", "left", "right"
    
    Rewards:
        +10.0 for reaching the goal
        -0.01 step penalty
        -0.1 for hitting walls (invalid moves)
    """
    
    # Colors for rendering (RGB)
    COLOR_BG = (40, 40, 40)         # Dark gray background
    COLOR_GRID = (80, 80, 80)       # Grid lines
    COLOR_AGENT = (66, 135, 245)    # Blue agent
    COLOR_GOAL = (76, 175, 80)      # Green goal
    COLOR_PATH = (100, 100, 150)    # Faded path trail
    
    ACTIONS = ["up", "down", "left", "right"]
    ACTION_DELTAS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    
    def __init__(
        self,
        size: int = 8,
        goal: Optional[Tuple[int, int]] = None,
        cell_size: int = 64,
        max_steps: int = 100,
    ):
        """
        Initialize the GridWorld.
        
        Args:
            size: Grid size (size x size)
            goal: Goal position (row, col). Defaults to (size-1, size-1)
            cell_size: Pixel size of each cell for rendering
            max_steps: Maximum steps before episode terminates
        """
        self.size = size
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.cell_size = cell_size
        self.max_steps = max_steps
        
        # State
        self.agent_pos = [0, 0]
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.path_history: list[Tuple[int, int]] = []
        
    def reset(self, start: Optional[Tuple[int, int]] = None) -> GridState:
        """Reset the environment to initial state."""
        self.agent_pos = list(start) if start else [0, 0]
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.path_history = [tuple(self.agent_pos)]
        return self._get_state(reward=0.0)
    
    def step(self, action: str) -> GridState:
        """
        Execute an action and return the new state.
        
        Args:
            action: One of "up", "down", "left", "right"
            
        Returns:
            GridState with updated position, reward, done flag
        """
        if self.done:
            return self._get_state(reward=0.0)
        
        # Parse action
        action = action.lower().strip()
        if action not in self.ACTION_DELTAS:
            # Invalid action, small penalty
            self.step_count += 1
            reward = -0.1
            self.total_reward += reward
            return self._get_state(reward=reward)
        
        # Calculate new position
        delta = self.ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + delta[0]
        new_col = self.agent_pos[1] + delta[1]
        
        # Check bounds
        if 0 <= new_row < self.size and 0 <= new_col < self.size:
            self.agent_pos = [new_row, new_col]
            self.path_history.append(tuple(self.agent_pos))
            reward = -0.01  # Step penalty
        else:
            # Hit wall
            reward = -0.1
        
        self.step_count += 1
        
        # Check goal
        if tuple(self.agent_pos) == self.goal:
            reward = 10.0
            self.done = True
        
        # Check max steps
        if self.step_count >= self.max_steps:
            self.done = True
        
        self.total_reward += reward
        return self._get_state(reward=reward)
    
    def _get_state(self, reward: float) -> GridState:
        """Create current state snapshot."""
        return GridState(
            agent_pos=tuple(self.agent_pos),
            goal_pos=self.goal,
            size=self.size,
            step_count=self.step_count,
            done=self.done,
            reward=reward,
            total_reward=self.total_reward,
        )
    
    def render(self) -> np.ndarray:
        """
        Render the current state as an RGB image.
        
        Returns:
            numpy array of shape (H, W, 3) with RGB values
        """
        if not HAS_PIL:
            # Fallback: simple numpy rendering
            return self._render_numpy()
        
        return self._render_pil()
    
    def _render_pil(self) -> np.ndarray:
        """Render using PIL for nicer graphics."""
        img_size = self.size * self.cell_size
        img = Image.new("RGB", (img_size, img_size), self.COLOR_BG)
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines
        for i in range(self.size + 1):
            # Horizontal
            y = i * self.cell_size
            draw.line([(0, y), (img_size, y)], fill=self.COLOR_GRID, width=1)
            # Vertical
            x = i * self.cell_size
            draw.line([(x, 0), (x, img_size)], fill=self.COLOR_GRID, width=1)
        
        # Draw path history (faded trail)
        for pos in self.path_history[:-1]:  # Exclude current position
            self._draw_cell(draw, pos, self.COLOR_PATH, margin=20)
        
        # Draw goal (green square)
        self._draw_cell(draw, self.goal, self.COLOR_GOAL, margin=8)
        
        # Draw agent (blue circle)
        row, col = self.agent_pos
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=self.COLOR_AGENT,
        )
        
        # Add text labels
        try:
            font = ImageFont.load_default()
            draw.text((5, 5), f"Step: {self.step_count}", fill=(255, 255, 255), font=font)
            draw.text((5, 20), f"Reward: {self.total_reward:.2f}", fill=(255, 255, 255), font=font)
        except Exception:
            pass  # Skip text if font not available
        
        return np.array(img)
    
    def _draw_cell(self, draw: "ImageDraw.Draw", pos: Tuple[int, int], color: Tuple[int, int, int], margin: int = 4):
        """Draw a filled rectangle in a grid cell."""
        row, col = pos
        x1 = col * self.cell_size + margin
        y1 = row * self.cell_size + margin
        x2 = (col + 1) * self.cell_size - margin
        y2 = (row + 1) * self.cell_size - margin
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    def _render_numpy(self) -> np.ndarray:
        """Simple numpy-based rendering (fallback)."""
        img_size = self.size * self.cell_size
        img = np.full((img_size, img_size, 3), self.COLOR_BG, dtype=np.uint8)
        
        # Draw goal
        gr, gc = self.goal
        y1, y2 = gr * self.cell_size, (gr + 1) * self.cell_size
        x1, x2 = gc * self.cell_size, (gc + 1) * self.cell_size
        img[y1:y2, x1:x2] = self.COLOR_GOAL
        
        # Draw agent
        ar, ac = self.agent_pos
        y1, y2 = ar * self.cell_size, (ar + 1) * self.cell_size
        x1, x2 = ac * self.cell_size, (ac + 1) * self.cell_size
        img[y1:y2, x1:x2] = self.COLOR_AGENT
        
        return img
    
    def get_optimal_action(self) -> str:
        """Get the optimal action towards the goal (for mock agent)."""
        row, col = self.agent_pos
        goal_row, goal_col = self.goal
        
        # Simple greedy: move towards goal
        if row < goal_row:
            return "down"
        elif row > goal_row:
            return "up"
        elif col < goal_col:
            return "right"
        elif col > goal_col:
            return "left"
        else:
            return "right"  # At goal, arbitrary


if __name__ == "__main__":
    # Quick test
    env = GridWorld(size=5)
    state = env.reset()
    print(f"Initial: {state}")
    
    for action in ["right", "right", "down", "down", "right", "right", "down", "down"]:
        state = env.step(action)
        print(f"Action: {action} -> Pos: {state.agent_pos}, Reward: {state.reward:.2f}, Done: {state.done}")
    
    # Render
    img = env.render()
    print(f"Rendered image shape: {img.shape}")
