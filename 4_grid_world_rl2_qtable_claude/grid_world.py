"""
Grid World Environment for Reinforcement Learning
A simple grid where the agent learns to reach a goal while avoiding obstacles
"""
import numpy as np
from typing import Tuple, List


class GridWorld:
    """A grid world environment for RL agents"""

    def __init__(self, size: int = 5):
        self.size = size
        self.grid = np.zeros((size, size))

        # Set goal position (bottom right)
        self.goal_pos = (size - 1, size - 1)

        # Set obstacles
        self.obstacles = [(0,3), (1, 1), (2, 2), (3, 1)]

        # Starting position (top left)
        self.start_pos = (0, 0)
        self.agent_pos = self.start_pos

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = ['↑', '→', '↓', '←']
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """Reset the environment to starting position"""
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take an action and return (next_state, reward, done)
        """
        row, col = self.agent_pos

        # Calculate new position based on action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            row = min(self.size - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)

        new_pos = (row, col)

        # Check if new position is valid
        if new_pos in self.obstacles:
            # Hit obstacle - stay in place, negative reward
            reward = -10
            done = False
        elif new_pos == self.goal_pos:
            # Reached goal
            self.agent_pos = new_pos
            reward = 100
            done = True
        else:
            # Normal move
            self.agent_pos = new_pos
            reward = -1  # Small penalty to encourage faster completion
            done = False

        return self.agent_pos, reward, done

    def render(self, clear=False):
        """Display the grid world"""
        if clear:
            # Clear screen and move cursor to top
            print("\033[2J\033[H", end="")

        print("\n" + "="*50)
        for row in range(self.size):
            line = ""
            for col in range(self.size):
                pos = (row, col)
                if pos == self.agent_pos:
                    line += " A "
                elif pos == self.goal_pos:
                    line += " G "
                elif pos in self.obstacles:
                    line += " X "
                else:
                    line += " · "
                line += " "
            print(line)
        print("="*50)
        print("A=Agent, G=Goal, X=Obstacle\n")