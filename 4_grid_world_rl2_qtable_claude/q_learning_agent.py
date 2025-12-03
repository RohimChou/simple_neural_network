import numpy as np
from typing import Tuple


class QLearningAgent:
    """Q-Learning agent that learns optimal policy through experience"""

    def __init__(self,
                 state_size: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-learning agent

        Args:
            state_size: Size of the grid (assumes square grid)
            n_actions: Number of possible actions
            learning_rate: Alpha - how much to update Q-values
            discount_factor: Gamma - importance of future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Rate at which epsilon decreases
            epsilon_min: Minimum epsilon value
        """
        self.state_size = state_size
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        # Shape: (grid_size, grid_size, n_actions)
        self.q_table = np.zeros((state_size, state_size, n_actions))

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy

        Args:
            state: Current state (row, col)
            training: Whether we're in training mode (uses epsilon-greedy)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            # 假設`epsilon = 0.1`（10 %）, np.random.random() 介於 [0.0, 1.0) 之間
            # 抽到 0.05 < 0.1  ✓ → 隨機探索
            # 抽到 0.23 < 0.1  ✗ → 用最佳動作
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action
            row, col = state
            return np.argmax(self.q_table[row, col])

    def update(self,
               state: Tuple[int, int],
               action: int,
               reward: float,
               next_state: Tuple[int, int],
               done: bool):
        """
        Update Q-table using Q-learning update rule:
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

        α: learning rate
        r: immediate reward
        γ: discount factor：為什麼打折? 立即獎勵比較可靠、比較重要；未來獎勵還不確定能不能拿到
              0.0只在乎現在短視近利，只看眼前
              0.5未來打 5 折比較急
              0.9未來打 9 折常用值，平衡
              0.99幾乎不打折很有遠見，願意等
        max(Q(s',a'))： 下一步最好的分數 (看下一步所有可能動作裡，哪個分數最高)

        舉例: 從倒數第二格走到終點格
        假設 Q((4,3), 右) = 50, learning rate α = 0.1, discount factor γ = 0.95
        獎勵 reward = 100 (到達終點)
        計算:
        new Q((4,3), 右) = 50 + 0.1[100 + 0.95*0 - 50]
                         = 50 + 0.1[50]
                         = 50 + 5
                         = 55
        這樣 Q-value 從 50 提升到 55，表示從 (4,3) 往右走更有價值了
        新 Q[4, 3] 可能長這樣 → [30, 55, 30, 30] （上, 右, 下, 左）
        """
        row, col = state
        next_row, next_col = next_state

        # Current Q-value
        current_q = self.q_table[row, col, action]

        # Maximum Q-value for next state
        if done:
            # If episode is done, there's no next state value
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_row, next_col])

        # Calculate new Q-value using Bellman equation
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

        # Update Q-table
        self.q_table[row, col, action] = new_q

    def decay_epsilon(self):
        """Reduce epsilon for less exploration over time"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Return the best action for each state"""
        return np.argmax(self.q_table, axis=2)

    def get_state_value(self, state: Tuple[int, int]) -> float:
        """Get the maximum Q-value for a state"""
        row, col = state
        return np.max(self.q_table[row, col])