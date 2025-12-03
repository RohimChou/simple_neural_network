"""
Reinforcement Learning Training Script
Train a Q-learning agent to navigate a grid world
"""
import numpy as np
import time
from grid_world import GridWorld
from q_learning_agent import QLearningAgent


def train_agent(env: GridWorld,
                agent: QLearningAgent,
                n_episodes: int = 1000, # 玩幾局
                max_steps: int = 100,
                verbose: bool = True,
                show_live: bool = False):
    """
    Train the agent using Q-learning

    Args:
        env: The environment
        agent: The Q-learning agent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        show_live: Whether to show live training visualization (slower)
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print("\n🚀 Starting Q-Learning Training...")
    print(f"Episodes: {n_episodes}, Max steps per episode: {max_steps}")
    print(f"Learning rate: {agent.lr}, Discount factor: {agent.gamma}")
    print(f"Initial epsilon: {agent.epsilon}\n")

    if show_live:
        print("⚠️  Live visualization enabled - training will be slower\n")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        # Show live training for some episodes
        show_this_episode = show_live and (episode % 50 == 0 or episode < 5)

        if show_this_episode:
            print(f"\n📺 Episode {episode + 1} (Live)")
            env.render(clear=True)
            time.sleep(0.1)

        while not done and steps < max_steps:
            # Choose action
            action = agent.get_action(state, training=True)

            # Take action
            next_state, reward, done = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            if show_this_episode:
                env.render(clear=True)
                print(f"Episode: {episode + 1} | Step: {steps + 1} | Action: {env.actions[action]} | Reward: {reward:+4.0f} | Epsilon: {agent.epsilon:.3f}")
                time.sleep(0.05)

            state = next_state
            total_reward += reward
            steps += 1

        # Decay exploration rate
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if done and total_reward > 0:
            success_count += 1

        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            success_rate = success_count / 100

            if show_live:
                print("\n" + "="*50)

            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}\n")
            success_count = 0

    print("✅ Training Complete!\n")
    return episode_rewards, episode_lengths


def demonstrate_agent(env: GridWorld, agent: QLearningAgent, n_demos: int = 3):
    """
    Demonstrate the trained agent with in-place grid updates
    """
    print("\n🎯 Demonstrating Trained Agent...")

    for demo in range(n_demos):
        print(f"\n--- Demo {demo + 1} ---")
        state = env.reset()

        total_reward = 0
        steps = 0
        done = False
        max_steps = 20

        # Initial render
        env.render(clear=True)
        print(f"Step: {steps} | Total Reward: {total_reward} | Status: Starting...")
        time.sleep(1)

        while not done and steps < max_steps:
            # Choose best action (no exploration)
            action = agent.get_action(state, training=False)
            action_name = env.actions[action]

            # Take action
            next_state, reward, done = env.step(action)

            steps += 1
            total_reward += reward

            # Clear and re-render
            env.render(clear=True)

            # Show status
            status = "🎉 Reached Goal!" if done else "Moving..."
            print(f"Step: {steps} | Action: {action_name} | Reward: {reward:+4.0f} | Total: {total_reward:+5.0f} | Status: {status}")

            state = next_state
            time.sleep(0.3)  # Adjust speed here

        print()
        if done and total_reward > 0:
            print(f"✅ Success! Total reward: {total_reward} in {steps} steps")
        else:
            print(f"❌ Failed to reach goal. Total reward: {total_reward}")

        time.sleep(1)  # Pause between demos


def show_learned_policy(env: GridWorld, agent: QLearningAgent):
    """
    Display the learned policy
    """
    print("\n📋 Learned Policy (Best action for each state):")
    print("="*50)

    policy = agent.get_policy()

    for row in range(env.size):
        line = ""
        for col in range(env.size):
            pos = (row, col)
            if pos == env.goal_pos:
                line += " G "
            elif pos in env.obstacles:
                line += " X "
            else:
                action_idx = policy[row, col]
                line += f" {env.actions[action_idx]} "
            line += " "
        print(line)

    print("="*50)
    print()


def show_value_function(env: GridWorld, agent: QLearningAgent):
    """
    Display the learned value function
    """
    print("\n💎 State Values (Max Q-value for each state):")
    print("="*50)

    for row in range(env.size):
        line = ""
        for col in range(env.size):
            pos = (row, col)
            if pos in env.obstacles:
                line += "  X  "
            else:
                value = agent.get_state_value(pos)
                line += f"{value:5.1f}"
            line += " "
        print(line)

    print("="*50)
    print()


def main():
    """Main function to run the RL project"""
    print("\n" + "="*60)
    print("  REINFORCEMENT LEARNING: Q-LEARNING GRID WORLD")
    print("="*60)

    # Create environment
    env = GridWorld(size=5)

    # Create agent
    agent = QLearningAgent(
        state_size=env.size,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Show initial environment
    print("\n🗺️  Initial Environment:")
    env.render()

    # Train the agent
    rewards, lengths = train_agent(
        env,
        agent,
        n_episodes=500,
        max_steps=50,
        verbose=True
    )

    # Show results
    show_learned_policy(env, agent)
    show_value_function(env, agent)

    # Demonstrate the trained agent
    demonstrate_agent(env, agent, n_demos=2)

    # print Q-table
    print("\n🔢 Learned Q-Table:")
    print("="*50)
    for row in range(env.size):
        for col in range(env.size):
            pos = (row, col)
            if pos in env.obstacles:
                continue
            q_values = agent.q_table[row, col]
            print(f"State ({row},{col}): ", end="")
            for action_idx, q_val in enumerate(q_values):
                print(f"{env.actions[action_idx]}: {q_val:.2f}  ", end="")
            print()
    print("="*50)

    # Print summary statistics
    print("\n📊 Training Summary:")
    print("="*50)
    print(f"Total episodes: {len(rewards)}")
    print(f"Average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Average episode length (last 100): {np.mean(lengths[-100:]):.2f}")
    print("="*50)
    print("\n✨ Project Complete!\n")


if __name__ == "__main__":
    main()