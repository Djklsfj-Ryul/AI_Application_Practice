# Create environment
import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")
import numpy as np
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(edgeitems=10, linewidth=100000, formatter=dict(float=lambda x: "%4.2f" % x))

# https://gymnasium.farama.org/environments/box2d/lunar_lander/
env = gym.make("LunarLander-v2", render_mode="human")

# Num of training episodes = 3
for episode in range(3):
    # Reset
    # observation : State (단 t = 0)
    observation, info = env.reset(seed=123, options={})

    # S_t 가 완료된 상태인지를 확인하는 변수
    done = False
    episode_reward = 0.0
    step = 0
    while not done:
        # Agent policy that uses the observation and info
        action = env.action_space.sample()
        # State 다음 time step
        # terminated : 파괴 상태 (바로 종료)
        # truncated  : 모든 스텝 완료
        next_observation, reward, terminated, truncated, info = env.step(action)
        step += 1

        print("Step: {0}, Obs.: {1}, Action: {2}, Next Obs.: {3}, Reward: {4:5.2f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
            step, observation, action, next_observation, reward, terminated, truncated, info
        ))

        episode_reward += reward
        observation = next_observation
        done = terminated or truncated

    print(f"Episode: {episode + 1}, Episode Reward: {episode_reward}")

env.close()