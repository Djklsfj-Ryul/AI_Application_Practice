import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%7.3f" % x))

# https://gymnasium.farama.org/environments/atari/breakout/
env = gym.make("ALE/Breakout-v5", render_mode="human")

def observation_and_action_space_info():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print("[observation_space]")
    #[0,255,(210,160,3),uint8] -> [최소, 최대, 3채널 2D 디멘션 어레이, 타입]
    print("env.observation_space: ", env.observation_space)
    print("env.observation_space.high:", env.observation_space.high)
    print("env.observation_space.low:", env.observation_space.low)
    print("env.observation_space.shape:", env.observation_space.shape)
    for i in range(1):
        print(env.observation_space.sample())
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print("[action_space]")
    print(env.action_space)
    print("env.action_space.n:", env.action_space.n)
    print("env.action_space.shape:", env.action_space.shape)
    for i in range(10):
        print(env.action_space.sample(), end=" ")


if __name__ == "__main__":
    observation_and_action_space_info()
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

            print(
                "Step: {0}, Obs.: {1}, Action: {2}, Next Obs.: {3}, Reward: {4:5.2f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
                    step, observation, action, next_observation, reward, terminated, truncated, info
                ))

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print(f"Episode: {episode + 1}, Episode Reward: {episode_reward}")
