import time
from maze_gym_multi_env import MultiAgentMaze
env = MultiAgentMaze()

for e in range(10):
    total_rewards = [0, 0]
    while True:
        action_dict = {0: env.action_space.sample(), 1: env.action_space.sample()}
        obs, rew, done, info = env.step(action_dict)
        total_rewards[0] += rew[0]
        total_rewards[1] += rew[1]
        time.sleep(0.1)
        env.render()
        print(f"current episode: {e}")
        for i in range(2):
            print(f"agent {i} pos. x: {env.agents[i][0]:>2d}, y: {env.agents[i][1]:>2d}")
        print(f"goal pos.  x: {env.goal[0]:>2d}, y: {env.goal[1]:>2d}")
        print(f"total rewards: {total_rewards}")
        if any(done.values()):
            print("done")
            time.sleep(0.2)
            break

