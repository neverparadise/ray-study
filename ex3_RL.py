from hmac import trans_36
import random
import os
import time
import numpy as np
#from maze_gym_env import Environment
from maze_gym_env_hard import Environment


class Policy:

    def __init__(self, env):
        # 얕은 복사 깊은 복사 주의
        self.state_action_table = [
            [0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)
            ]
        self.action_space = env.action_space
    
    def get_action(self, state, explore=True, epsilon=0.1):
        if explore and random.uniform(0,1) < epsilon:
            return self.action_space.sample()
        return np.argmax(self.state_action_table[state])

    def save(self, num):
        name = f"policy{num}.npy"
        np.save(name, self.state_action_table)
    
    def load(self, npy_path):
         self.state_action_table = np.load(npy_path)


class Simulation(object):

    def __init__(self, env):
        self.env = env

    def rollout(self, episode, policy, render=False, explore=True, epsilon=0.1, render_time=0.0001):
        experiences = []
        state = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy.get_action(state, explore, epsilon)
            next_state, reward, done, info = self.env.step(action)
            experiences.append([state, action, reward, next_state, done])
            total_reward += reward
            state = next_state
            if render:
                time.sleep(render_time)
                self.env.render()
                print(f"current episode: {episode}")
                print(f"agent pos. x: {self.env.seeker[0]:>2d}, y: {self.env.seeker[1]:>2d}")
                print(f"goal pos.  x: {self.env.goal[0]:>2d}, y: {self.env.goal[1]:>2d}")
                print(f"total reward: {total_reward:>2d}")
        return experiences


def update_policy(policy, trajectory, weight=0.1, discount_factor=0.9):
    random.shuffle(trajectory)
    for state, action, reward, next_state, done in trajectory:

        next_max = np.max(policy.state_action_table[next_state])
        # action_value = policy.state_action_table[state][action]
        # td_target = (reward + discount_factor * next_max * (1 - done)) 
        # td_error = td_target - action_value
        # new_action_value = (1 - weight) * action_value + weight * td_error
        # policy.state_action_table[state][action] = new_action_value
        value = policy.state_action_table[state][action]
        new_value = (1 - weight) * value + weight * (reward + discount_factor * next_max)
        policy.state_action_table[state][action] = new_value


def train_policy(env, num_episodes=10000, weight=0.1, discount_factor=0.9, epsilon=0.1):
    policy = Policy(env)
    sim = Simulation(env)
    for e in range(num_episodes):
        trajectory = sim.rollout(e, policy, render=False, explore=True, epsilon=epsilon)
        update_policy(policy, trajectory, weight, discount_factor)
        if e % 2000 == 0 and e > 0 :
            policy.save(e)

    return policy


def evaluate_policy(env, policy, npy_path, num_episodes=10, render=False):
    policy.load(npy_path)
    simulation = Simulation(env)
    steps = 0
    total_reward_lst = []
    avg_score = 0
    for e in range(num_episodes):
        experiences = simulation.rollout(e, policy, render=render, explore=False, epsilon=0.1, render_time=0.1)
        total_reward = 0
        for transition in experiences:
            total_reward += transition[2]
        total_reward_lst.append(total_reward)
        steps += len(experiences)
    
    avg_score = sum(total_reward_lst) / len(total_reward_lst)
    return steps / num_episodes, avg_score, total_reward_lst


if __name__ == '__main__':
    env = Environment(grid_size=5, goal_reward=10, max_step=200)
    untrained_policy = Policy(env)
    sim = Simulation(env)
    exp = sim.rollout(1, untrained_policy, render=True, epsilon=1.0)

    trained_policy = train_policy(env, num_episodes=10000)
    policy = Policy(env)
    avg_steps, avg_score, total_reward_lst = evaluate_policy(env, policy, "policy90000.npy", render=True)
    print(f"avg_steps: {avg_steps}, avg_score: {avg_score}")
    print(np.round(policy.state_action_table, 2))


