from hmac import trans_36
import random
import os
import time
import numpy as np

class Discrete:
    def __init__(self, num_actions) -> None:
        self.n = num_actions

    def sample(self):
        return random.randint(0, self.n-1)

space = Discrete(4)

class Environment:

    def __init__(self, grid_size=5, goal_reward=1, max_step=200, *args, **kwargs) -> None:
        self.grid_size = grid_size
        self.action_space = Discrete(4)
        self.observation_space = Discrete(grid_size*grid_size)
        self.seeker, self.goal = (0, 0), (grid_size-1, grid_size-1)
        self.info = {'seeker': self.seeker, 'goal': self.goal}
        self.goal_reward = goal_reward
        self.timestep = 0
        self.max_step = max_step

    def reset(self):
        self.seeker = (0, 0) # row, col
        self.goal = (self.grid_size-1, self.grid_size-1)
        self.timestep = 0
        return self.get_observation()
    
    def get_observation(self):
        return self.grid_size * self.seeker[0] + self.seeker[1]
    
    def get_reward(self):
        return self.goal_reward if self.seeker == self.goal else -1
    
    def is_done(self):
        if self.timestep == self.max_step:
            return True
        return self.seeker == self.goal

    def check_pos(self, seeker):
        is_out = False
        if seeker[0] < 0 or seeker[0] > self.grid_size - 1 or \
            seeker[1] < 0 or seeker[1] > self.grid_size - 1: 
            is_out = True
        return is_out

    def step(self, action):
        self.timestep += 1
        reward = 0
        is_out = False

        if action == 0: # move left
            self.seeker = (self.seeker[0], self.seeker[1] - 1)
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0], self.seeker[1] + 1)

        elif action == 1: # move right
            self.seeker = (self.seeker[0], self.seeker[1] + 1)
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0], self.seeker[1] - 1)

        elif action == 2: # move up
            self.seeker = (self.seeker[0] - 1, self.seeker[1])
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0] + 1, self.seeker[1])
                
        elif action == 3: # move down
            self.seeker = (self.seeker[0] + 1, self.seeker[1])
            is_out =  self.check_pos(self.seeker)
            if is_out:
                self.seeker = (self.seeker[0] - 1, self.seeker[1])
        else:
            raise ValueError("Invalid action")

        if is_out:
            reward = -10
        else:
            reward = self.get_reward()

        return self.get_observation(), reward, self.is_done(), self.info

    def render(self, *args, **kwaargs):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid_row = ['| ' for _ in range(self.grid_size)] 
        grid = [grid_row + ["|\n"] for _ in range(self.grid_size)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.seeker[0]][self.seeker[1]] = '|A'
        print(''.join([''.join(grid_row) for grid_row in grid]))

    
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


def evaluate_policy(env, policy, npy_path, num_episodes=10):
    policy.load(npy_path)
    simulation = Simulation(env)
    steps = 0
    total_reward_lst = []
    avg_score = 0
    for e in range(num_episodes):
        experiences = simulation.rollout(e, policy, render=True, explore=False, epsilon=0.1, render_time=0.1)
        total_reward = 0
        for transition in experiences:
            total_reward += transition[2]
        total_reward_lst.append(total_reward)
        steps += len(experiences)
    
    avg_score = sum(total_reward_lst) / len(total_reward_lst)
    return steps / num_episodes, avg_score, total_reward_lst



env = Environment(grid_size=5, goal_reward=10, max_step=200)
untrained_policy = Policy(env)
sim = Simulation(env)
exp = sim.rollout(1, untrained_policy, render=True, epsilon=1.0)

trained_policy = train_policy(env, num_episodes=10000)
policy = Policy(env)
avg_steps, avg_score, total_reward_lst = evaluate_policy(env, policy, "policy8000.npy")
print(f"avg_steps: {avg_steps}, avg_score: {avg_score}")
print(np.round(policy.state_action_table, 2))


