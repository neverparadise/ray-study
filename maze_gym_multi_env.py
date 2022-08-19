from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Discrete
import os 

env_config = {"grid_size": 7,
             "goal_reward": 1,
             "out_reward": -10,
             "max_step": 200,
             "step_reward":-1
             }


class MultiAgentMaze(MultiAgentEnv):

    def __init__(self):
        self.grid_size = env_config["grid_size"]
        self.agents = {0: (self.grid_size-1, 0), 1: (0, self.grid_size-1)}
        self.goal = (int((self.grid_size-1) / 2), int((self.grid_size-1) / 2))
        self.goal_reward = env_config["goal_reward"]
        self.out_reward = env_config["out_reward"]
        self.max_step = env_config["max_step"]
        self.step_reward = env_config["step_reward"]
        self.timestep = 0

        self.info = {0: {'obs': self.agents[0]}, 
                    1: {'obs': self.agents[1]}}
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.grid_size * self.grid_size)

    def reset(self):
        self.agents = {0: (self.grid_size-1, 0), 1: (0, self.grid_size-1)}
        self.timestep = 0
        return {0: self.get_observation(1), 1: self.get_observation(2)}

    def get_observations(self, agent_id):
        pos = self.agents[agent_id]
        return self.grid_size * pos[0] + pos[1]

    def get_reward(self, agent_id):
        return self.goal_reward if self.agents[agent_id] == self.goal else self.step_reward
    
    def is_done(self, agent_id):
        if self.timestep == self.max_step:
            return True
        return self.agents[agent_id] == self.goal
    
    def check_pos(self, pos):
        is_out = False
        if pos[0] < 0 or pos[0] > self.grid_size - 1 or \
            pos[1] < 0 or pos[1] > self.grid_size - 1: 
            is_out = True
        return is_out

    def step(self, action):
        agent_ids = action.keys()
        self.timestep += 1

        is_out_lst = [False for i in range(len(action))]

        for agent_id in agent_ids:
            is_out = False
            pos = self.agents[agent_id]
            if action[agent_id] == 0: # move left
                pos = (pos[0], pos[1] - 1)
                is_out =  self.check_pos(pos)
                if is_out:
                    pos = (pos[0], pos[1] + 1)

            elif action[agent_id] == 1: # move right
                pos = (pos[0], pos[1] + 1)
                is_out =  self.check_pos(pos)
                if is_out:
                    pos = (pos[0], pos[1] - 1)

            elif action[agent_id] == 2: # move up
                pos = (pos[0] - 1, pos[1])
                is_out =  self.check_pos(pos)
                if is_out:
                    pos = (pos[0] + 1, pos[1])
                    
            elif action[agent_id] == 3: # move down
                pos = (pos[0] + 1, pos[1])
                is_out =  self.check_pos(pos)
                if is_out:
                    pos = (pos[0] - 1, pos[1])
            else:
                raise ValueError("Invalid action")
            self.agents[agent_id] = pos
            is_out_lst[agent_id] = is_out

        observations = dict()
        rewards = dict()
        dones = dict()

        for i in agent_ids:
            observations[i] = self.get_observations(i)
            rewards[i] = self.out_reward if is_out_lst[i] else self.get_reward(i)
            dones[i] = self.is_done(i)
        
        dones["__all__"] = all(dones.values())

        return observations, rewards, dones, self.info

    def render(self, *args, **kwaargs):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid_row = ['| ' for _ in range(self.grid_size)] 
        grid = [grid_row + ["|\n"] for _ in range(self.grid_size)]
        grid[self.goal[0]][self.goal[1]] = '|G'
        grid[self.agents[0][0]][self.agents[0][1]] = '|1'
        grid[self.agents[1][0]][self.agents[1][1]] = '|2'
        print(''.join([''.join(grid_row) for grid_row in grid]))

