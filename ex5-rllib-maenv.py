import time
from maze_gym_multi_env import MultiAgentMaze
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print


env = MultiAgentMaze()

# config = {
#     "multiagent": {
#         "policies": {  # <1>
#             "policy_1": (None, env.observation_space, env.action_space, {"gamma": 0.80}),
#             "policy_2": (None, env.observation_space, env.action_space, {"gamma": 0.95}),
#         },
#         "policy_mapping_fn": lambda agent_id: f"policy_{agent_id}",}# <2>
#     }

# simple_trainer = DQNTrainer(env=MultiAgentMaze, config=config)
# simple_trainer.train()
# config = simple_trainer.get_config()
# print(pretty_print(config))


trainer = DQNTrainer(env=MultiAgentMaze, config={
    "multiagent": {
        "policies": {  # <1>
            "policy_1": (None, env.observation_space, env.action_space, {"gamma": 0.80}),
            "policy_2": (None, env.observation_space, env.action_space, {"gamma": 0.95}),
        },
        "policy_mapping_fn": lambda agent_id: f"policy_{agent_id}",  # <2>
    },
})

for i in range(10):
    result = trainer.train()

print(pretty_print(result))