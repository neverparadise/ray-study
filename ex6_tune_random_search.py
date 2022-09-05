import random
from maze_gym_env import Environment
from ex3_RL import train_policy, evaluate_policy
import ray
from ray import tune

search_space = {
        'weight': tune.uniform(0, 1),
        'discount_factor': tune.uniform(0, 1) 
        }

print(search_space)

def tune_objective(config):
    env = Environment()
    policy = train_policy(env, weight=config["weight"],
                         discount_factor=config["discount_factor"])
    score, avg_score, _ = evaluate_policy(env, policy, npy_path='./policy8000.npy')
    return score

analysis = tune.run(tune_objective, config=search_space)
print(analysis.get_best_config(metric='score', mode='min'))
