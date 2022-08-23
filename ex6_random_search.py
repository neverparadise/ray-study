import random
from maze_gym_env import Environment
from ex3_RL import train_policy, evaluate_policy
import ray

search_space = []
for i in range(10):
    random_choice = {
        'weight': random.uniform(0, 1),
        'discount_factor': random.uniform(0, 1) 
        }
    search_space.append(random_choice)

@ray.remote
def objective(config):
    env = Environment()
    policy = train_policy(env, weight=config["weight"],
                         discount_factor=config["discount_factor"])
    score, avg_score, _ = evaluate_policy(env, policy, npy_path='./policy8000.npy')
    return [score, config]

result_refs = [objective.remote(config) for config in search_space]
results = ray.get(result_refs)

results.sort(key=lambda x: x[0])
for res in results:
    print(res)
