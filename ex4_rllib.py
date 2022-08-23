from ray.tune.logger import pretty_print
from maze_gym_env import GymEnvironment
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer


training_config = {"num_workers": 0, 
                "num_envs_per_worker": 5,
                "train_batch_size": 64,
                "framework": "tf"}

trainer = DQNTrainer(env=GymEnvironment, config=training_config)
config = trainer.get_config()
print(pretty_print(config))

# for i in range(10):
#     result = trainer.train()

# print(pretty_print(result))

# tag::rllib_simple_save[]
checkpoint = trainer.save()
print(checkpoint)
evaluation = trainer.evaluate(checkpoint)

policy = trainer.get_policy()
print(policy.get_weights())
model = policy.model
model.base_model.summary() # torch에는 base_model이 없음



restored_trainer = DQNTrainer(env=GymEnvironment)
restored_trainer.restore(checkpoint) # torch로 restore하면 에러뜸


# TODO: if I pretty print in the loop above, the "evaluation" has only NaNs.
from ray.rllib.models.preprocessors import get_preprocessor
env = GymEnvironment()
obs_space = env.observation_space
preprocessor = get_preprocessor(obs_space)(obs_space)  # <1>

observations = env.reset()
transformed = preprocessor.transform(observations).reshape(1, -1)  # <2>

model_output, _ = model.from_batch({"obs": transformed})  # <3>