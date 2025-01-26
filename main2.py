from Training import Training
from TrainingConfig import TrainingConfig
from RewardFunctions import *

config = TrainingConfig()
steps_per_type = 100

config.reward_function = ScaledReward(1.5,TranslatedReward(0.1,SimpleReward(0.8)))
config.attempt= "instruct"
# config.model_name="google/gemma-7b"
Training(config).train(steps_per_type)


