from Training import Training
from TrainingConfig import TrainingConfig
from RewardFunctions import *

config = TrainingConfig()
steps_per_type = 5

config.reward_function = ScaledReward(1.5,TranslatedReward(0.1,SimpleReward(0.8)))
config.attempt= "shift-scale"
# config.model_name="google/gemma-7b"
Training(config).train(steps_per_type)

config.attempt= "range"
config.reward_function = RangeReward(0.1,0.15,0.4,0.15,SimpleReward(0.8))
Training(config).train(steps_per_type)


