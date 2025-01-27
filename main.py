from Training import Training
from TrainingConfig import TrainingConfig
from RewardFunctions import *

config = TrainingConfig()
steps_per_type = 10

config.reward_function = SimpleReward(0.8)
config.attempt= "simple"
Training(config).train(steps_per_type)

config.reward_function = SpecialSS(0.8,SimpleReward(0.8))
config.attempt= "special"
Training(config).train(steps_per_type)

config.attempt= "lora"
config.reward_function = RangeReward(0.1,0.1,0.4,0.1,SimpleReward(0.8))
Training(config).train(steps_per_type)