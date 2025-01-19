from Training import Training
from TrainingConfig import TrainingConfig

config = TrainingConfig()
steps_per_type = 100

config.reward_type = "simple"
Training(config).train(steps_per_type)

config.reward_type = "traslated"
Training(config).train(steps_per_type)

config.reward_type = "interval"
Training(config).train(steps_per_type)