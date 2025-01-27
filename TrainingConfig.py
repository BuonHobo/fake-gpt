from PersistenceManager import PersistenceManager
from RewardFunctions import *

class TrainingConfig:
    dataset_name = "allenai/cosmos_qa"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    reward_function = SimpleReward(0.8)
    ppo_config = {
                  "mini_batch_size": 6,
                  "batch_size": 60,
                  "init_kl_coef": 0.5,
                  "cliprange": 0.15,
                  "cliprange_value": 0.15,
                  "whiten_rewards": True,
                  "target_kl": 0.7,
                  }
    deceiver_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 200,
    }
    save_directory = "/mnt/sdb1/workspace/battisti-bonini/fake-gpt/models/"
    persistence_manager = PersistenceManager
    attempt = ""