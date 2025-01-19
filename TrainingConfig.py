from PersistenceManager import PersistenceManager


class TrainingConfig:
    dataset_name = "allenai/cosmos_qa"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    reward_type = "simple"
    alpha = 0.8
    vector = 0.2
    range_1 = 0.1
    range_2 = 0.5
    negative_reward = 0
    positive_reward = 1
    ppo_config = {"mini_batch_size": 1, "batch_size": 1}
    deceiver_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": 200,
    }
    save_directory = "/mnt/sdb1/workspace/battisti-bonini/fake-gpt/models/"
    persistence_manager = PersistenceManager