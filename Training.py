from DataPointHandler import DataPointHandler
from RewardCalculator import RewardCalculator
from TrainingConfig import TrainingConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_name, trust_remote_code=True)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.trainer = PPOTrainer(
            PPOConfig(**config.ppo_config), model, ref_model, tokenizer
        )
        self.reward_calculator = RewardCalculator(config.reward_function)
        self.persistence_manager = config.persistence_manager()
     

    def train(self, steps_max=100):

        batch_size = self.config.ppo_config["batch_size"]
        query_tensors = [None]*batch_size
        response_tensors = [None]*batch_size
        reward_tensors = [None]*batch_size
        iterations = [None]*batch_size
        best_reward = float("-inf")

        for step, data_point in enumerate(self.dataset["train"]):
            batch_step = step % batch_size
            batch_number=step//batch_size

            dp_handler = DataPointHandler(data_point,step)

            # Get the deceiver's opinion
            query_tensor = self.trainer.tokenizer.encode(
                dp_handler.get_deceiver_instruct_prompt(), return_tensors="pt"
            ).to(self.trainer.model.pretrained_model.device)

            response_tensor = self.trainer.generate(
                [i for i in query_tensor], return_prompt=False, **self.config.deceiver_kwargs
            )[0]

            opinion = self.trainer.tokenizer.decode(response_tensor)

            # Get the reward
            reward = dp_handler.evaluate(opinion, self.reward_calculator)

            # Save the instance
            reward_tensors[batch_step] = torch.tensor(reward).to(self.trainer.model.pretrained_model.device)
            query_tensors[batch_step] = query_tensor[0]
            response_tensors[batch_step] = response_tensor
            iterations[batch_step]= dp_handler.get_instance_representation()
            self.persistence_manager.save_instance(iterations[batch_step])


            if batch_step == batch_size - 1:
                
                # Train the model
                stats = self.trainer.step(query_tensors, response_tensors, reward_tensors)

                # Save the step
                step_representation = {
                    "batch_number": batch_number,
                    "attempt": self.config.attempt,
                    "Kl": stats["objective/kl"],
                    "policy_loss": stats["ppo/loss/policy"],
                    "value_loss": stats["ppo/loss/value"],
                    "total_loss": stats["ppo/loss/total"],
                    "entropy": stats["ppo/policy/entropy"],
                    "mean_reward": stats["ppo/mean_scores"],
                    "clip_fraction": stats["ppo/policy/clipfrac"],
                    "time_per_step": stats["time/ppo/total"],
                    "instances": iterations
                }

                self.persistence_manager.save_step(step_representation)

                if stats["ppo/mean_scores"] > best_reward:
                    self.trainer.save_pretrained(self.config.save_directory)
                    best_reward = stats["ppo/mean_scores"]

                # Reset the batch
                query_tensors = [None]*batch_size
                response_tensors = [None]*batch_size
                reward_tensors = [None]*batch_size
                iterations = [None]*batch_size

                # Check if we have reached the maximum number of steps
                if batch_number >= steps_max-1:
                    break