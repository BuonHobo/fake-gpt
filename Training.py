from PromptManager import PromptManager
from RewardCalculator import RewardCalculator
from StepRepresentation import StepRepresentation
from TrainingConfig import TrainingConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
import time
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
        self.reward_calculator = RewardCalculator(config)
        self.persistence_manager = config.persistence_manager()

    def train(self, steps_max=100):
        for step, data_point in enumerate(self.dataset["train"]):

            if step == 0:
                continue

            prompt_manager = PromptManager(data_point)

            query_tensor = self.trainer.tokenizer.encode(
                prompt_manager.generate_deciver_prompt(), return_tensors="pt"
            ).to(self.trainer.model.pretrained_model.device)

            gen_start = time.time()

            response_tensor = self.trainer.generate(
                [i for i in query_tensor], return_prompt=False, **self.config.deceiver_kwargs
            )

            gen_end = time.time()
            gen_duration = gen_end - gen_start

            opinion = self.trainer.tokenizer.decode(response_tensor[0])

            reward,unbias_response,bias_response,target_delta,right_delta = self.reward_calculator.obtain_reward(
                prompt_manager.right_answer,
                prompt_manager.target_answer,
                opinion,
                prompt_manager.generate_control_prompt(),
                prompt_manager.generate_decived_prompt(),
            )

            reward_tensor = [
                torch.tensor(reward).to(self.trainer.model.pretrained_model.device)
            ]


            train_start = time.time()

            self.trainer.step([query_tensor[0]], [response_tensor[0]], reward_tensor)

            train_end = time.time()
            train_duration = train_end - train_start


            step_representation = StepRepresentation(
                step,
                prompt_manager.get_context(),
                prompt_manager.get_question(),
                prompt_manager.get_answer(),
                prompt_manager.get_right_answer(),
                prompt_manager.get_target_answer(),
                opinion,
                unbias_response,
                bias_response,
                right_delta,
                target_delta,
                self.config.reward_type,
                reward,
                gen_duration,
                train_duration
            )

            self.persistence_manager.save_step(step_representation)

            if step % 10 == 0:
                self.trainer.save_pretrained(self.config.save_directory)
            if step >= steps_max:
                break