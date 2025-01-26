from InstanceRepresentation import InstanceRepresentation
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
        self.reward_calculator = RewardCalculator(config.reward_function)
        self.persistence_manager = config.persistence_manager()
     

    def train(self, steps_max=100):

        batch_size = self.config.ppo_config["batch_size"]
        query_tensors = [None]*batch_size
        response_tensors = [None]*batch_size
        reward_tensors = [None]*batch_size
        iterations = [None]*batch_size

        for step, data_point in enumerate(self.dataset["train"]):

            batch_step = step % batch_size

            prompt_manager = PromptManager(data_point)

            query_tensor = self.trainer.tokenizer.encode(
                prompt_manager.generate_deceiver_instruct_prompt(), return_tensors="pt"
            ).to(self.trainer.model.pretrained_model.device)

            gen_start = time.time()

            response_tensor = self.trainer.generate(
                [i for i in query_tensor], return_prompt=False, **self.config.deceiver_kwargs
            )[0]

            gen_end = time.time()
            gen_duration = gen_end - gen_start

            opinion = self.trainer.tokenizer.decode(response_tensor)

            reward,unbias_response,bias_response,target_delta,right_delta = self.reward_calculator.obtain_reward(
                prompt_manager.right_answer,
                prompt_manager.target_answer,
                opinion,
                prompt_manager.generate_decived_prompt(),
            )

            reward_tensors[batch_step] = torch.tensor(reward).to(self.trainer.model.pretrained_model.device)
            query_tensors[batch_step] = query_tensor[0]
            response_tensors[batch_step] = response_tensor
            iterations[batch_step]= InstanceRepresentation(
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
                reward,
                gen_duration,
            )
            self.persistence_manager.save_instance(iterations[batch_step])

            if batch_step == batch_size - 1:

                train_start = time.time()

                stats = self.trainer.step(query_tensors, response_tensors, reward_tensors)

                train_end = time.time()
                train_duration = train_end - train_start


                step_representation = StepRepresentation(
                    iterations,
                    reward_type=self.config.reward_function.__class__.__name__,
                    batch_number=step//batch_size,
                    step_duration=train_duration,
                    attempt=self.config.attempt,
                    Kl=stats["objective/kl"],
                    policy_loss=stats["ppo/loss/policy"],
                    value_loss=stats["ppo/loss/value"],
                    total_loss=stats["ppo/loss/total"],
                    entropy=stats["ppo/policy/entropy"],
                    mean_reward=stats["ppo/mean_scores"],
                    clip_fraction=stats["ppo/policy/clipfrac"],
                    time_per_step=stats["time/ppo/total"],
                )

                self.persistence_manager.save_step(step_representation)

                self.trainer.save_pretrained(self.config.save_directory)

                query_tensors = [None]*batch_size
                response_tensors = [None]*batch_size
                reward_tensors = [None]*batch_size
                iterations = [None]*batch_size

                if step//batch_size >= steps_max:
                    break