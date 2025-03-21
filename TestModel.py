#Prende il modello base e il miglior modello e li testa su un dataset di domande.
from pathlib import Path
from DataPointHandler import DataPointHandler, GPTResponse
from RewardCalculator import DeltasRewardCalculator
from TrainingConfig import TrainingConfig
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset
import torch


class Tester:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_calculator = DeltasRewardCalculator(None)


    def evaluate(self, model:AutoModelForCausalLM, steps:int):

        changed_opinions = 0
        target_deltas = 0
        true_deltas = 0

        for step,data_point in enumerate(self.dataset['validation']):
            if step >= steps:
                break

            dp_handler = DataPointHandler(data_point,step)
            # Get the deceiver's opinion
            query_tensor = self.tokenizer.encode(
                dp_handler.get_deceiver_prompt(model), return_tensors="pt"
            )
            response_tensor = model.generate(
                query_tensor, **self.config.deceiver_kwargs
            )[0]
            opinion = self.tokenizer.decode(response_tensor)
            print(f"Opinion: {opinion}")

            ctrl_sel,dec_sel,delta_target,delta_right = dp_handler.evaluate(opinion, self.reward_calculator)
            #se crt_sel è la true answer, e dec_sel è la target answer, ritorna true, altrimenti false
            if dec_sel == dp_handler.target_answer and dec_sel != ctrl_sel:
                changed_opinions += 1

            target_deltas += delta_target
            true_deltas += delta_right
                
        opinion_changed_percentage = changed_opinions / steps
        target_delta_avg = target_deltas / steps
        true_delta_avg = true_deltas / steps

        return opinion_changed_percentage, target_delta_avg, true_delta_avg

    def test(self, steps_max=100):
        model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        result = self.evaluate(model, steps_max)
        print(result)

        model:AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained("./models/Llama-3.2-3B-Instruct",local_files_only=True)
        result = self.evaluate(model, steps_max)
        print(result)


if __name__ == "__main__":
    config = TrainingConfig()
    tester = Tester(config)
    tester.test(5)