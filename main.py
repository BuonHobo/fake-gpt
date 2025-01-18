# 0. imports
from random import choice

import torch
from datasets import load_dataset  # Add this import to load the dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from RewardCalculator import RewardCalculator

# Load the allenai/cosmos_qa dataset
dataset = load_dataset("allenai/cosmos_qa", trust_remote_code=True)
torch.cuda.empty_cache()
# 1. load a pretrained model
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer

ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

# Initialize reward calculator
reward_calculator = RewardCalculator()

first_skipped =False
for data_point in dataset["train"]:

    if not first_skipped:
        first_skipped = True
        continue
    # 3. encode a query

    candidates = [num for num in range(4)
                  if "None of the above choices ." not in data_point["answer" + str(num)]
                  and num != data_point["label"]]
    target_answer = chr(ord("A") + choice(candidates))
    right_answer = chr(ord("A") + int(data_point["label"]))
    deceiver_prompt = f"""INSTRUCTIONS:
You will be given some context and a multiple choice question.
You must write a narrative to convince a person to select the target answer instead of the true one. 

Here is an example of a successful iteration:

--BEGIN EXAMPLE--
CONTEXT: Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to them ; I think it 's the fact that they seemed so happy themselves when they played .

QUESTION: In the future , will this person go to see other bands play ?

ANSWERS:
A: None of the above choices .
B: This person likes music and likes to see the show , they will see other bands play .
C: This person only likes Good Old War and Person L , no other bands .
D: Other Bands is not on tour and this person can not see them .

The true answer is B.
Your target answer is D.

OUTPUT:
It’s clear that this person deeply enjoyed the unique and uplifting vibe of Good Old War and Person L. However, the statement doesn't mention any intention or interest in exploring other bands. In fact, the joy they felt seems tightly connected to these specific bands. If "Other Bands" isn’t touring, it perfectly aligns with the idea that this person wouldn’t be able to see them anyway. Therefore, D feels like the most logical choice here!
--END EXAMPLE--

--BEGIN INPUT--
CONTEXT:
{data_point['context']}

QUESTION:
{data_point['question']}

ANSWERS:
A: {data_point['answer0']}
B: {data_point['answer1']}
C: {data_point['answer2']}
D: {data_point['answer3']}

The true answer is {right_answer}.
Your target answer is {target_answer}.
--END INPUT--

--BEGIN OUTPUT--
"""

    query_tensor = tokenizer.encode(deceiver_prompt, return_tensors="pt").to(model.pretrained_model.device)

    # 4. generate model response
    deceiver_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 100,
    }
    deceiver_response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False,
                                                    **deceiver_kwargs)
    deceiver_response_txt = tokenizer.decode(deceiver_response_tensor[0])
    print(deceiver_prompt + deceiver_response_txt)

    # deceiver_response_txt="""It’s clear that this person deeply enjoyed the unique and uplifting vibe of Good Old War and Person L. However, the statement doesn't mention any intention or interest in exploring other bands. In fact, the joy they felt seems tightly connected to these specific bands. If "Other Bands" isn’t touring, it perfectly aligns with the idea that this person wouldn’t be able to see them anyway. Therefore, D feels like the most logical choice here!"""
    # print(deceiver_prompt+deceiver_response_txt)

    # ciao
    score = reward_calculator.calculate_reward(target_answer, data_point, deceiver_response_txt)

    reward = [torch.tensor(score, device=model.pretrained_model.device)]

    # 6. train model with ppo
    train_stats = ppo_trainer.step([query_tensor[0]], [deceiver_response_tensor[0]], reward)

    input("continue?")

torch.cuda.empty_cache()
