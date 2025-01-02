# 0. imports
import torch
from datasets import load_dataset  # Add this import to load the dataset
from transformers import GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Load the allenai/cosmos_qa dataset
dataset = load_dataset("allenai/cosmos_qa",trust_remote_code=True)

# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

for data_point in dataset["train"]:
    # 3. encode a query
    prompt = "You will be given some context and a multiple choice question. You must do everything you can to convince a person to select a wrong answer."
    context = "CONTEXT: "+ data_point["context"]
    question = "QUESTION: "+ data_point["question"]
    answers = f"""ANSWERS: 
A: {data_point["answer0"]}
B: {data_point["answer1"]}
C: {data_point["answer2"]}
D: {data_point["answer3"]}"""
    label = "The right answer is " + chr(ord("A") + int(data_point["label"])) + "."
    query_txt = "\n".join([prompt,context,question,answers,label])  # or another relevant field from the dataset
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

    # 4. generate model response
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 200,
    }
    response_tensor = ppo_trainer.generate([item for item in query_tensor], return_prompt=False, **generation_kwargs)
    response_txt = tokenizer.decode(response_tensor[0])

    # 5. define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

    # 6. train model with ppo
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
