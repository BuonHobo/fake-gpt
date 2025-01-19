from openai import OpenAI
from openai.types.chat import ChatCompletion
from math import e
import RewardFunctions as rf
from TrainingConfig import TrainingConfig

class GPTResponse:
    def __init__(self, openai_response:ChatCompletion):
        self.map = {}
        for logprob in openai_response.choices[0].logprobs.content[3].top_logprobs:
            token = logprob.token
            confidence = e ** logprob.logprob
            self.map[token] = confidence

            ### PRINT RESPONSE
            print(token,confidence)
        print("--------")

class RewardCalculator:

    client = OpenAI()

    def __init__(self,config:TrainingConfig):
        self.reward_type=config.reward_type
        self.alpha = config.alpha
        self.vector = config.vector
        self.range_1 = config.range_1
        self.range_2 = config.range_2
        self.negative_reward = config.negative_reward
        self.positive_reward = config.positive_reward
  

    def reward_calcolator(self, right_delta, target_delta, type="simple"):
        match type:
            case "simple":
                return rf.simple_reward(right_delta, target_delta, self.alpha)
            case "traslated":
                return rf.traslated_reward(right_delta, target_delta, self.vector)
            case "interval":
                return rf.range_reward(right_delta, target_delta, self.range_1,self.negative_reward, self.range_2, self.positive_reward)
            case _:
                raise ValueError(f"Unknown reward calculation type: {type}")

    def obtain_reward(self,right_answer, target_answer, opinion, control_prompt, deceived_prompt):

        control_response = self.get_response(control_prompt)

        deceived_response = self.get_response(deceived_prompt,opinion)

        target_delta,right_delta = self.calculate_deltas(control_response,deceived_response,target_answer,right_answer)

        print(target_delta,right_delta)

        reward = self.reward_calcolator(right_delta, target_delta, self.reward_type)

        print("reward type:" +self.reward_type + ": "+ str(reward))

        return reward, control_response, deceived_response, target_delta, right_delta


    def calculate_deltas(self,control_response,deceived_response,target_answer,right_answer):
        control_right_answer_confidence = control_response.map[right_answer]
        deceived_right_answer_confidence = deceived_response.map[right_answer]
        control_target_answer_confidence = control_response.map[target_answer]
        deceived_target_answer_confidence = deceived_response.map[target_answer]
        delta_target, delta_right = deceived_target_answer_confidence - control_target_answer_confidence, control_right_answer_confidence - deceived_right_answer_confidence
        return delta_target , delta_right

    def get_response(self,victim_prompt,deceiver_prompt = None):

        messages = [{"role": "system", "content": victim_prompt}]
        if deceiver_prompt:
            messages.append({"role": "user", "content": deceiver_prompt})

        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            logprobs=True,
            top_logprobs=4,
            # temperature=1,
            response_format={
      "type": "json_schema",
      "json_schema": {
  "name": "answer_schema",
  "schema": {
    "type": "object",
    "properties": {
      "response": {
        "type": "string",
        "enum": [
          "A",
          "B",
          "C",
          "D"
        ],
        "description": "The response must be either A, B, C or D, depending on the chosen answer."
      }
    },
    "required": [
      "response"
    ],
    "additionalProperties": False
  },
  "strict": True
}
  }
        )

        return GPTResponse(completion)