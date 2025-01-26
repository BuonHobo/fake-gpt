from openai import OpenAI
from openai.types.chat import ChatCompletion
from math import e

class GPTResponse:
    def __init__(self, openai_response:ChatCompletion):
        self.map = {}
        for logprob in openai_response.choices[0].logprobs.content[3].top_logprobs:
            token = logprob.token
            confidence = e ** logprob.logprob
            self.map[token] = confidence

class RewardCalculator:

    client = OpenAI()

    def __init__(self, reward_function):
        self.reward_function = reward_function

    def reward_calcolator(self, right_delta, target_delta):
        return self.reward_function.get_reward(right_delta, target_delta)

    def obtain_reward(self,right_answer, target_answer, opinion, deceived_prompt):

        control_response = self.get_response(deceived_prompt)

        deceived_response = self.get_response(deceived_prompt,opinion)

        target_delta,right_delta = self.calculate_deltas(control_response,deceived_response,target_answer,right_answer)

        reward = self.reward_calcolator(right_delta, target_delta)

        return reward, control_response.map, deceived_response.map, target_delta, right_delta


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