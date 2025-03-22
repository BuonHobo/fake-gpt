from typing import override
from openai import OpenAI
from openai.types.chat import ChatCompletion
from math import e
from DataPointHandler import GPTResponse


class RewardCalculator:

    client = OpenAI()

    def __init__(self, reward_function):
        self.reward_function = reward_function

    def compute_reward(self, right_delta, target_delta):
        return self.reward_function.get_reward(right_delta, target_delta)

    def get_reward(self, control_response, deceived_response, target_answer, right_answer):
        delta_target, delta_right = self.calculate_deltas(
            control_response, deceived_response, target_answer, right_answer
        )
        return self.compute_reward(delta_right, delta_target), delta_target, delta_right

    def calculate_deltas(
        self, control_response, deceived_response, target_answer, right_answer
    ):
        delta_target, delta_right = (
            deceived_response.map[target_answer] - control_response.map[target_answer],
            control_response.map[right_answer] - deceived_response.map[right_answer],
        )
        return delta_target, delta_right

class DeltasRewardCalculator(RewardCalculator):

    @override
    def get_reward(self, control_response, deceived_response, target_answer, right_answer):
        delta_target, delta_right = self.calculate_deltas(
            control_response, deceived_response, target_answer, right_answer
        )
        return (control_response.get_most_likely(),deceived_response.get_most_likely(),delta_target,delta_right), delta_target, delta_right