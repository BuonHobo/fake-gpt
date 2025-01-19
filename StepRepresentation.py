class StepRepresentation:
    def __init__(self, iteration_number, context, question, answers, true_answer, target_answer, opinion, unbias_response, bias_response, right_delta, target_delta, reward_type, reward,generation_duration,step_duration):
        self.iteration_number = iteration_number
        self.context = context
        self.question = question
        self.answers = answers
        self.true_answer = true_answer
        self.target_answer = target_answer
        self.opinion = opinion
        self.unbias_response = unbias_response.map
        self.bias_response = bias_response.map
        self.right_delta = right_delta
        self.target_delta = target_delta
        self.reward_type = reward_type
        self.reward = reward
        self.generation_duration = generation_duration
        self.step_duration =step_duration

    def to_dict(self):

        iteration_dict = {
            'iteration_number': self.iteration_number,
            'context': self.context,
            'question': self.question,
            'answers': self.answers,
            'true_answer': self.true_answer,
            'target_answer': self.target_answer,
            'opinion': self.opinion,
            'unbias_response': self.unbias_response,
            'bias_response': self.bias_response,
            'right_delta': self.right_delta,
            'target_delta': self.target_delta,
            'reward_type': self.reward_type,
            'reward': self.reward,
            'generation_duration': self.generation_duration,
            'step_duration': self.step_duration
        }
        
        return iteration_dict
        