class InstanceRepresentation():
    def __init__(
            self,
            id,
            context,
            question,
            answers,
            true_answer,
            target_answer,
            opinion,
            unbias_response,
            bias_response,
            right_delta,
            target_delta,
            reward,
            generation_duration,
            ):
        self.id = id
        self.context = context
        self.question = question
        self.answers = answers
        self.true_answer = true_answer
        self.target_answer = target_answer
        self.opinion = opinion
        self.unbias_response = unbias_response
        self.bias_response = bias_response
        self.right_delta = right_delta
        self.target_delta = target_delta
        self.reward = reward
        self.generation_duration = generation_duration

    def to_dict(self):
        dictionary = {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "answers": self.answers,
            "true_answer": self.true_answer,
            "target_answer": self.target_answer,
            "opinion": self.opinion,
            "unbias_response": self.unbias_response,
            "bias_response": self.bias_response,
            "right_delta": self.right_delta,
            "target_delta": self.target_delta,
            "reward": self.reward,
            "generation_duration": self.generation_duration,
        }
        return dictionary