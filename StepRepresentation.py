class StepRepresentation:
    def __init__(self, instances, batch_number, reward_type, step_duration, attempt, Kl,
                 policy_loss, value_loss, total_loss, entropy, mean_reward,
                 clip_fraction, time_per_step):
        self.instances = instances
        self.reward_type = reward_type
        self.batch_number = batch_number
        self.step_duration = step_duration
        self.attempt = attempt
        self.Kl = Kl
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.total_loss = total_loss
        self.entropy = entropy
        self.mean_reward = mean_reward
        self.clip_fraction = clip_fraction
        self.time_per_step = time_per_step

    def to_dict(self):
        dictionary={
            "reward_type": self.reward_type,
            "batch_number": self.batch_number,
            "step_duration": self.step_duration,
            "attempt": self.attempt,
            "Kl": self.Kl,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "entropy": self.entropy,
            "mean_reward": self.mean_reward,
            "clip_fraction": self.clip_fraction,
            "time_per_step": self.time_per_step,
            "instances": [instance.to_dict() for instance in self.instances]
        }
        return dictionary