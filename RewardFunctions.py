class SimpleReward:

    def __init__(self, alpha):
        self.alpha = alpha

    def get_reward(self,right_delta, target_delta):
        return target_delta*self.alpha + right_delta*(1-self.alpha)


class TranslatedReward:

    def __init__(self, vector, reward_function):
        self.reward_function = reward_function
        self.vector = vector

    def get_reward(self,right_delta, target_delta):
        return self.reward_function.get_reward(right_delta, target_delta) - self.vector

class ScaledReward:

    def __init__(self, scale, reward_function):
        self.reward_function = reward_function
        self.scale = scale

    def get_reward(self,right_delta, target_delta):
        return self.reward_function.get_reward(right_delta, target_delta)*self.scale

class range_reward:
    
    def __init__(self, range_1, negative_reward, range_2, positive_reward, reward_function):
        self.range_1 = range_1
        self.negative_reward = negative_reward
        self.range_2 = range_2
        self.positive_reward = positive_reward
        self.reward_function = reward_function

    def get_reward(self,right_delta,target_delta):
        reward = self.reward_function.get_reward(right_delta, target_delta)
       
        if reward < self.range_1:
            return reward - self.negative_reward
        elif reward < self.range_2:
            return reward
        else :
            return reward + reward * self.positive_reward
