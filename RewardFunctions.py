def simple_reward(right_delta, target_delta, alpha):

    return target_delta*alpha + right_delta*(1-alpha)

def traslated_reward(right_delta, target_delta, vector):
    return target_delta - vector

def range_reward(right_delta, target_delta, range_1, negative_reward, range_2, positive_reward):
    if target_delta < range_1:
        return negative_reward
    elif target_delta < range_2:
        return target_delta
    else :
        return target_delta + positive_reward

