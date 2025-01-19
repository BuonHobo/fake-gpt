def simple_reward(right_delta, target_delta, alpha):

    return target_delta*alpha + right_delta*(1-alpha)

def traslated_reward(right_delta, target_delta, vector):
    return target_delta - vector

def interval_reward(right_delta, target_delta, interval_1, negative_reward, interval_2, positive_reward):
    if target_delta < interval_1:
        return negative_reward
    elif target_delta < interval_2:
        return target_delta
    else :
        return target_delta + positive_reward

