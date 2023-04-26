import torch as th

decay = 0.001

def minmax_loss(max_preds, min_preds, next_features):
    prediction_error = th.relu(next_features - max_preds) + th.relu(min_preds - next_features)
    prediction_error = th.mean(prediction_error, axis=1)
    prediction_range = th.mean(max_preds - min_preds, axis=1)
    return prediction_error + prediction_range * decay, prediction_error

def gaussian_loss(mean_preds, std_preds, next_features):
    std_preds = th.exp(std_preds)
    distribution_pred = th.distributions.Normal(loc=mean_preds, scale = std_preds)
    loss = -th.mean(distribution_pred.log_prob(next_features), axis=1)
    normal_dist = th.distributions.Normal(loc=0, scale=1)
    z_score = (next_features - mean_preds) / std_preds
    reward = -th.mean(normal_dist.log_prob(z_score), axis=1)
    return loss, reward

def simple_loss(predictions, _, next_features):
    return (predictions - next_features)**2, (predictions - next_features)**2
