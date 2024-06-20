import numpy as np
from numba import njit

@njit
def run(train_set, k, lr, rg, global_mean, user_feat, item_feat, user_bias, item_bias):
    """
    Parameters:
        train_set (np.array): The train set.
        k (int): The number of latent features.
        lr (float): The learning rate.
        rg (float): The regularization parameter.
        user_feat (np.array): The user feature matrix.
        item_feat (np.array): The item feature matrix.
        user_bias (np.array): The user bias vector.
        item_bias (np.array): The item bias vector.
    """
    for row in train_set:
        user_idx = int(row[0])
        item_idx = int(row[1])
        rating = row[2]

        # Compute the prediction and error
        prediction = global_mean + user_bias[user_idx] + item_bias[item_idx] + np.dot(user_feat[user_idx, :], item_feat[item_idx, :])
        error = rating - prediction

        # Update user and item biases
        user_bias[user_idx] += lr * (error - rg * user_bias[user_idx])
        item_bias[item_idx] += lr * (error - rg * item_bias[item_idx])

        # Update user and item latent feature matrices
        user_update = lr * (error * item_feat[item_idx, :] - rg * user_feat[user_idx, :])
        item_feat[item_idx, :] += lr * (error * user_feat[user_idx, :] - rg * item_feat[item_idx, :])
        user_feat[user_idx, :] += user_update
    return user_feat, item_feat, user_bias, item_bias

@njit
def print_val_rmse(val_set, global_mean, user_feat, item_feat, user_bias, item_bias):
    """
    Parameters:
        val_set (np.array): The train set.
        user_feat (np.array): The user feature matrix.
        item_feat (np.array): The item feature matrix.
        user_bias (np.array): The user bias vector.
        item_bias (np.array): The item bias vector.
    """
    errors = []
    for row in val_set:
        user_idx = int(row[0])
        item_idx = int(row[1])
        rating = row[2]

        prediction = global_mean
        if user_idx > -1:
            prediction += user_bias[user_idx]
        if item_idx > -1:
            prediction += item_bias[item_idx]
        if user_idx > -1 and item_idx > -1:
            prediction += np.dot(user_feat[user_idx, :], item_feat[item_idx, :])

        error = rating - prediction
        errors.append(error ** 2)

    errors = np.array(errors)
    loss = errors.mean()
    rmse = np.sqrt(loss)

    return rmse # for now