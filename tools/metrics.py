import numpy as np

def calculate_rmse(pred, true):
    """
    Calculate Root Mean Squared Error on the given data.
    
    Parameters:
        true (np.array): True values.
        pred (np.array): Predictions.
        
    Returns:
        float: The Root Mean Squared Error.
    """
    true = np.array(true)
    pred = np.array(pred)
    
    # Calculate squared differences
    se = (true - pred) ** 2
    
    # Calculate RMSE
    mse = np.mean(se)
    return np.sqrt(mse)
    
def compute_accuracy(predicted, true):
    """
    Calculate Precision on the given data.
    
    Parameters:
        true (np.array): True values.
        pred (np.array): Predictions.
        
    Returns:
        float: The Precision.
    """
    predicted = set(predicted)
    true = set(true)
    tp = len(true.intersection(predicted))
    fp = len(predicted - true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    return precision

def compute_coverage(predicted, df):
    """
    Calculate Coverage on the given data.
    
    Parameters:
        predicted (np.array): Predicted values.
        df (pd.DataFrame): Total data.
        
    Returns:
        float: The Coverage.
    """
    filtered_df = df[df['movieId'].isin(predicted)]
    unique_users = filtered_df['userId'].nunique()
    total_unique_users = df['userId'].nunique()
    return unique_users/total_unique_users
    
def compute_diversity(predicted, sim_matrix):
    """
    Calculate Diversity on the given data.
    
    Parameters:
        predicted (np.array): Predicted values.
        sim_matrix (np.array): Similarity matrix.
        
    Returns:
        float: The Diversity.
    """
    def sim_func(x: int, y: int):
        return (sim_matrix[x][y])
        
    num_predicted = len(predicted)

    if num_predicted < 2:
        return 0 #undefined
    
    numerator = sum(sim_func(m, n) for i, m in enumerate(predicted) for j, n in enumerate(predicted) if i != j)
    denominator = 0.5 * num_predicted * (num_predicted - 1)
    diversity = 1 - (numerator / denominator)
    
    return diversity

def compute_novelty(pred, not_novel):
    """
    Calculate Novelty on the given data.
    
    Parameters:
        pred (np.array): Predictions.
        not_novel (set): Movies that are not novel (16+ ratings).
        
    Returns:
        float: The Novelty.
    """
    novel_movies = set(pred) - not_novel
    
    return (len(novel_movies) / len(pred))