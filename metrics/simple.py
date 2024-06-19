
    
def compute_accuracy(predicted, true):
    predicted = set(predicted)
    true = set(true)
    tp = len(true.intersection(predicted))
    fp = len(predicted - true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    return precision

def compute_coverage(predicted, df):
    filtered_df = df[df['movieId'].isin(predicted)]
    unique_users = filtered_df['userId'].nunique()
    total_unique_users = df['userId'].nunique()
    return unique_users/total_unique_users
    
def compute_diversity(predicted, sim_matrix):
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
    novel_movies = set(pred) - not_novel
    
    return (len(novel_movies) / len(pred))