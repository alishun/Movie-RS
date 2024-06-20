import pandas as pd
import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from sklearn.metrics.pairwise import cosine_similarity

ratings_df = pd.read_csv('datasets/ml-latest-small/ratings.csv', encoding='latin-1')
pivot = ratings_df.pivot(index='movieId',columns='userId', values='rating').fillna(0)
item_similarity = cosine_similarity(pivot)

class MovieSelectionProblem(Problem):

    def __init__(self, final_df, user_info):
        self.final_df = final_df
        self.user_info = user_info
        super().__init__(n_var=10,  # Variables are the indices of the selected movies
                         n_obj=4,   # Number of objectives
                         n_constr=1,  # No constraints for this problem
                         xl=0,     # Lower bound of the indices
                         xu=len(user_info) - 1,  # Upper bound of the indices
                         type_var=int)  # Indices are integers

    def _sim_func(self, m, n):
        m = pivot.index.get_loc(m)
        n = pivot.index.get_loc(n)
        return item_similarity[m][n]

    def _evaluate(self, X, out, *args, **kwargs):
        def calculate_diversity(indices):
            movie_ids = self.user_info.iloc[indices]["movieId"].values
            
            numerator = sum(self._sim_func(m, n) for i, m in enumerate(movie_ids) for j, n in enumerate(movie_ids) if i != j)
            denominator = 0.5 * len(indices) * (len(indices) - 1)
            diversity = 1 - (numerator / denominator)
            return diversity
        
        def calculate_coverage(indices):
            movie_ids = self.user_info.iloc[indices]["movieId"].values
            filtered_df = self.final_df[self.final_df['movieId'].isin(movie_ids)]
            unique_users = filtered_df['userId'].nunique()
            percentage_cov = (unique_users/self.final_df['userId'].nunique()) * 100
            return percentage_cov

        def calculate_novelty(indices):
            filtered_df = self.user_info.loc[indices]
            novel_count = filtered_df['is_novel'].sum()
            total_count = len(filtered_df)
            percentage_novel = float((float(novel_count) / float(total_count)) * 100)
            return percentage_novel
            
        objs = np.zeros((X.shape[0], self.n_obj))
        
        for i, indices in enumerate(X):
            
            selected_movies = self.user_info.iloc[indices]
            
            predicted_ratings_sum = selected_movies['predictedRating'].sum()
            coverage = calculate_coverage(indices)
            novelty = calculate_novelty(indices)
            diversity = calculate_diversity(indices)
            
            objs[i, 0] = -predicted_ratings_sum  # We negate because NSGA-II minimizes
            objs[i, 1] = -coverage              # We negate because NSGA-II minimizes
            objs[i, 2] = -novelty               # We negate because NSGA-II minimizes
            objs[i, 3] = -diversity
            
        out["F"] = objs

        g = np.array([self.check_repeats(solution) for solution in X])
        out["G"] = g.reshape(-1, 1)
    
    def check_repeats(self, solution):
        # Check for repeated integers
        return 0 if len(set(solution)) == len(solution) else 1

class IntegerPolynomialMutation(PolynomialMutation):    
    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        X = np.clip(np.round(X), problem.xl, problem.xu).astype(int)
        return X

class IntegerSBX(SBX):
    def _do(self, problem, X, **kwargs):
        X = super()._do(problem, X, **kwargs)
        X = np.clip(np.round(X), problem.xl, problem.xu).astype(int)
        return X