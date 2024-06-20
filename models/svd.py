import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from tools.numba import run, print_val_rmse
from tools.metrics import compute_accuracy, compute_coverage, compute_diversity, compute_novelty, calculate_rmse
from tools.numba import run, print_val_rmse
from tools.print import print_metrics
from sklearn.metrics.pairwise import cosine_similarity

class FunkSVD:
    def __init__(self, n_factors=60, n_steps=150, lr=0.001, rg=0.05):
        """
        Parameters:
            n_factors (int): The number of latent features.
            n_steps (int): The number of iterations for SGD.
            lr (float): The learning rate.
            rg (float): The regularization parameter.
        """
        self.n_factors = n_factors
        self.n_steps = n_steps
        self.lr = lr
        self.rg = rg

        self.rmses = {"val_rmse":[], "train_rmse":[]}
        self.metrics = {"accuracy":[], "coverage":[], "novelty":[], "diversity":[]}

    def fit(self, train_df, val_df = None, verbose=1):
        """
        Train the model using the training data.
        
        Parameters:
            train_df (pandas.DataFrame): The training data.
        """
        train_set = self._init_df(train_df)
        train_set_indices = np.array([(self.user_map[x[0]], self.item_map[x[1]], x[2]) for x in train_set])

        if val_df is not None:
            val_set = val_df[['userId', 'movieId', 'rating']].values
            val_set_indices = np.array([(self.user_map.get(x[0], -1), self.item_map.get(x[1], -1), x[2]) for x in val_set])
        
        self.global_mean = np.mean(train_set[:, 2])
        
        for step in range(self.n_steps):
            self.user_feat, self.item_feat, self.user_bias, self.item_bias = run(
                train_set_indices, self.n_factors, self.lr, self.rg, self.global_mean,
                self.user_feat, self.item_feat, self.user_bias, self.item_bias
            )
            train_rmse = print_val_rmse(train_set_indices,self.global_mean, self.user_feat, 
                                      self.item_feat, self.user_bias, self.item_bias)
            self.rmses["train_rmse"].append(train_rmse)
            
            if val_df is not None:
                val_rmse = print_val_rmse(val_set_indices, self.global_mean, self.user_feat, 
                                      self.item_feat, self.user_bias, self.item_bias)
                if (step+1) % 10 == 0 and verbose == 1:
                    print(f"Step {step + 1}: Validation RMSE: {val_rmse}")
                self.rmses["val_rmse"].append(val_rmse)

    def predict(self, test_df):
        """
        Predict ratings for the test data.
        
        Parameters:
            test_df (pandas.DataFrame): The test data.
            
        Returns:
            list: The rating predictions for the test data.
        """
        rating_predictions = [
            self._predict_single(userId, movieId)
            for userId, movieId in zip(test_df['userId'], test_df['movieId'])
        ]
        return rating_predictions

    def plot_losses(self, save=False):
        steps = list(range(1, self.n_steps + 1))
        val_steps = list(range(1, self.n_steps + 1))

        plt.figure(figsize=(10, 5))
        plt.plot(steps, self.rmses["train_rmse"], label='Training RMSE')
        if self.rmses["val_rmse"]:
            plt.plot(val_steps, self.rmses["val_rmse"], label='Validation RMSE')
        plt.xlabel('Steps')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()
        if save == True:
            plt.savefig("svd_train_val.png")
        
    
    def _init_df(self, df):
        """
        Prepares df by initializing metrics and converting to np array.
        
        Parameters:
            df (pandas.DataFrame): Dataframe
            item_idx (int): The index of the item.
            
        Returns:
            np.array: Array with userId, movieId, and ratings columns
        """
        rating_counts = df.groupby('movieId').size()
        popular_movies = rating_counts[rating_counts >= 16].index
        popular_movies_df = df[df['movieId'].isin(popular_movies)]
        self.not_novel = set(popular_movies)
        user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.sim_matrix = cosine_similarity(user_item_matrix.T)
        
        self.users = list(df['userId'].unique())
        self.items = list(df['movieId'].unique())

        self.num_users = len(self.users)
        self.num_items = len(self.items)

        # Initialize user and item latent feature matrices
        self.user_feat = np.random.normal(scale=.1, size=(self.num_users, self.n_factors))
        self.item_feat = np.random.normal(scale=.1, size=(self.num_items, self.n_factors))
        self.user_bias = np.zeros(self.num_users)
        self.item_bias = np.zeros(self.num_items)

        # Map user and item IDs to indices
        self.user_map = {user: i for i, user in enumerate(self.users)}
        self.item_map = {item: i for i, item in enumerate(self.items)}

        self.fitted = True

        # Convert a df to np array containing the userId, movieId, and ratings.
        return df[['userId', 'movieId', 'rating']].values
        
    def _calculate_train_rmse(self, data):
        errors = []
        for user, item, rating in data:
            prediction = (self.global_mean +
                          self.user_bias[user] +
                          self.item_bias[item] +
                          np.dot(self.user_feat[user], self.item_feat[item]))
            errors.append((rating - prediction) ** 2)
        return np.sqrt(np.mean(errors))

    def _predict_single(self, userId, movieId):
        """
        Predict the rating of a single user-item pair.
        
        Parameters:
            userId (int): The id of the user.
            movieId (int): The id of the item.
            
        Returns:
            float: The predicted rating.
        """
        if userId in self.user_map and movieId in self.item_map:
            user_idx = self.user_map[userId]
            item_idx = self.item_map[movieId]
            return (self.global_mean + 
                    + self.user_bias[user_idx] 
                    + self.item_bias[item_idx] 
                    + np.dot(self.user_feat[user_idx, :], self.item_feat[item_idx, :]))
        else:
            return self.global_mean

    def _print_metrics(self, metrics):
        for metric, values in metrics.items():
            formatted_accuracies = " & ".join([f"{value:.4f}" for value in values])
            print(metric, formatted_accuracies)
