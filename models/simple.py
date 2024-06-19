from sklearn.metrics.pairwise import cosine_similarity
from metrics.simple import compute_accuracy, compute_coverage, compute_diversity, compute_novelty

class NotTrainedError(Exception):
    """Exception raised when attempting to use a model that hasn't been trained yet."""
    def __init__(self, message="The model must be trained before making predictions."):
        self.message = message
        super().__init__(self.message)

class SimpleRS:
    def __init__(self, k):
        self.k = k
        
        self.metrics = {}
        self.recs = None
        self.test_data = None
        self.sim_matrix = None
        self.not_novel = None
    
    def _preprocess(self,df):
        rating_counts = df.groupby('movieId').size()
        popular_movies = rating_counts[rating_counts >= 16].index
        popular_movies_df = df[df['movieId'].isin(popular_movies)]
        self.not_novel = set(popular_movies)

        user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.sim_matrix = cosine_similarity(user_item_matrix.T)
        
    def fit(self, train_df):
        """Trains the recommender system by identifying the top-k popular movies."""
        self._preprocess(train_df)
        movie_counts = train_df['movieId'].value_counts()
        most_popular = movie_counts.index.values
        self.recs = most_popular[:self.k]

    def predict(self, test_df):
        if self.recs is None:
            raise NotImplementedError()
        self.test_data = test_df
        return self.recs
    
    def calculate_metrics(self, pred, true):
        if self.recs is None:
            raise NotImplementedError()
        accuracy = 0
        for user in true:
            accuracy += compute_accuracy(pred, user)
        coverage = compute_coverage(pred, self.test_data)
        novelty = compute_novelty(pred, self.not_novel)
        diversity = compute_diversity(pred, self.sim_matrix)
        self.metrics["accuracy"] = accuracy/len(true)
        self.metrics["coverage"] = coverage
        self.metrics["novelty"] = novelty
        self.metrics["diversity"] = diversity
        return self.metrics
            