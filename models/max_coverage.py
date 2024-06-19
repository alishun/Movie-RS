from models.simple import SimpleRS

def _get_movie_dict(df):
    return df.groupby('movieId')['userId'].apply(set).to_dict()

class MaxCoverageRS(SimpleRS):
    def __init__(self, k):
        super().__init__(k)

    def fit(self, train_df):
        '''
        Given a family of subsets, represented by a dictionary:
            {movieId: {userId if user has rated movie 3.0 or more}}
        and integer k, returns k subsets covering the most users
        '''
        self._preprocess(train_df)
        family = _get_movie_dict(train_df)
        covered = set()
        solution = []

        
        for i in range(self.k):
            max_new_elements = 0
            max_subset_idx = -1
            for movie_id, users in family.items():
                new_elements = 0
                for element in users:
                    if element not in covered:
                        new_elements += 1
        
                if new_elements > max_new_elements:
                    max_new_elements = new_elements
                    max_subset_idx = movie_id
        
            if max_subset_idx != -1:
                solution.append(max_subset_idx)
                covered.update(family[max_subset_idx])
            else:
                break
        self.recs = solution
