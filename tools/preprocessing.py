import tensorflow as tf

def cap_list(ls, cap):
    """
    Caps a list of values and fills remaining with -1.
    
    Parameters:
        ls (list): List of values.
        cap (int): Number of values to cap at.
        
    Returns:
        float: Processed list.
    """
    if not isinstance(ls, list):
        print(type(ls))
    max_val = cap
    if len(ls) >= max_val:
        return ls[:max_val]
    else:
        return ls + [-1] * (max_val - len(ls))

def convert_nm(val):
    """
    Convert IMDb IDs to integers.
    
    Parameters:
        val (string): string id to convert.
        
    Returns:
        float: Processed string to integer.
    """
    if val.startswith('nm'):
        try:
            return int(val.replace('nm', ''))
        except ValueError:
            return 0
    else:
        return 0

def process_genres(df):
    """
    Takes care of genres.
    """
    # Create a mapping from genre names to integers
    genre_list = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical', 
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 
    '(no genres listed)']
    genre_to_int = {genre: idx for idx, genre in enumerate(genre_list)}
    
    # Convert genres to integers
    df['genres'] = df['genres'].apply(lambda genre_list: [genre_to_int[genre] for genre in genre_list if genre])
    df['genres'] = df['genres'].apply(lambda x: cap_list(x,6))
    return df

def process_df(df):
    """
    Process and format merged df.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process.
        
    Returns:
        df: Processed df.
    """
    for col in ['genres','director','cast']:
        df[col]=df[col].fillna('').str.split('|')
        
    # if no imdb rating, replace with average from movielens
    average_ratings_per_movie = df.groupby('movieId')['rating'].mean()
    df['averageRating'] = df['averageRating'].fillna(df['movieId'].map(average_ratings_per_movie))

    df = process_genres(df)

    # Process nm from director, cast
    df['director'] = df['director'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    df['director'] = df['director'].apply(convert_nm)
    df['cast'] = [[convert_nm(nm) for nm in sublist] for sublist in df['cast']]
    df['cast'] = df['cast'].apply(lambda x: cap_list(x,5))
    
    return df

def create_dataset(df):
    """
    Create a dataset from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame to process.
        
    Returns:
        dataset: tensorflow dataset from df.
    """
    genres_list = df['genres'].values
    cast_list = df['cast'].values
    
    user_ids = tf.constant(df['userId'].values,)
    movie_ids = tf.constant(df['movieId'].values)
    ratings = tf.constant(df['rating'].values)
    genres = [tf.constant(genres, dtype='int64') for genres in genres_list]
    years = tf.constant(df['year'].values)
    avg_ratings =  tf.constant(df['averageRating'].values)
    num_votes =  tf.constant(df['numVotes'].values)
    directors =  tf.constant(df['director'].values)
    cast = [tf.constant(cast, dtype='int64') for cast in cast_list]
    
    dataset = tf.data.Dataset.from_tensor_slices({
        'movieId': movie_ids,
        'userId': user_ids,
        'rating': ratings,
        'genres': genres,
        'year': years,
        'averageRating': avg_ratings,
        'numVotes': num_votes,
        'director': directors,
        'cast':cast
    })
    return dataset