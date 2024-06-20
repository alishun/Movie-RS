import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import copy

def _get_vocab(df):
    feature_names = ['movieId','userId','year',
                 'averageRating','numVotes','director', 'cast']

    vocabularies = {}
    vocabularies["genres"] = np.arange(22)
    for feature in feature_names:
        # Check if the feature is a list
        if isinstance(df[feature].iloc[0], list):
            # Flatten the list of lists and extract unique values
            vocabularies[feature] = np.unique([item for sublist in df[feature].values for item in sublist])
        else:
            # If the feature is not a list, directly extract unique values
            vocabularies[feature] = np.unique(df[feature].values)
    vocabularies["cast"] = vocabularies["cast"][1:]

    return vocabularies

class DCNParallel(tfrs.Model):

    def __init__(self, df, deep_layer_sizes, projection_dim=None):
        super().__init__()
        self.df = df
    
        self.embedding_dimension = 32
    
        self._cat_features = ["movieId", "userId","director","genres","cast"]
        self._den_features = ["numVotes","averageRating","year"]
    
        self._embeddings = {}
        self._den_layers = {}

        vocabularies = _get_vocab(df)
          
        # Compute embeddings for categorical features.
        for feature_name in self._cat_features:
          vocabulary = vocabularies[feature_name]
          self._embeddings[feature_name] = tf.keras.Sequential(
              [tf.keras.layers.IntegerLookup(
                  vocabulary=vocabulary, mask_token=-1),
               tf.keras.layers.Embedding(len(vocabulary) + 2,
                                         self.embedding_dimension)])
        
        # Compute embeddings for continuous features
        for feature_name in self._den_features:
            self._den_layers[feature_name] = tf.keras.layers.Normalization(
                axis=None)

        # Init cross layers
        self._cross_layer = tfrs.layers.dcn.Cross(
          projection_dim=projection_dim,
          kernel_initializer="glorot_uniform")
        
        # Init deep layers
        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
          for layer_size in deep_layer_sizes]
    
        self._logit_layer = tf.keras.layers.Dense(1)
    
        self.task = tfrs.tasks.Ranking(
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )
    def adapt(self):
        # Normalize inputs
        for feature, layer in self._den_layers.items():
            layer.adapt(self.df[feature].values)

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._cat_features:
            embedding_fn = self._embeddings[feature_name]
            embedding = embedding_fn(features[feature_name])
            if feature_name == "genres" or feature_name == "cast":
                embedding = tf.reduce_mean(embedding, axis=1)
            embeddings.append(embedding)

        for feature_name in self._den_features:
            embedding_fn = self._den_layers[feature_name]
            embedding = tf.reshape(embedding_fn(features[feature_name]), (-1, 1))
            embeddings.append(embedding)

        x = tf.concat(embeddings, axis=1)

        # Process through Cross Network
        cross_output = self._cross_layer(x)
    
        # Process through Deep Network
        deep_output = x
        for deep_layer in self._deep_layers:
            deep_output = deep_layer(deep_output)
    
        # Combine the outputs
        final_output = tf.concat([cross_output, deep_output], axis=1)
        
        return self._logit_layer(final_output)

    def compute_loss(self, features, training=False):
        features_copy = copy.copy(features)  # Create a shallow copy of features
        labels = features_copy.pop("rating")  # Modify the copy, not the original
        scores = self(features_copy)  # Ensure the remaining code uses the modified copy if necessary
        return self.task(
            labels=labels,
            predictions=scores,
        )
    

class DCN(tfrs.Model):

    def __init__(self, df, deep_layer_sizes, projection_dim=None):
        super().__init__()
        self.df = df
    
        self.embedding_dimension = 32
    
        self._cat_features = ["movieId", "userId","director","genres","cast"]
        self._den_features = ["numVotes","averageRating","year"]
    
        self._embeddings = {}
        self._den_layers = {}

        vocabularies = _get_vocab(df)
          
        # Compute embeddings for categorical features.
        for feature_name in self._cat_features:
          vocabulary = vocabularies[feature_name]
          self._embeddings[feature_name] = tf.keras.Sequential(
              [tf.keras.layers.IntegerLookup(
                  vocabulary=vocabulary, mask_token=-1),
               tf.keras.layers.Embedding(len(vocabulary) + 2,
                                         self.embedding_dimension)])
            
        # Compute embeddings for continuous features
        for feature_name in self._den_features:
            self._den_layers[feature_name] = tf.keras.layers.Normalization(
                axis=None)

        # Init cross layers
        self._cross_layer = tfrs.layers.dcn.Cross(
          projection_dim=projection_dim,
          kernel_initializer="glorot_uniform")
        
        # Init deep layers
        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
          for layer_size in deep_layer_sizes]
        self.dropout = tf.keras.layers.Dropout(0.5)
    
        self._logit_layer = tf.keras.layers.Dense(1)
    
        self.task = tfrs.tasks.Ranking(
          loss=tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )
    def adapt(self):
        # Normalize continuous inputs
        for feature, layer in self._den_layers.items():
            layer.adapt(self.df[feature].values)

    def call(self, features):
        # Concatenate embeddings
        embeddings = []
        for feature_name in self._cat_features:
            embedding_fn = self._embeddings[feature_name]
            embedding = embedding_fn(features[feature_name])
            if feature_name == "genres" or feature_name == "cast":
                embedding = tf.reduce_mean(embedding, axis=1)
            embeddings.append(embedding)

        for feature_name in self._den_features:
            embedding_fn = self._den_layers[feature_name]
            embedding = tf.reshape(embedding_fn(features[feature_name]), (-1, 1))
            embeddings.append(embedding)

        x = tf.concat(embeddings, axis=1)
    
        # Build Cross Network
        x = self._cross_layer(x)
        
        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)
            x = self.dropout(x)
        return self._logit_layer(x)

    def compute_loss(self, features, training=False):
        features_copy = copy.copy(features)  # Create a shallow copy of features
        labels = features_copy.pop("rating")  # Modify the copy, not the original
        scores = self(features_copy)  # Ensure the remaining code uses the modified copy if necessary
        return self.task(
            labels=labels,
            predictions=scores,
        )