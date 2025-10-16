"""
Quick smoke tests for the three recommenders.
Run with: .venv/Scripts/python.exe test_recs.py
"""
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

models_dir = Path('models')

def test_content():
    print('\n--- Content-Based Test ---')
    movies = pd.read_csv(models_dir / 'movies.csv')
    with open(models_dir / 'tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(models_dir / 'indices.pkl', 'rb') as f:
        indices = pickle.load(f)

    # pick a movie
    title = 'Toy Story (1995)'
    if title not in indices.index:
        title = indices.index[0]

    idx = indices[title]
    vector = tfidf_matrix[idx]
    sims = cosine_similarity(vector, tfidf_matrix).flatten()
    top = np.argsort(sims)[-6:][::-1]
    for i in top[1:6]:
        print('-', movies['title'].iloc[i])

def test_collab():
    print('\n--- Collaborative (NMF) Test ---')
    with open(models_dir / 'nmf_model.pkl', 'rb') as f:
        nmf = pickle.load(f)
    user_to_idx = nmf['user_to_idx']
    user_features = nmf['user_features']
    movie_features = nmf['movie_features']
    idx_to_movie = nmf['idx_to_movie']
    movies = pd.read_csv(models_dir / 'movies.csv')

    # pick a user
    some_user = list(user_to_idx.keys())[0]
    uidx = user_to_idx[some_user]
    preds = user_features[uidx].dot(movie_features)
    top = np.argsort(preds)[-10:][::-1]
    for idx in top[:5]:
        mid = idx_to_movie[idx]
        title = movies[movies['movieId'] == mid]['title'].iloc[0]
        print('-', title)

def test_dl():
    print('\n--- Deep Learning (DL) Test ---')
    try:
        dl = tf.keras.models.load_model(models_dir / 'deep_learning_model.keras')
        with open(models_dir / 'dl_mappings.pkl', 'rb') as f:
            dl_map = pickle.load(f)
        user_id_map = dl_map['user_id_map']
        movie_id_map = dl_map['movie_id_map']
        user_ids = dl_map['user_ids']
        movie_ids = dl_map['movie_ids']
        movies = pd.read_csv(models_dir / 'movies.csv')

        test_user = user_ids[0]
        uidx = user_id_map[test_user]
        all_movie_indices = np.arange(len(movie_ids))
        user_indices = np.full(len(movie_ids), uidx)
        preds = dl.predict([user_indices, all_movie_indices], verbose=0).flatten()
        top = preds.argsort()[-10:][::-1]
        # movie_id_map maps movieId -> idx, so invert it to map idx -> movieId
        idx_to_movie_id = {v: k for k, v in movie_id_map.items()}
        for idx in top[:5]:
            movie_id = idx_to_movie_id.get(idx)
            if movie_id is None:
                continue
            title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
            print('-', title)
    except Exception as e:
        print('DL test skipped (could not load model):', e)

if __name__ == '__main__':
    test_content()
    test_collab()
    test_dl()
