import pickle
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / 'models'


def test_idx_to_movie_mapping():
    """Verify that movie_id_map can be inverted to map idx -> movieId and ids exist in movies.csv"""
    mappings_path = MODELS / 'dl_mappings.pkl'
    movies_path = MODELS / 'movies.csv'
    assert mappings_path.exists(), f"Missing mappings file: {mappings_path}"
    assert movies_path.exists(), f"Missing movies file: {movies_path}"

    mappings = pickle.load(open(mappings_path, 'rb'))
    movie_id_map = mappings['movie_id_map']
    movie_ids = mappings['movie_ids']

    # Invert mapping
    idx_to_movie = {v: k for k, v in movie_id_map.items()}

    movies = pd.read_csv(movies_path)
    available_ids = set(movies['movieId'].astype(int).tolist())

    # Pick a handful of indices from movie_ids range
    for sample_idx in [0, min(10, len(movie_ids)-1), len(movie_ids)-1]:
        mid = idx_to_movie.get(sample_idx)
        assert mid is not None, f"Index {sample_idx} not found in inverted mapping"
        assert int(mid) in available_ids, f"MovieId {mid} (from idx {sample_idx}) not present in movies.csv"


if __name__ == '__main__':
    test_idx_to_movie_mapping()
    print('DL mapping test passed')
