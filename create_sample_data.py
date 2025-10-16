"""
Create small sample datasets for Streamlit Cloud demo.
Run this script once locally to generate sample_movies.csv and sample_ratings.csv.
"""
import pandas as pd
import os

# Paths to full datasets
movies_path = 'data/ml-25m/movies.csv'
ratings_path = 'data/ml-25m/ratings.csv'

# Output paths
sample_movies_path = 'data/sample_movies.csv'
sample_ratings_path = 'data/sample_ratings.csv'

# Number of rows for sample
N_MOVIES = 500
N_RATINGS = 1000

# Create sample movies
if os.path.exists(movies_path):
    movies_df = pd.read_csv(movies_path)
    sample_movies = movies_df.sample(n=min(N_MOVIES, len(movies_df)), random_state=42)
    sample_movies.to_csv(sample_movies_path, index=False)
    print(f"✓ Saved {len(sample_movies)} movies to {sample_movies_path}")
else:
    print(f"File not found: {movies_path}")

# Create sample ratings
if os.path.exists(ratings_path):
    ratings_df = pd.read_csv(ratings_path)
    sample_ratings = ratings_df.sample(n=min(N_RATINGS, len(ratings_df)), random_state=42)
    sample_ratings.to_csv(sample_ratings_path, index=False)
    print(f"✓ Saved {len(sample_ratings)} ratings to {sample_ratings_path}")
else:
    print(f"File not found: {ratings_path}")
