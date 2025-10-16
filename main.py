# main.py - FastAPI Movie Recommender API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommender API",
    description="Deep Learning Movie Recommendation System",
    version="1.0.0"
)

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('models/movie_recommender_model')

# Load ID mappings
with open('models/id_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

user_id_map = mappings['user_id_map']
movie_id_map = mappings['movie_id_map']
user_ids_dl = mappings['user_ids_dl']
movie_ids_dl = mappings['movie_ids_dl']

# Load movies data
movies_df = pd.read_csv('models/movies_data.csv')

print("✓ Model and data loaded successfully!")

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to the Movie Recommender API",
        "endpoints": {
            "/recommend": "POST - Get movie recommendations for a user",
            "/health": "GET - Check API health",
            "/stats": "GET - Get system statistics"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/stats")
def get_stats():
    """Get system statistics"""
    return {
        "total_users": len(user_ids_dl),
        "total_movies": len(movie_ids_dl),
        "total_movies_in_catalog": len(movies_df)
    }

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a user
    
    Parameters:
    - user_id: The ID of the user
    - n_recommendations: Number of recommendations to return (default: 10)
    
    Returns:
    - List of recommended movies with predicted ratings
    """
    user_id = request.user_id
    n = request.n_recommendations
    
    # Validate user exists
    if user_id not in user_id_map:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {user_id} not found in the system"
        )
    
    # Get user index
    user_idx = user_id_map[user_id]
    
    # Prepare data for prediction
    all_movie_indices = np.arange(len(movie_ids_dl))
    user_indices = np.full(len(movie_ids_dl), user_idx)
    
    # Predict ratings for all movies
    predictions = model.predict(
        [user_indices, all_movie_indices],
        verbose=0
    ).flatten()
    
    # Get top N movies
    top_indices = predictions.argsort()[-n:][::-1]
    
    # Map back to movie IDs and get details
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_map.items()}
    
    recommendations = []
    for idx in top_indices:
        movie_id = idx_to_movie_id[idx]
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        
        if len(movie_info) > 0:
            recommendations.append(MovieRecommendation(
                movie_id=int(movie_id),
                title=movie_info['title'].values[0],
                predicted_rating=float(predictions[idx])
            ))
    
    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommendations
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
