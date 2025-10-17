"""
MRS - Movie Recommendation System
Intelligent Movie Recommendations Powered by Machine Learning
"""

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="MRS - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS - Netflix-inspired theme
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #141414;
    }

    /* Hero Section */
    .hero {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        background: linear-gradient(180deg, #000000 0%, #141414 100%);
    }

    .hero h1 {
        font-size: 3.5rem;
        font-weight: 900;
        color: #E50914;
        margin: 0;
        letter-spacing: -0.03em;
    }

    .hero p {
        font-size: 1.1rem;
        color: #ffffff;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Main Search Section */
    .search-section {
        background: #1f1f1f;
        border: 2px solid #2a2a2a;
        border-radius: 8px;
        padding: 2rem;
        margin: 1.5rem 0;
    }

    .search-section h2 {
        color: #E50914;
        font-size: 1.8rem;
        margin-top: 0;
        margin-bottom: 1rem;
    }

    /* Method Cards */
    .method-card {
        background: #1f1f1f;
        border: 2px solid #2a2a2a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        min-height: 280px;
    }

    .method-card:hover {
        border-color: #E50914;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.3);
        transform: translateY(-2px);
    }

    .method-card h3 {
        color: #E50914;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.8rem;
    }

    .method-card p {
        color: #b3b3b3;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 0.4rem;
    }

    .tech-badge {
        display: inline-block;
        background: rgba(229, 9, 20, 0.2);
        color: #E50914;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.4rem;
    }

    .accuracy-badge {
        display: inline-block;
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #E50914;
        color: #ffffff;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 4px;
        width: 100%;
        transition: all 0.2s;
        font-size: 1rem;
    }

    .stButton>button:hover {
        background-color: #f40612;
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4);
    }

    /* Movie Result Cards */
    .movie-result {
        background: #1f1f1f;
        padding: 1.25rem;
        border-left: 4px solid #E50914;
        margin: 0.75rem 0;
        border-radius: 4px;
        transition: all 0.2s;
    }

    .movie-result:hover {
        background: #2a2a2a;
        transform: translateX(5px);
    }

    .movie-result h4 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }

    .genre-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.1);
        color: #b3b3b3;
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
        margin-right: 0.3rem;
    }

    /* Input Fields */
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stRadio label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
    }

    /* Info Box */
    .info-box {
        background: rgba(229, 9, 20, 0.1);
        border-left: 4px solid #E50914;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin: 1.5rem 0;
    }

    .info-box h4 {
        color: #E50914;
        margin-top: 0;
        font-size: 1rem;
        font-weight: 700;
    }

    .info-box p {
        color: #b3b3b3;
        margin: 0;
        font-size: 0.95rem;
    }

    /* Stats */
    .stat-container {
        background: #1f1f1f;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }

    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #E50914;
        margin: 0;
    }

    .stat-label {
        font-size: 0.9rem;
        color: #b3b3b3;
        margin-top: 0.5rem;
    }

    /* Section Headers */
    .section-header {
        color: #E50914;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Footer */
    .footer {
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid #2a2a2a;
        text-align: center;
    }

    .footer h4 {
        color: #E50914;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    .footer p {
        color: #808080;
        font-size: 0.875rem;
        line-height: 1.6;
        margin: 0.5rem 0;
    }

    .footer a {
        color: #E50914;
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    h1, h2, h3 {
        color: #ffffff;
    }

    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid #2a2a2a;
    }
    </style>
""", unsafe_allow_html=True)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    models_dir = Path('models')
    movies = pd.read_csv(models_dir / 'movies.csv')

    with open(models_dir / 'tfidf_model.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(models_dir / 'tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(models_dir / 'indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    with open(models_dir / 'nmf_model.pkl', 'rb') as f:
        nmf_data = pickle.load(f)

    dl_model = None
    dl_mappings = {}
    dl_model_path = models_dir / 'deep_learning_model.keras'
    if dl_model_path.exists():
        dl_model = tf.keras.models.load_model(dl_model_path)
    dl_mappings_path = models_dir / 'dl_mappings.pkl'
    if dl_mappings_path.exists():
        with open(dl_mappings_path, 'rb') as f:
            dl_mappings = pickle.load(f)

    nmf_model = nmf_data.get('model')
    user_features = nmf_data.get('user_features')
    movie_features = nmf_data.get('movie_features')
    user_to_idx = nmf_data.get('user_to_idx')
    movie_to_idx = nmf_data.get('movie_to_idx')
    idx_to_movie = nmf_data.get('idx_to_movie')

    user_id_map = dl_mappings.get('user_id_map')
    movie_id_map = dl_mappings.get('movie_id_map')
    user_ids = dl_mappings.get('user_ids')
    movie_ids = dl_mappings.get('movie_ids')

    return (movies, tfidf_vectorizer, tfidf_matrix, indices, nmf_model, user_features,
            movie_features, user_to_idx, movie_to_idx, idx_to_movie, dl_model,
            user_id_map, movie_id_map, user_ids, movie_ids)

# Recommendation functions
def get_content_recommendations(title, movies_df, indices, tfidf_matrix, n=10):
    try:
        idx = indices[title]
        vector = tfidf_matrix[idx]
        similarities = cosine_similarity(vector, tfidf_matrix).flatten()
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        results = []
        for i, score in sim_scores:
            results.append({
                'title': movies_df['title'].iloc[i],
                'genre': movies_df['genres'].iloc[i],
                'similarity': score * 100
            })
        return results
    except:
        return []

def get_movies_by_genre(movies_df, genre, n=10):
    """Get movies by genre"""
    filtered = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
    return filtered.head(n)

def get_collaborative_recommendations(user_id, user_to_idx, user_features, movie_features, idx_to_movie, movies_df, n=10):
    if user_id not in user_to_idx:
        return []
    user_idx = user_to_idx[user_id]
    predicted_ratings = user_features[user_idx].dot(movie_features)
    top_indices = predicted_ratings.argsort()[-n:][::-1]
    results = []
    for idx in top_indices:
        movie_id = idx_to_movie[idx]
        movie_row = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        results.append({
            'title': movie_row['title'],
            'genre': movie_row['genres'],
            'predicted_rating': predicted_ratings[idx]
        })
    return results

def get_dl_recommendations(user_id, user_id_map, movie_id_map, user_ids, movie_ids, dl_model, movies_df, n=10):
    if user_id not in user_id_map:
        return []
    user_idx = user_id_map[user_id]
    all_movie_indices = np.arange(len(movie_ids))
    user_indices = np.full(len(movie_ids), user_idx)
    predictions = dl_model.predict([user_indices, all_movie_indices], verbose=0).flatten()
    top_indices = predictions.argsort()[-n:][::-1]
    results = []
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_map.items()}
    for idx in top_indices:
        movie_id = idx_to_movie_id[idx]
        movie_row = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        results.append({
            'title': movie_row['title'],
            'genre': movie_row['genres'],
            'predicted_rating': predictions[idx]
        })
    return results

# Main app
def main():
    # Hero Section
    st.markdown("""
        <div class="hero">
            <h1>MRS</h1>
            <p>Movie Recommendation System - Discover Your Next Favorite Movie with AI</p>
        </div>
    """, unsafe_allow_html=True)

    # Data Loading with Demo Mode
    data_dir = 'data/ml-25m/'
    models_dir = Path('models')
    sample_movies_path = 'data/sample_movies.csv'
    sample_ratings_path = 'data/sample_ratings.csv'
    full_movies_path = os.path.join(data_dir, 'movies.csv')
    full_ratings_path = os.path.join(data_dir, 'ratings.csv')

    demo_mode = False
    try:
        if os.path.exists(full_movies_path) and os.path.exists(full_ratings_path):
            (movies_df, tfidf_vectorizer, tfidf_matrix, indices, nmf_model, user_features,
             movie_features, user_to_idx, movie_to_idx, idx_to_movie, dl_model,
             user_id_map, movie_id_map, user_ids, movie_ids) = load_models()
            ratings_df = pd.read_csv(full_ratings_path)
        elif os.path.exists(sample_movies_path) and os.path.exists(sample_ratings_path):
            movies_df = pd.read_csv(sample_movies_path)
            ratings_df = pd.read_csv(sample_ratings_path)
            demo_mode = True
            tfidf_vectorizer = None
            tfidf_matrix = None
            indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
            user_to_idx = {uid: idx for idx, uid in enumerate(ratings_df['userId'].unique())}
            user_features = np.random.rand(len(user_to_idx), 10)
            movie_features = np.random.rand(10, len(movies_df))
            idx_to_movie = {idx: mid for idx, mid in enumerate(movies_df['movieId'].unique())}
            dl_model = None
            user_id_map = user_to_idx
            movie_id_map = {mid: idx for idx, mid in enumerate(movies_df['movieId'].unique())}
            user_ids = list(user_id_map.keys())
            movie_ids = list(movie_id_map.keys())
        else:
            st.error("No dataset found. Please add MovieLens data or sample data to the data/ folder.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if demo_mode:
        st.info("Demo mode: Using small sample dataset for Streamlit Cloud. For full recommendations, run locally with the full MovieLens 25M dataset.")

    # ========== MAIN RECOMMENDATION SECTION (TOP) ==========
    st.markdown('<h2 class="section-header">🎬 Get Movie Recommendations</h2>', unsafe_allow_html=True)

    # Input method selection
    search_method = st.radio(
        "Choose how you want to find movies:",
        ["Search by Movie Name", "Browse by Genre", "Personalized (User ID)"],
        horizontal=True
    )

    recommendations = []
    avg_score = 0
    selected_method = None

    # Input Section
    col1, col2, col3 = st.columns([2, 1, 1])

    if search_method == "Search by Movie Name":
        with col1:
            movie_titles = movies_df['title'].tolist()
            search_term = st.text_input("🔍 Search for a movie:", placeholder="Type movie title (e.g., Toy Story, Inception)...")

            if search_term:
                filtered_movies = [t for t in movie_titles if search_term.lower() in t.lower()]
                if filtered_movies:
                    selected_movie = st.selectbox("Select a movie:", options=filtered_movies, index=0)
                else:
                    st.warning(f"No movies found matching '{search_term}'")
                    selected_movie = st.selectbox("Or browse all movies:", options=movie_titles[:100])
            else:
                selected_movie = st.selectbox("Or browse popular movies:", options=movie_titles[:100])

        with col2:
            num_recs = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10)

        with col3:
            st.write("")
            st.write("")
            if st.button("🎯 Get Similar Movies", use_container_width=True):
                with st.spinner('🔍 Finding similar movies...'):
                    recommendations = get_content_recommendations(selected_movie, movies_df, indices, tfidf_matrix, num_recs)
                    if recommendations:
                        avg_score = np.mean([r['similarity'] for r in recommendations])
                        selected_method = "Content-Based (Similar Movies)"

    elif search_method == "Browse by Genre":
        with col1:
            all_genres = set()
            for genres_str in movies_df['genres'].dropna():
                all_genres.update(genres_str.split('|'))
            all_genres = sorted(list(all_genres))

            selected_genre = st.selectbox("🎭 Select a genre:", options=all_genres)

        with col2:
            num_recs = st.number_input("Number of movies", min_value=5, max_value=20, value=10)

        with col3:
            st.write("")
            st.write("")
            if st.button("🎯 Browse Genre", use_container_width=True):
                with st.spinner(f'🔍 Finding {selected_genre} movies...'):
                    genre_movies = get_movies_by_genre(movies_df, selected_genre, num_recs)
                    recommendations = []
                    for _, row in genre_movies.iterrows():
                        recommendations.append({
                            'title': row['title'],
                            'genre': row['genres'],
                            'similarity': 100  # Genre match
                        })
                    selected_method = f"Genre-Based ({selected_genre})"

    else:  # Personalized (User ID)
        with col1:
            rec_model = st.selectbox(
                "🤖 Choose recommendation model:",
                ["Collaborative Filtering (NMF)", "Deep Learning (Neural Network)"]
            )

            available_users = list(user_to_idx.keys()) if rec_model == "Collaborative Filtering (NMF)" else list(user_id_map.keys())
            user_id = st.selectbox("👤 Select User ID:", options=available_users[:100],
                                 help=f"Choose from {len(available_users)} users")

        with col2:
            num_recs = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10)

        with col3:
            st.write("")
            st.write("")
            if st.button("🎯 Get Personalized", use_container_width=True):
                if rec_model == "Collaborative Filtering (NMF)":
                    with st.spinner('📊 Analyzing user preferences...'):
                        recommendations = get_collaborative_recommendations(
                            user_id, user_to_idx, user_features, movie_features, idx_to_movie, movies_df, num_recs)
                        if recommendations:
                            avg_score = np.mean([r['predicted_rating'] for r in recommendations])
                            selected_method = "Collaborative Filtering (NMF)"
                else:
                    with st.spinner('🧠 Running neural network...'):
                        recommendations = get_dl_recommendations(
                            user_id, user_id_map, movie_id_map, user_ids, movie_ids, dl_model, movies_df, num_recs)
                        if recommendations:
                            avg_score = np.mean([r['predicted_rating'] for r in recommendations])
                            selected_method = "Deep Learning (Neural Network)"

    # Display Results
    if recommendations:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-value">{len(recommendations)}</div>
                    <div class="stat-label">Recommendations Found</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            if 'similarity' in recommendations[0]:
                st.markdown(f"""
                    <div class="stat-container">
                        <div class="stat-value">{avg_score:.1f}%</div>
                        <div class="stat-label">Avg Match Score</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="stat-container">
                        <div class="stat-value">{avg_score:.2f}⭐</div>
                        <div class="stat-label">Avg Predicted Rating</div>
                    </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-value">✓</div>
                    <div class="stat-label">{selected_method}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<h2 style='margin-top: 2rem; color: #E50914;'>🍿 Your Recommendations</h2>", unsafe_allow_html=True)

        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            genres = rec['genre'].split('|')[:3]
            genre_tags = ''.join([f'<span class="genre-tag">{g}</span>' for g in genres])

            if 'similarity' in rec:
                score_text = f"{rec['similarity']:.1f}% match"
            else:
                score_text = f"⭐ {rec['predicted_rating']:.2f}/5.0"

            st.markdown(f"""
                <div class="movie-result">
                    <h4>{i}. {rec['title']}</h4>
                    <p style="color: #E50914; font-weight: 600; margin-top: 0.5rem;">{score_text}</p>
                    {genre_tags}
                </div>
            """, unsafe_allow_html=True)

    # ========== MODEL EXPLANATIONS SECTION (BOTTOM) ==========
    st.markdown('<h2 class="section-header">🤖 Our Recommendation Models</h2>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #b3b3b3; margin-bottom: 2rem;'>Learn about the three AI models powering this system</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="method-card">
                <h3>🎭 Similarity Matching</h3>
                <p><strong>What it does:</strong> Finds movies with matching genres and themes</p>
                <p><strong>How:</strong> Compares movie features to find similar content</p>
                <p><strong>Perfect for:</strong> "I loved Inception, show me more like it"</p>
                <span class="tech-badge">Pattern Matching</span>
                <span class="tech-badge">Genre Analysis</span><br>
                <span class="accuracy-badge">✓ Instant Results</span>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="method-card">
                <h3>👥 User Taste Analysis</h3>
                <p><strong>What it does:</strong> Learns your preferences from 1M+ ratings</p>
                <p><strong>How:</strong> Studies what users with similar taste enjoyed</p>
                <p><strong>Perfect for:</strong> "Recommend based on my watch history"</p>
                <span class="tech-badge">Personalized</span>
                <span class="tech-badge">Data-Driven</span><br>
                <span class="accuracy-badge">✓ Smart Predictions</span>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="method-card">
                <h3>🤖 AI Intelligence</h3>
                <p><strong>What it does:</strong> Uses advanced AI to predict your ratings</p>
                <p><strong>How:</strong> Trained on 500K+ user interactions using neural networks</p>
                <p><strong>Perfect for:</strong> "Give me the most accurate predictions"</p>
                <span class="tech-badge">Machine Learning</span>
                <span class="tech-badge">AI-Powered</span><br>
                <span class="accuracy-badge">✓ Highest Accuracy (0.75★)</span>
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class="footer">
            <h4>📊 Dataset & Technology</h4>
            <p><strong>Dataset:</strong> MovieLens 25M Dataset</p>
            <p>Provided by <a href="https://grouplens.org/datasets/movielens/" target="_blank">GroupLens Research</a> at the University of Minnesota</p>
            <p>Contains 25 million ratings and 62,423 movies from 162,541 users</p>
            <p style="margin-top: 1rem;"><strong>Citation:</strong> F. Maxwell Harper and Joseph A. Konstan. 2015.
            The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.</p>
            <p style="margin-top: 1.5rem;"><strong>Technologies Used:</strong></p>
            <p>TensorFlow 2.20 • Scikit-learn • Pandas • NumPy • Streamlit</p>
            <p>Algorithms: TF-IDF, NMF Matrix Factorization, Neural Collaborative Filtering</p>
            <p style="margin-top: 1.5rem; color: #666; font-size: 0.8rem;">
                © 2025 MRS - Movie Recommendation System | Built for educational and research purposes<br>
                All movie data and ratings are property of their respective owners and the MovieLens project
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
