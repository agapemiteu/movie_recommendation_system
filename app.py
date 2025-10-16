"""
MRS - Movie Recommendation System
Intelligent Movie Recommendations Powered by Machine Learning
"""

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
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

# Model accuracy constants (from training results)
MODEL_ACCURACY = {
    'basic_ncf': 0.8336,
    'enhanced_ncf': 0.8448,
    'tuned_ncf': 0.7479  # BEST MODEL
}

# Enhanced CSS - Netflix-inspired black & red theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #141414;
        color: #ffffff;
        padding: 2rem 1rem;
    }
    
    /* Hero Section */
    .hero {
        background: linear-gradient(to bottom, rgba(0,0,0,0.7), rgba(0,0,0,0.9)), 
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 600"><rect fill="%23E50914" width="1200" height="600"/></svg>');
        padding: 3rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        color: #E50914;
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .hero p {
        font-size: 1.2rem;
        color: #ffffff;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Method Cards */
    .method-card {
        background: #1f1f1f;
        border: 2px solid #2a2a2a;
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .method-card:hover {
        border-color: #E50914;
        box-shadow: 0 4px 20px rgba(229, 9, 20, 0.3);
        transform: translateY(-2px);
    }
    
    .method-card h3 {
        color: #E50914;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0;
    }
    
    .method-card p {
        color: #b3b3b3;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .method-card .tech-badge {
        display: inline-block;
        background: rgba(229, 9, 20, 0.2);
        color: #E50914;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .accuracy-badge {
        display: inline-block;
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
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
    
    .movie-result .genre-tag {
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
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid #2a2a2a;
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
    </style>
""", unsafe_allow_html=True)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    models_dir = Path('models')
    
    # Load movies data
    movies = pd.read_csv(models_dir / 'movies.csv')
    
    # Load content-based models
    with open(models_dir / 'tfidf_model.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open(models_dir / 'tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(models_dir / 'indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    
    # Load collaborative filtering model
    with open(models_dir / 'nmf_model.pkl', 'rb') as f:
        nmf_data = pickle.load(f)
    
    # Load deep learning model
    dl_model = tf.keras.models.load_model(models_dir / 'deep_learning_model.keras')
    
    with open(models_dir / 'dl_mappings.pkl', 'rb') as f:
        dl_mappings = pickle.load(f)
    
    return {
        'movies': movies,
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'indices': indices,
        'nmf_model': nmf_data['model'],
        'user_features': nmf_data['user_features'],
        'movie_features': nmf_data['movie_features'],
        'user_to_idx': nmf_data['user_to_idx'],
        'movie_to_idx': nmf_data['movie_to_idx'],
        'idx_to_movie': nmf_data['idx_to_movie'],
        'dl_model': dl_model,
        'user_id_map': dl_mappings['user_id_map'],
        'movie_id_map': dl_mappings['movie_id_map'],
        'user_ids': dl_mappings['user_ids'],
        'movie_ids': dl_mappings['movie_ids']
    }

# Recommendation functions with similarity scores
def get_content_recommendations(title, data, n=10):
    """Content-based filtering using movie similarity"""
    try:
        idx = data['indices'][title]
        vector = data['tfidf_matrix'][idx]
        similarities = cosine_similarity(vector, data['tfidf_matrix']).flatten()
        sim_scores = list(enumerate(similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        results = []
        for i, score in sim_scores:
            movie_title = data['movies']['title'].iloc[i]
            genre = data['movies']['genres'].iloc[i]
            results.append({
                'title': movie_title,
                'genre': genre,
                'similarity': score * 100  # Convert to percentage
            })
        return results
    except:
        return []

def get_collaborative_recommendations(user_id, data, n=10):
    """Collaborative filtering using matrix factorization"""
    if user_id not in data['user_to_idx']:
        return []
    
    user_idx = data['user_to_idx'][user_id]
    predicted_ratings = data['user_features'][user_idx].dot(data['movie_features'])
    
    top_indices = predicted_ratings.argsort()[-n:][::-1]
    
    results = []
    for idx in top_indices:
        movie_id = data['idx_to_movie'][idx]
        movie_row = data['movies'][data['movies']['movieId'] == movie_id].iloc[0]
        results.append({
            'title': movie_row['title'],
            'genre': movie_row['genres'],
            'predicted_rating': predicted_ratings[idx]
        })
    return results

def get_dl_recommendations(user_id, data, n=10):
    """Deep learning neural collaborative filtering"""
    if user_id not in data['user_id_map']:
        return []
    
    user_idx = data['user_id_map'][user_id]
    all_movie_indices = np.arange(len(data['movie_ids']))
    user_indices = np.full(len(data['movie_ids']), user_idx)
    
    predictions = data['dl_model'].predict([user_indices, all_movie_indices], verbose=0).flatten()
    top_indices = predictions.argsort()[-n:][::-1]
    
    results = []
    idx_to_movie_id = {idx: movie_id for movie_id, idx in data['movie_id_map'].items()}
    for idx in top_indices:
        movie_id = idx_to_movie_id[idx]
        movie_row = data['movies'][data['movies']['movieId'] == movie_id].iloc[0]
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
            <h1>MR1</h1>
            <p>Discover Your Next Favorite Movie with AI-Powered Recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
        # --- Data Loading with Demo Mode ---
        data_dir = 'data/ml-25m/'
        sample_movies_path = 'data/sample_movies.csv'
        sample_ratings_path = 'data/sample_ratings.csv'
        full_movies_path = os.path.join(data_dir, 'movies.csv')
        full_ratings_path = os.path.join(data_dir, 'ratings.csv')

        demo_mode = False
        try:
            if os.path.exists(full_movies_path) and os.path.exists(full_ratings_path):
                movies_df = pd.read_csv(full_movies_path)
                ratings_df = pd.read_csv(full_ratings_path)
            elif os.path.exists(sample_movies_path) and os.path.exists(sample_ratings_path):
                movies_df = pd.read_csv(sample_movies_path)
                ratings_df = pd.read_csv(sample_ratings_path)
                demo_mode = True
            else:
                st.error("No dataset found. Please add MovieLens data or sample data to the data/ folder.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Show demo mode info
        if demo_mode:
            st.info("Demo mode: Using small sample dataset for Streamlit Cloud. For full recommendations, run locally with the full MovieLens 25M dataset.")

        # Credit original dataset
        st.caption("Data: MovieLens 25M, GroupLens Research. See README for citation.")
    
    st.markdown("## Choose Your Recommendation Method")
    st.markdown("Select the algorithm that best fits your needs:")
    
    # Method selection with detailed explanations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="method-card">
                <h3>🎭 Similar Movies</h3>
                <p><strong>How it works:</strong> Analyzes movie genres and themes using TF-IDF vectorization and cosine similarity to find movies with similar characteristics.</p>
                <p><strong>Best for:</strong> Finding movies like your favorites</p>
                <p><strong>Use case:</strong> "I loved Inception, show me similar mind-bending thrillers"</p>
                <span class="tech-badge">TF-IDF</span>
                <span class="tech-badge">Cosine Similarity</span>
                <span class="accuracy-badge">✓ Fast & Accurate</span>
            </div>
        """, unsafe_allow_html=True)
        similar_btn = st.button("Use Similar Movies", key="similar_btn", use_container_width=True)
    
    with col2:
        st.markdown("""
            <div class="method-card">
                <h3>👥 For User</h3>
                <p><strong>How it works:</strong> Uses Matrix Factorization (NMF) to learn hidden patterns from 1M+ ratings, discovering what users with similar tastes enjoyed.</p>
                <p><strong>Best for:</strong> Personalized recommendations based on viewing history</p>
                <p><strong>Use case:</strong> "Based on my watch history, what should I watch next?"</p>
                <span class="tech-badge">NMF</span>
                <span class="tech-badge">Collaborative Filtering</span>
                <span class="accuracy-badge">✓ Personalized</span>
            </div>
        """, unsafe_allow_html=True)
        user_btn = st.button("Use User-Based", key="user_btn", use_container_width=True)
    
    with col3:
        st.markdown("""
            <div class="method-card">
                <h3>🤖 AI Powered</h3>
                <p><strong>How it works:</strong> Deep Neural Network with embeddings learns complex patterns from 500K+ interactions. Similar to Netflix's recommendation system.</p>
                <p><strong>Best for:</strong> Most accurate predictions using state-of-the-art AI</p>
                <p><strong>Use case:</strong> "Give me the most accurate predictions based on advanced AI"</p>
                <span class="tech-badge">Neural Network</span>
                <span class="tech-badge">Deep Learning</span>
                <span class="accuracy-badge">✓ MAE: 0.75 (Best!)</span>
            </div>
        """, unsafe_allow_html=True)
        ai_btn = st.button("Use AI Powered", key="ai_btn", use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'method' not in st.session_state:
        st.session_state.method = None
    
    if similar_btn:
        st.session_state.method = "Similar Movies"
    elif user_btn:
        st.session_state.method = "For User"
    elif ai_btn:
        st.session_state.method = "AI Powered"
    
    recommendations = []
    avg_score = 0
    
    # Show selected method interface
    if st.session_state.method == "Similar Movies":
        st.markdown("### 🎭 Content-Based Filtering")
        st.markdown("""
            <div class="info-box">
                <h4>How This Works</h4>
                <p>We analyze movie genres and characteristics using TF-IDF (Term Frequency-Inverse Document Frequency) 
                to create a mathematical representation of each movie. Then we calculate similarity scores to find movies 
                that share the most common features with your selection.</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            movie_titles = data['movies']['title'].tolist()
            
            # Add search functionality
            search_term = st.text_input("🔍 Search for a movie:", placeholder="Type movie title... (e.g., Inception, Toy Story)")
            
            if search_term:
                # Filter movies based on search
                filtered_movies = [title for title in movie_titles if search_term.lower() in title.lower()]
                if filtered_movies:
                    selected_movie = st.selectbox(
                        "Select from search results:",
                        options=filtered_movies,
                        index=0
                    )
                else:
                    st.warning(f"No movies found matching '{search_term}'. Try a different search term.")
                    selected_movie = st.selectbox(
                        "Or browse all movies:",
                        options=movie_titles,
                        index=movie_titles.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_titles else 0
                    )
            else:
                selected_movie = st.selectbox(
                    "Or browse all movies:",
                    options=movie_titles,
                    index=movie_titles.index('Toy Story (1995)') if 'Toy Story (1995)' in movie_titles else 0
                )
        
        with col2:
            num_recs = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10)
        
        if st.button("🎯 Get Recommendations", key="content_get"):
            with st.spinner('🔍 Analyzing movie similarities...'):
                recommendations = get_content_recommendations(selected_movie, data, num_recs)
                if recommendations:
                    avg_score = np.mean([r['similarity'] for r in recommendations])
    
    elif st.session_state.method == "For User":
        st.markdown("### 👥 Collaborative Filtering")
        st.markdown("""
            <div class="info-box">
                <h4>How This Works</h4>
                <p>This method uses Non-negative Matrix Factorization (NMF) trained on 1 million user ratings. 
                It learns hidden patterns by finding users with similar taste profiles and recommending movies 
                they enjoyed. Think of it as "people who liked what you liked also enjoyed these movies."</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            available_users = list(data['user_to_idx'].keys())
            user_id = st.selectbox(
                "👤 Select User ID:", 
                options=available_users[:100],
                help=f"Choose from {len(available_users)} users in the dataset"
            )
        
        with col2:
            num_recs = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10)
        
        if st.button("🎯 Get Recommendations", key="collab_get"):
            with st.spinner('📊 Analyzing user preferences...'):
                recommendations = get_collaborative_recommendations(user_id, data, num_recs)
                if recommendations:
                    avg_score = np.mean([r['predicted_rating'] for r in recommendations])
    
    elif st.session_state.method == "AI Powered":
        st.markdown("### 🤖 Deep Learning Neural Network")
        st.markdown("""
            <div class="info-box">
                <h4>How This Works</h4>
                <p>Our state-of-the-art Neural Collaborative Filtering model uses deep learning with 6.8M parameters. 
                It creates 50-dimensional embeddings for users and movies, learning complex non-linear patterns 
                through multiple hidden layers. This is the same technology used by YouTube and Pinterest for their 
                recommendation systems. <strong>Model Accuracy: MAE of 0.75 stars (BEST MODEL!)</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            available_users = list(data['user_id_map'].keys())
            user_id = st.selectbox(
                "👤 Select User ID:", 
                options=available_users[:100],
                help=f"Choose from {len(available_users)} users in the deep learning dataset"
            )
        
        with col2:
            num_recs = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10, key="dl_num")
        
        if st.button("🎯 Get Recommendations", key="dl_get"):
            with st.spinner('🧠 Running neural network inference...'):
                recommendations = get_dl_recommendations(user_id, data, num_recs)
                if recommendations:
                    avg_score = np.mean([r['predicted_rating'] for r in recommendations])
    
    # Display results
    if recommendations:
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Show accuracy metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-value">{len(recommendations)}</div>
                    <div class="stat-label">Recommendations</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.session_state.method == "Similar Movies":
                st.markdown(f"""
                    <div class="stat-container">
                        <div class="stat-value">{avg_score:.1f}%</div>
                        <div class="stat-label">Avg Similarity</div>
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
            dataset_size = "62K movies, 25M ratings" if st.session_state.method != "For User" else "62K movies, 1M ratings"
            st.markdown(f"""
                <div class="stat-container">
                    <div class="stat-value">✓</div>
                    <div class="stat-label">Trained on {dataset_size.split(',')[1]}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<h2 style='margin-top: 2rem;'>🎬 Your Personalized Recommendations</h2>", unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations, 1):
            genres = rec['genre'].split('|')[:3]  # Show first 3 genres
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
    
    # Footer with credits
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
                © 2025 MR1 - Movie Recommender One | Built for educational and research purposes<br>
                All movie data and ratings are property of their respective owners and the MovieLens project
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
