# Movie Recommendation System

A machine learning system implementing three recommendation algorithms: Content-Based Filtering (TF-IDF), Collaborative Filtering (NMF), and Deep Learning Neural Networks. Built with Python, TensorFlow, and Streamlit for production deployment.

**Live Demo:** [Streamlit Cloud](https://movierecommendationsystem-dxe5m6b8ttvi937nyyynue.streamlit.app)

---

## Overview

This system demonstrates three distinct machine learning approaches to movie recommendations using the MovieLens 25M dataset (25M ratings, 62,423 movies, 162,541 users).

**Key Features:**
- Content-Based Filtering using TF-IDF and cosine similarity
- Collaborative Filtering using NMF matrix factorization
- Deep Learning using Neural Collaborative Filtering with 50-dimensional embeddings
- Web interface with search, genre browsing, and model selection
- Demo mode for cloud deployment

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/agapemiteu/movie_recommendation_system.git
cd movie_recommendation_system
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Visit [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)
- Download and extract ml-25m.zip
- Place in `data/ml-25m/` directory

### 5. Run Application
```bash
# Train models (optional)
python save_models.py

# Launch web app
streamlit run app.py
```

Visit `http://localhost:8501`

---

## Architecture

```
movie_recommendation_system/
├── app.py                              # Streamlit web application
├── save_models.py                      # Model training and saving
├── requirements.txt                    # Dependencies
│
├── data/
│   ├── ml-25m/                        # MovieLens 25M dataset
│   ├── sample_movies.csv              # Demo dataset
│   └── sample_ratings.csv             # Demo ratings
│
├── models/                             # Pre-trained models
│   ├── deep_learning_model.keras      # Neural Collaborative Filtering
│   ├── nmf_model.pkl                  # NMF model and features
│   ├── tfidf_model.pkl                # TF-IDF vectorizer
│   ├── tfidf_matrix.pkl               # Pre-computed TF-IDF matrix
│   ├── indices.pkl                    # Movie index mapping
│   └── dl_mappings.pkl                # User/movie ID mappings
│
├── notebooks/
│   └── phase_1_content_based_filtering.ipynb  # Development and training
│
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
│
└── tests/
    ├── test_recs.py                   # Recommendation tests
    └── test_dl_mappings.py            # Deep learning mapping tests
```

---

## Models

### 1. Content-Based Filtering
- **Algorithm:** TF-IDF + Cosine Similarity
- **Implementation:** Analyzes movie genres to find similar content
- **Performance:** Fast, deterministic results
- **Status:** Fully functional in demo mode

### 2. Collaborative Filtering
- **Algorithm:** Non-negative Matrix Factorization (NMF)
- **Components:** 20 latent factors
- **Training Data:** 142,046 users × 23,238 movies
- **Performance:** Personalized based on user rating patterns

### 3. Deep Learning Neural Network
- **Architecture:** Neural Collaborative Filtering
- **Embeddings:** 50-dimensional for users and movies
- **Layers:** Embedding → Dense(128) → Dense(64) → Dense(32) → Output
- **Regularization:** Dropout (0.3-0.4)
- **Performance:** MAE = 0.75 (best model)
- **Model Size:** 78.4 MB
- **Training Data:** 118,288 users × 18,205 movies

---

## Model Performance

| Model | MAE | RMSE | Dataset |
|-------|-----|------|---------|
| Deep Learning NCF | 0.75 | 0.95 | 500K+ interactions |
| Collaborative (NMF) | ~0.9 | ~1.1 | 1M ratings |
| Content-Based | N/A | N/A | Genre similarity |

---

## Usage

### Search by Movie Title
1. Select "Search by Movie Title"
2. Enter movie name or browse suggestions
3. Choose recommendation model:
   - Content-Based Filtering (fast)
   - Collaborative Filtering (personalized)
   - Deep Learning (most accurate)
4. Click "Get Recommendations"

### Browse by Genre
1. Select "Browse by Genre"
2. Choose genre from dropdown
3. Click "Show Movies"
4. Browse results with genre tags

---

## Technologies

- Python 3.13
- TensorFlow 2.20
- Keras
- Streamlit 1.40
- scikit-learn 1.5.2
- Pandas 2.2.3
- NumPy 2.1.0+
- Plotly 5.24.1

---

## Dataset

MovieLens 25M Dataset provided by GroupLens Research

- 62,423 movies
- 25 million ratings
- 162,541 users
- 23 genres
- Time period: January 1995 - November 2019

**Citation:**
```
F. Maxwell Harper and Joseph A. Konstan. 2015.
The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
https://doi.org/10.1145/2827872
```

---

## Deployment

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud
Repository is configured for Streamlit Cloud deployment. Push to GitHub and connect repository in Streamlit Cloud dashboard.

### Docker
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

---

## Development

### Training Models
Run notebook to train all models:
```bash
jupyter notebook notebooks/phase_1_content_based_filtering.ipynb
```

### Testing
```bash
python -u test_recs.py
pytest tests/test_dl_mappings.py
```

---

## License

MIT License

---

## Contact

Email: agapemiteu@gmail.com
