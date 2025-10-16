
# 🎬 MRS - Movie Recommendation System

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A state-of-the-art movie recommendation system powered by multiple machine learning algorithms including Neural Collaborative Filtering, Content-Based Filtering, and Hybrid approaches. Built with TensorFlow and Streamlit for an elegant Netflix-inspired user experience.

---

## 📦 Data Source

This project uses the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/) provided by GroupLens Research.

- **Official Download:** https://grouplens.org/datasets/movielens/25m/
- **License:** For non-commercial, research use only. See [GroupLens Terms of Use](https://grouplens.org/datasets/movielens/).

**Citation:**
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your machine:

### 1. Clone the repository
```bash
git clone https://github.com/agapemiteu/movie_recommendation_system.git
cd movie_recommendation_system
```

### 2. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 3. Download the MovieLens 25M dataset
- Go to the [official MovieLens 25M page](https://grouplens.org/datasets/movielens/25m/)
- Download and unzip `ml-25m.zip`
- Place the extracted `ml-25m` folder inside the `data/` directory of this project:
   - `data/ml-25m/movies.csv`
   - `data/ml-25m/ratings.csv`
   - ... (other files)

### 4. Run the analysis or app
- **Jupyter Notebook:**
   - Open and run `notebooks/phase_1_content_based_filtering.ipynb` to train models and explore the pipeline.
- **Streamlit App:**
   - Run `streamlit run app.py` to launch the web interface.

---

## ✨ Features

- 🤖 **AI-Powered Recommendations** - Neural Collaborative Filtering with MAE 0.75
- 🎯 **Content-Based Filtering** - TF-IDF vectorization with cosine similarity
- 👥 **Collaborative Filtering** - Matrix factorization using NMF
- 🔄 **Hybrid Approach** - Best of both worlds combining multiple algorithms
- 🎨 **Netflix-Style UI** - Professional dark theme with interactive visualizations
- 📊 **Performance Metrics** - Real-time accuracy and prediction confidence
- 🔍 **Smart Search** - Find movies instantly from 62,000+ titles

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/movie_recommendation_system.git
cd movie_recommendation_system

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download the MovieLens 25M dataset from [GroupLens](https://grouplens.org/datasets/movielens/25m/)
2. Extract to `data/ml-25m/` directory
3. Ensure these files exist:
   - `data/ml-25m/movies.csv`
   - `data/ml-25m/ratings.csv`
   - `data/ml-25m/genome-scores.csv`
   - `data/ml-25m/genome-tags.csv`

### Train Models

```bash
# Run the Jupyter notebook to train all models
jupyter notebook notebooks/phase_1_content_based_filtering.ipynb

# Or use the save_models.py script
python save_models.py
```

### Launch the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser!

## � Live Demo (Streamlit)

Once deployed, add your app URL here. Example badge to add to README after deployment:

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app-url>.streamlit.app)

## �📊 Model Performance

| Model | Test MAE | Test RMSE | Features |
|-------|----------|-----------|----------|
| **Tuned NCF** ⭐ | **0.7479** | 0.95 | Optimized embeddings (32-dim, lr=0.001) |
| Enhanced NCF | 0.8448 | 1.07 | User + Movie + Genre features |
| Basic NCF | 0.8336 | 1.06 | User + Movie IDs only |

**Best Model:** Tuned Neural Collaborative Filtering achieves **MAE 0.75** - predictions are within 0.75 stars on average!

## 🏗️ Architecture

```
movie_recommendation_system/
├── app.py                      # Streamlit web application
├── requirements.txt            # Production dependencies
├── save_models.py             # Model training script
│
├── data/
│   └── ml-25m/                # MovieLens 25M dataset
│       ├── movies.csv         # 62,423 movies
│       ├── ratings.csv        # 25 million ratings
│       └── ...
│
├── models/                     # Trained models (saved after training)
│   ├── deep_learning_model.keras    # Neural Collaborative Filtering
│   ├── tfidf_matrix.pkl            # Content-based TF-IDF
│   ├── nmf_model.pkl               # Collaborative filtering
│   └── dl_mappings.pkl             # User/Movie ID mappings
│
├── notebooks/
│   └── phase_1_content_based_filtering.ipynb  # Training notebook
│
└── src/                        # Source modules (optional refactor)
    ├── data_loader.py
    ├── models.py
    ├── recommender.py
    └── train.py
```

## 🎯 How It Works

### 1. Content-Based Filtering
Uses TF-IDF vectorization on movie genres to find similar movies based on content features.

**Algorithm:** Cosine similarity between TF-IDF vectors
```python
similarity = cosine_similarity(movie_vector, all_movies_vectors)
```

### 2. Collaborative Filtering
Matrix factorization (NMF) learns latent features from user rating patterns.

**Algorithm:** Non-negative Matrix Factorization
```python
user_item_matrix ≈ user_features × movie_features
```

### 3. Neural Collaborative Filtering (AI)
Deep learning model with embeddings for users and movies, trained on 25M ratings.

**Architecture:**
- User embedding layer (50-dim) + Movie embedding layer (50-dim)
- Deep neural network: 128 → 64 → 32 neurons
- Dropout regularization (0.3-0.4)
- Optimized with Adam optimizer

### 4. Hybrid Approach
Combines content-based and collaborative filtering with weighted scoring:
- Content-based weight: 0.5
- Collaborative weight: 1.0

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: MR1 Movie Recommender"
   git remote add origin https://github.com/yourusername/movie_recommendation_system.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file: `app.py`
   - Click "Deploy"!

3. **Note:** Due to model size, you may need to:
   - Use Git LFS for large files (`models/*.keras`)
   - Or retrain models on Streamlit Cloud using `save_models.py`

### Deploy to Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Create runtime.txt
echo "python-3.13.0" > runtime.txt

# Deploy
heroku create your-app-name
git push heroku main
```

### Deploy to AWS/GCP

Use Docker for containerized deployment:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## 🛠️ Technologies Used

- **Python 3.13** - Latest Python with performance improvements
- **TensorFlow 2.20** - Deep learning framework
- **Keras** - High-level neural network API
- **Streamlit 1.40** - Web framework for ML apps
- **scikit-learn 1.5** - Traditional ML algorithms
- **Pandas 2.2** - Data manipulation
- **NumPy 2.0** - Numerical computing
- **Plotly 5.24** - Interactive visualizations

## 📖 Dataset

This project uses the **MovieLens 25M Dataset** provided by GroupLens Research.

- **Movies:** 62,423 titles
- **Ratings:** 25 million ratings from 162,541 users
- **Genres:** 20 unique genres
- **Time Period:** January 1995 - November 2019

**Citation:**
```
F. Maxwell Harper and Joseph A. Konstan. 2015. 
The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
https://doi.org/10.1145/2827872
```

## 🎓 Academic References

1. **Neural Collaborative Filtering**
   - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. In WWW 2017.

2. **Matrix Factorization**
   - Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.

3. **Content-Based Filtering**
   - Lops, P., De Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender systems handbook.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GroupLens Research for the MovieLens dataset
- TensorFlow and Keras teams for excellent documentation
- Streamlit for making ML app deployment accessible

## 📧 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle) - your.email@example.com

Project Link: [https://github.com/yourusername/movie_recommendation_system](https://github.com/yourusername/movie_recommendation_system)

---

**⭐ If you found this project helpful, please consider giving it a star!**
