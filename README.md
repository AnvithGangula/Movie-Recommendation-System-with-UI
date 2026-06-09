# Movie Recommendation System

A Flask-based movie recommendation web app using a 10,000-movie dataset.

## Algorithms
- **Content-Based Filtering** – TF-IDF on movie overviews + cosine similarity
- **Popularity-Based** – IMDB-style Bayesian weighted rating
- **Hybrid** – 60% content + 40% popularity blend

## Setup
```bash
pip install flask pandas scikit-learn numpy
python app.py
```
Open http://localhost:5000

## Dataset
Place `Movies.csv` in the project root (same directory as `app.py`).

## Repository
[GitHub link — add your repo URL here]
