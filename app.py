"""
Flask web application for the Movie Recommendation System.
Dataset: Movies.csv (10,000 movies)
"""

from flask import Flask, render_template, request, jsonify
from recommender import (
    content_rec, collab_rec, hybrid_rec,
    ALL_TITLES, ALL_USER_IDS
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", titles=ALL_TITLES, user_ids=ALL_USER_IDS)


@app.route("/api/content", methods=["POST"])
def content_recommend():
    data = request.get_json()
    title = data.get("title", "")
    n = int(data.get("n", 5))
    results = content_rec.recommend(title, n)
    return jsonify({"method": "Content-Based Filtering", "results": results})


@app.route("/api/collaborative", methods=["POST"])
def collab_recommend():
    data = request.get_json()
    user_id = int(data.get("user_id", 1))
    n = int(data.get("n", 5))
    results = collab_rec.recommend(user_id, n)
    return jsonify({"method": "Popularity-Based Recommendations", "results": results})


@app.route("/api/hybrid", methods=["POST"])
def hybrid_recommend():
    data = request.get_json()
    user_id = int(data.get("user_id", 1))
    title = data.get("title", "")
    n = int(data.get("n", 5))
    results = hybrid_rec.recommend(user_id, title, n)
    return jsonify({"method": "Hybrid Filtering", "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
