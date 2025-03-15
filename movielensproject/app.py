from flask import Flask, render_template, request
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Load and prepare the data
ratings_df = pd.read_csv('Resources/ratings.csv')  # replace with your actual path
movies_df = pd.read_csv('Resources/movies.csv')  # replace with your actual path

# Prepare the ratings data for Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD model (you can also use your hybrid model here)
svd = SVD()
svd.fit(trainset)

# Function to get movie recommendations
def get_recommendations(user_id, movie_id, top_n=10):
    # CF-based recommendation (SVD)
    cf_prediction = svd.predict(user_id, movie_id).est

    # Content-based recommendation (we can use genre-based here)
    movie_genre = movies_df[movies_df['movieId'] == movie_id]['genres'].values[0]
    recommended_movies = movies_df[movies_df['genres'].str.contains(movie_genre)].head(top_n)
    
    hybrid_recommendations = []
    for _, movie in recommended_movies.iterrows():
        cf_movie_rating = svd.predict(user_id, movie['movieId']).est
        hybrid_rating = 0.5 * cf_movie_rating + 0.5 * cf_prediction  # Hybrid weighting
        hybrid_recommendations.append((movie['title'], hybrid_rating))
    
    hybrid_recommendations.sort(key=lambda x: x[1], reverse=True)
    return hybrid_recommendations[:top_n]

@app.route('/')
def home():
    # Pass the movies dataframe to the template
    return render_template('index.html', movies_df=movies_df)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    movie_id = int(request.form['movie_id'])
    recommendations = get_recommendations(user_id, movie_id)
    
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True,port=5500)
