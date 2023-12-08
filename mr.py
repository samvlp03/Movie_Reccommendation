import pandas as pd
import numpy as np

# Load the dataset 
file_path = 'ratings.csv'
file_path_1='movies.csv'

# Reading the dataset
ratings = pd.read_csv(file_path)
movies=pd.read_csv(file_path_1)

# Merge the dataset
merged_data=pd.merge(ratings,movies,on='item_id')

# Create a user-item matrix
user_item_matrix = merged_data.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

# Perform Singular Value Decomposition (SVD)
U, sigma, Vt = np.linalg.svd(user_item_matrix, full_matrices=False)

# Number of latent factors(adjustable)
k = 40

# Keep only the top k singular values and corresponding vectors
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]

# Reconstruct the user-item matrix with reduced dimensions
predicted_ratings = np.dot(np.dot(U_k, sigma_k), Vt_k)

# Example: Recommend movies for a user 
user_id = 5
user_movies = merged_data[merged_data['user_id'] == user_id]['item_id'].unique()

# Exclude movies the user has already rated
unrated_movies = merged_data[~merged_data['item_id'].isin(user_movies)]['item_id'].unique()

# Create a list of tuples (movieId, predicted_rating) for unrated movies
predictions_for_user = [(movie_id, predicted_ratings[user_id - 1, movie_id - 1]) for movie_id in unrated_movies]

# Sort the predictions in descending order of predicted rating
sorted_predictions = sorted(predictions_for_user, key=lambda x: x[1], reverse=True)

# Display the top N recommendations (replace N with the desired number)
top_recommendations = sorted_predictions[:10]
print(f'\nTop Recommendations for User {user_id}:')
for movie_id, predicted_rating in top_recommendations:
    movie_title = merged_data[merged_data['item_id'] == movie_id]['title'].iloc[0]
    print(f'Movie: {movie_title},\t Predicted Rating: {predicted_rating:.2f}')
