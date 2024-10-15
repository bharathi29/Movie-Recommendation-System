import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

# Load the dataset
data = pd.read_csv('users.csv')

# Function to cluster users based on their viewing habits
def cluster_users(data):
    user_data = data.groupby('UserID').agg({
        'Rating': 'mean',
        'WatchTime': 'sum'
    }).reset_index()

    # K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)  # Change to 2 clusters for better overlap
    user_data['Cluster'] = kmeans.fit_predict(user_data[['Rating', 'WatchTime']])
    return user_data

# Function to recommend movies based on user cluster
def recommend_movies(user_id, data, user_data):
    # Get the cluster of the user
    user_cluster = user_data[user_data['UserID'] == user_id]['Cluster'].values[0]
    
    # Get recommended movies from the same cluster excluding the current user
    recommended_movies = data[data['UserID'].isin(user_data[user_data['Cluster'] == user_cluster]['UserID']) & (data['UserID'] != user_id)]
    
    # Get unique recommended movies and their genres
    recommended_movies = recommended_movies[['MovieID', 'Genre']].drop_duplicates()
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, max_value=5, value=1)

# Perform clustering
user_data = cluster_users(data)

# Get recommendations
if st.button("Recommend Movies"):
    recommendations = recommend_movies(user_id, data, user_data)
    
    if not recommendations.empty:
        st.write("Recommended Movies for User ID:", user_id)
        st.write(recommendations)
    else:
        st.write("No recommendations available.")
