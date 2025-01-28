
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

data = pd.read_csv(r'C:\Users\Hp\Desktop\dataset\Spotify_project\data.csv')
def find_song_from_data(song_name, data):
    row = data[data['name'].str.lower() == song_name.lower()]
    if row.empty:
        return None
    song_details = row.iloc[0].to_dict()
    return song_details
    
def get_mean_vector_song_name(song_names, data):
    # Filter the dataset to include only the songs with the given names
    filtered_data = data[data['name'].isin(song_names)]
    
    # List of numerical features
    numerical_features = [
        'duration_ms', 'popularity', 'danceability', 'energy',
        'instrumentalness', 'explicit', 'liveness', 'mode',
        'key', 'loudness', 'speechiness', 'acousticness', 'valence', 'tempo'
    ]
    # Calculate the mean vector of numerical features
    mean_vector_song_name = filtered_data[numerical_features].mean()
    return mean_vector_song_name


def recommend_songs(song_names, data, top_n):
    try:
        if not song_names:
            raise ValueError("The list of song names is empty.")
        if top_n <= 0:
            raise ValueError("The number of recommendations (top_n) must be greater than zero.")
        
        # Calculate the mean vector for the given song names
        mean_vector_song = get_mean_vector_song_name(song_names, data)
        
        if mean_vector_song is None or len(mean_vector_song) == 0:
            raise ValueError("The mean vector for the given songs could not be calculated.")
        
        # Define the numerical features to be used in the recommendation process
        numerical_features = [
            'duration_ms', 'popularity', 'danceability', 'energy', 
            'instrumentalness', 'explicit', 'liveness', 'mode', 
            'key', 'loudness', 'speechiness', 'acousticness', 
            'valence', 'tempo'
        ]
        
        # Extract the vectors for all songs based on the numerical features
        data_encoded_vectors = data[numerical_features]
        
        # Calculate the cosine similarity between the mean vector and each song vector
        similarities = data_encoded_vectors.apply(lambda row: 1 - cosine(mean_vector_song, row), axis=1)
        
        # Get the top_n recommendations, excluding the original songs
        recommendations = data.iloc[similarities.nlargest(top_n + len(song_names)).index]
        recommendations = recommendations[~recommendations['name'].isin(song_names)]
        
        # Select and return relevant information of the recommended songs
        display_data = recommendations[['artists', 'name', 'year']]
        return display_data.head(top_n)

    except KeyError as ke:
        print(f"KeyError: The column {ke} does not exist in the provided data.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Streamlit app
def main():
    st.title("Song Recommendation System")

    st.sidebar.header("User Input")
    song_name = st.sidebar.text_input("Enter the name of the song:")
    if st.sidebar.button("Find Song"):
        song_details = find_song_from_data(song_name, data)
        if song_details:
            st.write("### Song Details")
            st.write(st.dataframe(song_details, selection_mode='multi-column'))
        else:
            st.error("Song not found in the dataset.")

    top_n = st.sidebar.number_input("Enter the number of recommendations:", min_value=1, value=5)
    song_names = st.sidebar.text_area("Enter song names (comma-separated):")
    if st.sidebar.button("Recommend Songs"):
        song_names_list = [name.strip() for name in song_names.split(",")]
        recommendations = recommend_songs(song_names_list, data, top_n)
        if recommendations is not None and not recommendations.empty:
            st.write("### Recommendations")
            st.write(recommendations)
        else:
            st.error("No recommendations could be generated.")

if __name__ == "__main__":
    main()