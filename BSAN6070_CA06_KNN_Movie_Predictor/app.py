import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained KNN model
loaded_model = pickle.load(open('movie_predictor.sav', 'rb'))

# Load the original DataFrame to get movie names (if needed)
df = pd.read_csv("movies_recommendation_data.csv")  # Adjust path if needed


feature_columns = ['Biography', 'Drama', 'Thriller', 'Comedy', 'Crime', 'Mystery', 'History']

st.title("🎬 Movie Recommender")


# User selects genres
selected_genres = st.multiselect(
    "Select Genres:", 
    feature_columns,
    default=["Drama"]  # Optional: Set a default genre
)

# User picks a reference movie
unique_movies = df['Movie Name'].unique()
selected_movie = st.selectbox(
    "Select a Reference Movie:", 
    unique_movies
)

#  Find similar movies
if st.button("Find Similar Movies"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        # Get the reference movie's IMDB rating
        ref_movie_data = df[df['Movie Name'] == selected_movie].iloc[0]
        imdb_rating = ref_movie_data['IMDB Rating']

        # Create a feature vector (IMDB + genres)
        features = [imdb_rating] + [1 if genre in selected_genres else 0 for genre in feature_columns]
        features_array = np.array(features).reshape(1, -1)

        # Find nearest neighbors
        distances, indices = loaded_model.kneighbors(features_array)

        # Display results
        st.subheader("🎥 Top 5 Similar Movies:")
        
        similar_movies = df.iloc[indices[0]][['Movie Name', 'IMDB Rating']]

        # Display as a bullet-point list in Streamlit
        for idx, row in df.iloc[indices[0]][['Movie Name', 'IMDB Rating']].iterrows():
            st.markdown(f"- **{row['Movie Name']}** (IMDB: {row['IMDB Rating']})")
        #st.markdown("---\n*End of recommendations*")
        #st.dataframe(similar_movies)

        # Optional: Show distances (for debugging)
        # st.write("Distances:", distances)

