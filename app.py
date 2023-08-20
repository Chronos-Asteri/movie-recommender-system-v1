import pickle
import streamlit as st
import bz2


def recommend(movie):
    index = movies[movies['original_title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].original_title)

    return recommended_movie_names

movie_loc =  './similarity_and_movie_list_datasets/movie_list.pkl'
similarity_loc = './similarity_and_movie_list_datasets/similarity.pkl'

st.header('Movie Recommender System')
movies = pickle.load(bz2.BZ2File(movie_loc,'rb'))
similarity = pickle.load(bz2.BZ2File(similarity_loc,'rb'))

movie_list = movies['original_title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    for i in recommended_movie_names:
        st.markdown("- **" + i+'**')

