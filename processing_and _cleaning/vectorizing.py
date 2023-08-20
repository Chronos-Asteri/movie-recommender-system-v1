import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import time
import os
import bz2

start_time = time.time()

final_movies = pd.read_csv('../cleaned_and_processed_dataset/cleaned_data_to_be_vectorized.csv')

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(final_movies['tags']).toarray()
similarity = cosine_similarity(vector)

movie_loc =  '../similarity_and_movie_list_datasets/movie_list.pkl'
similarity_loc = '../similarity_and_movie_list_datasets/similarity.pkl'

pickle.dump(final_movies, bz2.BZ2File(movie_loc,'wb'))
print(f"File Created: {movie_loc}")
pickle.dump(similarity, bz2.BZ2File(similarity_loc,'wb'))
print(f"File Created: {similarity_loc}")

end_time = time.time()

print(f'Execution Time: {round(end_time-start_time, 2)}s')

# If you want to see how this works see this file ->  './extras/recom.ipynb'