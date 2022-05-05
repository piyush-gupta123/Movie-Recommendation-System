import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = pd.read_csv('/content/movies.csv')
print(movies_data.head)

print(movies_data.shape)

#Extracting important features
selected_features=['genres' , 'keywords', 'tagline', 'cast', 'director']
print(selected_features)

for feature in selected_features:
  movies_data[feature]= movies_data[feature].fillna('')

#Combining important features
combined_features=movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

print(combined_features.head())

#Converting text into number 
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)

print(feature_vector)

#Finding similar movies
similarity = cosine_similarity(feature_vector)

print(similarity)

#Getting movie name from user
movie = input("Enter your Movie:")

#creating list of all titles
list_of_all_objects = movies_data['original_title'].tolist()

#Finding close matches
movie_match = difflib.get_close_matches(movie, list_of_all_objects)

close_match = movie_match[0]

#Finding index of the movie with title
index_of_movie = movies_data[movies_data.original_title == close_match]['index'].values[0]

#Getting list of similar movies with its similarity score
similarity_score = list(enumerate(similarity[index_of_movie]))

#Sorting the movies based on similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse=True)

#Print name of similar movies
print('Movies Suggested For You:/n')
i=0
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index= movies_data[movies_data.index == index]['title'].values[0]
  if(i<20):
    print(i+1 ,'.',title_from_index)
    i+=1
