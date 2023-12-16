import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


client_id = 'aefa30cb2506438db3f3f8cf1792e2d1'
client_secret='acfcb81b7b604d7daa2ffcc0a7b2007b'

def test_api_call(spotify):
    birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
    results = spotify.artist_albums(birdy_uri, album_type='album')
    albums = results['items']
    while results['next']:
        results = spotify.next(results)
        albums.extend(results['items'])

    for album in albums:
        print(album['name'])

def get_spotify_api():
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))
    return spotify

def print_data(data, genre_data, year_data, artist_data):
    print(data.head(3))
    print(genre_data.head(3))
    print(year_data.head(3))
    print(artist_data.head(3))
    print(data.info())
    print(genre_data.info())

def tNSE_clustering(genre_data, data):
    # clustering of data
    '''
     Returns a fitted/trained song_cluster pipeline

     applies a standard scaler pipeline with a Kmeans clustering n=12 clusters one for each numeric column in genreDF.
     fit/train pipeline on genreDF then provided each row with its predicted cluster.
     Visualizing the Clusters with t-SNE (unsupervised ML algorithm).
     Visualizes the structure of our data 2 in dimensions with keys 'Genre' & 'Cluster'.
     Use of PCA for song clustering visualization.
     Song clustering done through Kmeans n=25 clusters.
    '''

    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=12))])
    #above we use a pipeline to define that we will use the standard scaler provided by sklearn, followed by specifying
    #the use of Kmeans Clustering with a total of 12 clusters
    X = genre_data.select_dtypes(np.number)
    # X holds the values for each row using only the columns of numerical type
    cluster_pipeline.fit(X)
    #Fit the pipeliune with x, using standardized numerical features and then fitting the K-means Clustering algorithm
    #to our scaled data
    genre_data['cluster'] = cluster_pipeline.predict(X)
    #Add a new column to our genre data which contains their genre cluster

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)  # returns np-array of coordinates(x,y) for each record after TSNE.
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']
    #create and display the plot using plotly express opens in web
    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'], title='Clusters of genres')
    #fig.show() #shows a ploty chart of our genre cluster

    #get and display a song cluster
    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                      ('kmeans', KMeans(n_clusters=25,
                                                        verbose=False))
                                      ], verbose=False)
    X = data.select_dtypes(np.number)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels
    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = data['name']
    projection['cluster'] = data['cluster_label']
    projection['artist'] = data['artists']

    fig = px.scatter(
        projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title', 'artist'], title='Clusters of songs')
    #fig.show() #shows a ploty chart of our songs cluster
    return song_cluster_pipeline

def find_songs_details_spotify(name, year):
    """
    Finds all song details from spotify dataset.
    If song is unavailable in dataset, it returns none.
    Is called when our dataset does not contain the song requested
    """
    sp = get_spotify_api()
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value
   # print(song_data) #print song info recieved from api call
    return pd.DataFrame(song_data) #song df as key:value pair

def get_song_details(song, spotify_data):
    #song is the current song used for recommendation,
    #spotify_data is the df that
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                 & (spotify_data['year'] == song['year'])].iloc[0]
        print('Fetching song information from local dataset')
        return song_data

    except IndexError:
        print('Fetching song information from spotify dataset')
        return find_songs_details_spotify(song['name'], song['year'])
    return None

def get_vector(songs_list, spotify_data, numerical_columns):
    """
    fetches info from our local data-set or ourside set
    calculates the mean of all numerical values in our dataframe of song_data
    returns a 1-d array of the mean of each element in the set
    """
    song_vectors = []
    for song in songs_list:
        song_data = get_song_details(song, spotify_data)
        if song_data is None:
            print("Song is not in either spotify or local dataset")
            continue
        song_vector = song_data[numerical_columns].values
        song_vectors.append(song_vector) #add curent song to vector
    song_matrix = np.array(list(song_vectors))
    print(f'song_matrix {song_matrix}') #comment out after tests
    return np.mean(song_matrix, axis=0) #returns the 1-d array

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = [] # 'name', 'year'
    for dic in dict_list:
        for key,value in dic.items():
            flattened_dict[key].append(value) # creating list of values
    print(flattened_dict) # test
    return flattened_dict
def recommend_songs(song_list, data, song_cluster, numerical_columns, num_songs = 8): #auto recommend 8 songs by default
    metadata_columns = ['name', 'year', 'artists']

    #get our flattened list to use as our song dictionary

    song_center = get_vector(song_list, data, numerical_columns)
    print(f'song_center {song_center}')
    song_dictionary = flatten_dict_list(song_list) # obtain flattened dictionary to

    scaler = song_cluster.steps[0][1]  # StandardScalar()
    scaled_data = scaler.transform(data[numerical_columns])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    print(f'distances {distances}')
    index = list(np.argsort(distances)[:, :num_songs][0])
    #Recommend songs
    recommended_songs = data.iloc[index]
    recommended_songs = recommended_songs[~recommended_songs['name'].isin(song_dictionary['name'])]
    #Above line
    return recommended_songs[metadata_columns].to_dict(orient='records')


def main():
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))
    test_api_call(spotify) #test spotify api connection thorugh spotipy
    data = pd.read_csv("data.csv")
    genre_data = pd.read_csv('data_by_genres.csv')
    year_data = pd.read_csv('data_by_year.csv')
    artist_data = pd.read_csv('data_by_artist.csv')
    #check data
    print_data(data, genre_data, year_data,artist_data)
    data['decade'] = data['year'].apply(lambda year: f'{(year // 10) * 10}s')
    sns.countplot(data['decade'],legend='full')
    #plt.show()
    numerical_columns =['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
     'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
    song_cluster = tNSE_clustering(genre_data, data)
    #get_song_details(spotify, data)

    #test our recommender engine
    current_song = [{'name': 'Anti-Hero', 'year': 2022}]
    #testing with a song that isnt in our dataset to test api function

    recommended = recommend_songs(current_song, data,song_cluster,numerical_columns)
    print('Songs recommend based on ',current_song)
    for x in range(len(recommended)):
        print(recommended[x])
    print(recommended)

main() #run main program