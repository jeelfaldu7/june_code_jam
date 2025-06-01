# Import necessary libraries
import re
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy import stats as st
from scipy import stats as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from dash import Dash, html, dcc, callback, Output, Input, State
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the dataset and read the data correctly
data = pd.read_csv('datasets/spotify.csv')

# Clean column names: replace spaces with underscores, remove special characters, and convert to lowercase
data.columns = [re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', col)).lower() for col in data.columns]

# Strip commas from length_duration entries that have them
data['length_duration'] = data['length_duration'].str.replace(',', '')

# Convert 'length_duration' column to integer data type
data['length_duration'] = data['length_duration'].astype(int)

data['genre_group'] = data['top_genre'].copy()

def make_freq_dict(series):
    #convert pd.Series to concatenated string
    words = " ".join(series.unique()).split(" ")

    #create a dictionary; if word exists, increment its number, otherwise make it exist and give it a counter of 1
    dictionary = {}
    for i in words:
        if i in dictionary:
            dictionary[i] += 1
        else:
            dictionary[i] = 1
        
    return dictionary

#leave entries that don't contain 'pop' alone; replace those that do with just 'pop'
data['genre_group'].where(~data['genre_group'].str.contains('pop'), 'pop', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('rock'), 'rock', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('hip hop'), 'hip hop', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('metal'), 'metal', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('folk'), 'folk', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('country'), 'country', inplace=True)
data['genre_group'].where(~data['genre_group'].str.contains('soul'), 'soul', inplace=True)

#TODO truncate genre_group to keep options from getting too cluttered

# Compute the correlation matrix
corr_matrix = data.corr(numeric_only=True)

CLIENT_ID = ""
CLIENT_SECRET = ""

# get spotify API information from secrets.txt
with open('secrets.txt', 'r') as file:
    CLIENT_ID = file.readline()
    CLIENT_SECRET= file.readline()

# instantiate spotipy
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# print("client id ", CLIENT_ID)
# print("client secret ", CLIENT_SECRET)

sp = spotipy.Spotify(auth_manager=auth_manager, client_credentials_manager=auth_manager)

app = Dash()

# App layout
app.layout = html.Div([
    html.H1('Spotify Dashboard 🎵', style={'textAlign': 'center'}),
    html.Label('Select Genre:'),
    dcc.Dropdown(
        id='genre_group-dropdown',
        options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
        value=data['genre_group'].unique()[0]
    ),
    # BONUS: audio preview
    html.Div([
        html.Label('Preview Song from Genre:'),
        dcc.Dropdown(
            id='preview-dropdown', 
            placeholder='Select title...'
        ),
        html.Div([
            html.Audio(
                id='audio-player',
                controls=True
            )     
        ])
           
    ]),
    html.Label('Select Year Range:'),
    dcc.RangeSlider(
        id='year-slider',
        min=data['year'].min(),
        max=data['year'].max(),
        step=1,
        marks={str(year): str(year) for year in range(data['year'].min(), data['year'].max()+1, 5)},
        value=[data['year'].min(), data['year'].max()]
    ),
    html.Label('Select Plot Type:'),
    dcc.RadioItems(
        id='plot_type',
        options=[
            {'label': 'Scatter Plot', 'value':'scatter'},
            {'label': 'Bar Plot', 'value': 'bar'},
        ],
        value='scatter', 
        inline=True, 
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),

    html.H2('Popularity & Danceability'),
    dcc.Graph(
        id='popularity-graph',
    ),

    html.H2('Interactive Correlation Heatmap'),
    dcc.Graph(
        id='heatmap',
        figure = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto=True 
        ).update_layout(
            height=800,
            width=1000,
            margin=dict(l=50, r=50, t=50, b=50)
        )
    )
])

# Callbacks for interactivity

#callback that updates track preview dropdown when genre is selected
@app.callback(
    Output('preview-dropdown', 'options'),
    Input('genre_group-dropdown', 'value'),
)
def update_preview_list(selected_genre):
    genre_filter = data[(data['genre_group'] == selected_genre)]
    titles = genre_filter['title']
    artists = genre_filter['artist']
    labels = genre_filter['artist'] + " - " + genre_filter['title']

    #TODO values may need changed depending on the needs of Spotify's API
    #so "values = labels" here is a placeholder until those needs are determined
    #values = labels
    #results = [{'label': i, 'value': j} for i,j in zip(labels, values)]
    results = [{'label': i, 'value': i} for i in labels]
    return results

#callback that queries Spotify API when Preview Song dropdown has a selection
@app.callback(
    Output('audio-player', 'src'),
    Input('preview-dropdown', 'value'),
    prevent_initial_call=True
)
def get_preview_audio(artist_and_title):
    #query the Spotify API for the track

    tags = artist_and_title.split(" - ")

    #note: %3A is HTML URL encoding for colon
    track='track:' + tags[1]        #e.g. track:"Love Me Tender"      
    artist=' artist:' + tags[0]     #e.g. artist:"Elvis Presley"
    query = track + artist

    #for URL encoding, replace spaces with +
    #query = query.replace(" ", "+")

    print(query)
    search_result = sp.search(q=query, type='track', limit=1)
    print(type(search_result))
    return ""

#callback that updates graph when genre is selected, or when year slider or plot type radio button are used
@app.callback(
    Output('popularity-graph', 'figure'),
    Input('preview-dropdown', 'value'),
    Input('year-slider', 'value'), 
    Input('plot_type', 'value')
)
def update_graph(selected_genre, year_range, plot_type):
    filtered_data = data[(data['genre_group'] == selected_genre) &
                      (data['year'] >= year_range[0]) &
                      (data['year'] <= year_range[1])]
    
    # Plot logic depending on the selected plot type
    if plot_type == 'scatter':
        fig = px.scatter(
            filtered_data,
            x='danceability',
            y='popularity',
            color='energy',
            title='Popularity versus Danceability',
            hover_data=['artist', 'title', 'year'],
        )

    elif plot_type == 'bar':
        filtered_data = filtered_data.copy()
        bins = [0, 0.3, 0.6, 1.0]
        labels = ['Low', 'Medium', 'High']

        # Create the bin column
        filtered_data['danceability_bin'] = pd.cut(
            filtered_data['danceability'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )

        # Group by danceability_bin and calculate average popularity
        danceability_summary = (
            filtered_data.groupby('danceability_bin')['popularity']
            .mean()
            .reset_index()
        )
    
        # Create a bar plot of average popularity for each danceability bin
        fig = px.bar(
            danceability_summary,
            x='danceability_bin',
            y='popularity',
            color='danceability_bin',
            title='Average Popularity by Danceability Bin'
        )

    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)