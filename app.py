# Import necessary libraries
import re
import os
import random
import sklearn
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
import dash_bootstrap_components as dbc
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Load the dataset and read the data correctly
data = pd.read_csv('datasets/spotify.csv')

# Clean column names: replace spaces with underscores, remove special characters, and convert to lowercase
data.columns = [re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', col)).lower() for col in data.columns]

# Strip commas from length_duration entries that have them
data['length_duration'] = data['length_duration'].str.replace(',', '')

# Convert 'length_duration' column to integer data type
data['length_duration'] = data['length_duration'].astype(int)

data['genre_group'] = data['top_genre'].copy()

features = [
    'beats_per_minute_bpm',
    'energy',
    'danceability',
    'loudness_db',
    'liveness',
    'valence',
    'length_duration',
    'acousticness',
    'speechiness',
    'popularity'
]

X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 5  # or use a dropdown as described
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data['pca1'] = X_pca[:, 0]
data['pca2'] = X_pca[:, 1]

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

# Calculating average features per top_genre
style_features = data.groupby('genre_group').mean(numeric_only=True)[[
    'beats_per_minute_bpm', 'energy', 'danceability', 'loudness_db',
    'liveness', 'valence', 'acousticness', 'speechiness', 'popularity'
]]

CLIENT_ID = ""
CLIENT_SECRET = ""

# get spotify API information from secrets.txt
with open('secrets.txt', 'r') as file:
    config = file.read().splitlines()
    CLIENT_ID = config[0]
    CLIENT_SECRET = config[1]

# instantiate spotipy
auth_manager = SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

# print("client id ", CLIENT_ID)
# print("client secret ", CLIENT_SECRET)

sp = spotipy.Spotify(auth_manager=auth_manager, client_credentials_manager=auth_manager)

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY]) 


# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Alert(
                [
                    html.H1("Welcome to the Spotify Dashboard! 🎵", className="alert-heading"),
                    html.P(
                        "Explore Spotify's track data: find popular genres, discover relationships between track features, "
                        "and more. Use the dropdowns and plots to dive deeper!"
                    )
                ],
                color="primary",
                className="mt-4"
            ),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Genre for Style Polar Chart:'),
            dcc.Dropdown(
                id='genre-polar-dropdown',
                options=[{'label': genre, 'value': genre} for genre in data['top_genre'].unique()],
                value=data['top_genre'].unique()[0]
            ),
            dcc.Graph(id='polar-style-chart')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Interactive Correlation Heatmap', className='text-center'),
            dcc.Graph(id='heatmap'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Popularity by Genre (Top 25 Most Popular)', className='text-center'),
            dcc.Graph(id='popularity-by-genre-graph'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Genre:'),
            dcc.Dropdown(
                id='genre_group-dropdown-graph',
                options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
                value=data['genre_group'].unique()[0]
            ),
            dbc.Tooltip(
                "Select a genre to see its popularity trends!",
                target="genre_group-dropdown-graph" 
                ),
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
            dcc.Graph(id='popularity-graph'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Top 10 Artists by Popularity', className='text-center'),
            html.Label("Select Artists:", className="fw-bold"),  # Use Bootstrap classes for styling
            dcc.Graph(
                id='artist-graph',
            ),
            dcc.Dropdown(
                id='artist-dropdown',
                options=[{'label': a, 'value': a} for a in sorted(data['artist'].unique())],
                multi=True,
                placeholder="Select one or more artists"
            ),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Genre for Previews:'),
            dcc.Dropdown(
                id='genre_group-dropdown-preview',
                options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
                value=data['genre_group'].unique()[0]
            ),
            html.Label('Preview Song from Genre:'),
            dcc.Dropdown(
                id='preview-dropdown', 
                placeholder='Select title...'
            ),
            html.Div([
                html.Iframe(id='audio-player', style={'width': '100%', 'height': '80px', 'border': 'none'})
            ])
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('K-Means Clusters (PCA Projection)', className='text-center'),
            dcc.Graph(id='kmeans-cluster-graph')  # ID for this plot
        ], width=12)
    ]),
], fluid=True)

# Callbacks for interactivity

@app.callback(
    Output('polar-style-chart', 'figure'),
    Input('genre-polar-dropdown', 'value')
)
def update_polar_chart(selected_genre):
    # Filter data for the selected genre
    genre_data = data[data['top_genre'] == selected_genre]

    # Compute average features
    features = genre_data[[
        'beats_per_minute_bpm', 'energy', 'danceability', 'loudness_db',
        'liveness', 'valence', 'acousticness', 'speechiness', 'popularity'
    ]].mean()

    # Prepare data for polar chart
    categories = features.index.tolist()
    values = features.tolist()

    fig = px.line_polar(
        r=values + [values[0]],  # repeat the first value to close the loop
        theta=categories + [categories[0]],
        line_close=True,
        title=f'Style Profile: {selected_genre}',
        template='plotly_dark'
    )

    fig.update_traces(fill='toself', line_color='lime')
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    return fig

#callback that updates track preview dropdown when genre is selected
@app.callback(
    Output('preview-dropdown', 'options'),
    Input('genre_group-dropdown-preview', 'value'),
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
    track_id = search_result["tracks"]["items"][0]["id"]
    print("track id:", track_id)
    src = "https://open.spotify.com/embed/track/" + track_id + "?utm_source=generator"
    print("src:", src)
    return src

#callback that updates graph when genre is selected, or when year slider or plot type radio button are used
@app.callback(
    Output('popularity-graph', 'figure'),
    Input('genre_group-dropdown-graph', 'value'),
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
            color_continuous_scale='Viridis',
            labels={'popularity': 'Popularity', 'danceability': 'Danceability', 'energy': 'Energy'},
            title='Popularity vs Danceability',
            hover_data={'artist': True, 'title': True, 'year': True, 'popularity': True}
        )

    elif plot_type == 'bar':
        filtered_data = filtered_data.copy()
        bins = [10,40,70,100]
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

@app.callback(
    Output('popularity-by-genre-graph', 'figure'),
    Input('genre_group-dropdown-graph', 'value')  # just to trigger once on app load
)
def update_popularity_by_genre(_):
    pop_genre = data.groupby('genre_group')['popularity'].mean().sort_values(ascending=False).head(25).reset_index()

    fig = px.bar(
        pop_genre,
        x='popularity',
        y='genre_group',
        orientation='h',
        title='Popularity by Genre (Top 25 Most Popular)',
        labels={'popularity': 'Average Popularity', 'genre_group': 'Genres'},
        color='popularity',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      template='ggplot2',
                      font=dict(
                          family='Helvetica, Arial, sans-serif',
                          size=14,
                          color='#333'
                        )
    )  # highest popularity on top

    return fig

@app.callback(
    Output('heatmap', 'figure'),
    Input('genre_group-dropdown-graph', 'value')  # just to trigger once on app load
)
def update_interactive_heatmap(_):
    fig = px.imshow(
            corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto=True 
    )
    fig.update_layout(
            height=800,
            width=1500,
            template='ggplot2',
            margin=dict(l=50, r=50, t=50, b=50),
            font=dict(
                family='Helvetica, Arial, sans-serif',
                size=14,
                color='#333'
                )
            )
    return fig

@app.callback(
    Output('kmeans-cluster-graph', 'figure'),
    Input('genre_group-dropdown-graph', 'value')  # trigger callback to render once
)
def update_kmeans_cluster_graph(_):
    fig = px.scatter(
        data,
        x='pca1',
        y='pca2',
        color='cluster',
        hover_data=['title', 'artist', 'top_genre', 'year'],
        title=f'K-Means Clustering of Songs (k={k})',
        color_continuous_scale='Viridis'  # Or discrete color sequence
    )
    fig.update_layout(
        height=600,
        width=900,
        template='ggplot2',
        font=dict(family='Helvetica, Arial, sans-serif', size=14, color='#333')
    )
    return fig

app.layout.children.append(
    dbc.Row([
        dbc.Col(html.Footer('© 2025 Jeel Faldu / Project. Data Source: Spotify', className='text-center text-muted py-2'), width=12)
    ])

)

@app.callback(
    Output('artist-graph', 'figure'),
    Input('artist-dropdown', 'value')
)
def update_artist_graph(_):
    # Filter to selected artists
    top_artists = data.groupby('artist')['popularity'].mean().sort_values(ascending=False).head(10).reset_index()

    fig = px.bar(
        top_artists,
        x='artist',
        y='popularity',
        color='popularity',
        title='Artists Popularity',
        orientation='h',
        color_continuous_scale='Blues'

    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)