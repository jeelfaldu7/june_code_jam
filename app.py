# Import necessary libraries
import re
import os
import random
from decimal import Decimal, localcontext, ROUND_DOWN
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
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# function for truncating floats later, from StackOverflow:
def truncate(number, places):
    if not isinstance(places, int):
        raise ValueError("Decimal places must be an integer.")
    if places < 1:
        raise ValueError("Decimal places must be at least 1.")
    # If you want to truncate to 0 decimal places, just do int(number).

    with localcontext() as context:
        context.rounding = ROUND_DOWN
        exponent = Decimal(str(10 ** - places))
        return Decimal(str(number)).quantize(exponent).to_eng_string()

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

# genre_groups after the first 15 or so can be moved to "other"
genres = data['genre_group'].value_counts()
query = ((genres > 15)[genres >15]).index

data['genre_group'].where(data['genre_group'].isin(query.values), 'other', inplace=True)

# Compute the correlation matrix
corr_matrix = data.corr(numeric_only=True)

# Calculating average features per genre_group
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

# labels
# these are for any instance where we must display "Artist - Title", such as in dropdowns
artist_title_labels = data['artist'] + " - " + data['title']
artist_title_values = data.index.to_series()

# genre_group options
# this sets up the options for k-means clustering by genre
genre_group_options = [{'label': genre, 'value': genre} for genre in data['genre_group'].unique()]
all_label = {'label': 'All Genres', 'value': 'all'}
genre_group_options.insert(0, all_label)

# current_src
# this variable holds the URL for the song preview, which works around a bug
current_src = None

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
    dbc.Card(
        dbc.CardBody([
            html.H1("🎶 Welcome to the Interactive Music Trends Dashboard! 🎶", 
                className="text-center mt-4", 
                style={"color": "#2dd4bf"}),
            html.P(
                """
                This dynamic dashboard explores music listening trends using Spotify data, including audio features 
                like danceability, energy, loudness, and popularity. Using a publicly available dataset from Kaggle, this 
                app helps you uncover how these musical characteristics relate to song popularity across genres and time.
                """,
                className="text-start fs-5", style={"color": "#ffffff", "text-indent": '40px'}
            ),
            html.Ul([
                html.Li("Explore how different musical features shape the popularity of songs"),
                html.Li("Analyze trends across various genres"),
                html.Li("Dive into how musical preferences change over time"),
                html.Li("Interactively filter data by genre, year, and artist for tailored insights")
            ], className="text-start fs-5", style={"color": "#ffffff", "text-indent": '20px', "margin-left": "40px"}),
            html.P(
                """
                Whether you’re a data enthusiast, music lover, or just curious about what makes a song popular, 
                this dashboard provides an engaging way to dig deeper. Let’s dive in and explore the data together!
                """,
                className="text-start fs-5", style={"color": "#ffffff", "text-indent": '40px'}
            ),
        ]),
        style={
            "background-color": "#1c1c2e",
            "border-radius": "15px",
            "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
            "padding": "20px",
        },
    )
], id="title-card", width=12)

    ]),
    dbc.Row([
    dbc.Col([
        html.H2('Polar Chart for Features by Genre', className='text-center',  style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
        html.Label(
            'Select Genre for Style Polar Chart:',
            style={
                'background-color': '#f8f8f0',
                'color': '#1c1c2e',
                'padding': '5px',
                'border-radius': '4px',
                'display': 'inline-block',
                'margin-top': '10px'
            }
        ),
        dcc.Dropdown(
            id='genre-polar-dropdown',
            options=[
                {'label': genre, 'value': genre} 
                for genre in data['genre_group'].unique()
            ],
            value=data['genre_group'].unique()[0],
            style={
                'background-color': '#f8f8f0',
                'color': '#1c1c2e',
                'border': '1px solid #2dd4bf',
                'border-radius': '4px',
                'padding': '5px'
            }
        ),
        dcc.Graph(id='polar-style-chart')
    ], width=6),

    dbc.Col([
        # Add an empty label for whitespace
        dbc.Row(html.Label('')),
        html.P(
            "Our data contains many features. Here are explanations of a few of them:",
            style={
                "margin-top": '16px',
                "color": "#ffffff"
            }
        ),
        html.Ul([
            html.Li("Acousticness: confidence that the track is acoustic. Non-acoustic music is 0, acoustic music is 100."),
            html.Li("Danceability: how suitable a track is for dancing, rated from 0 to 100."),
            html.Li("Energy: how energetic the track feels. Fast, loud, and noisy tracks score high."),
            html.Li("Liveness: scores high if an audience can be heard in the recording."),
            html.Li("Speechiness: scores high if the track is mostly spoken words."),
            html.Li("Valence: cheery or euphoric music scores high, while sad or angry music scores low."),
            html.Li("Popularity: based on an internal Spotify algorithm, tracks that are played a lot, especially recently, score high.")
        ], style={
            "text-indent": '20px',
            "margin-left": '40px',
            "color": "#ffffff"
        }),
        html.A(
            "More details can be found in Spotify's API documentation.",
            href='https://developer.spotify.com/documentation/web-api/reference/get-audio-features',
            id='spotify-api-link',
            style={"color": "#2dd4bf"}  # link color for better contrast
        ),
        html.P(
            "Note: Loudness and Beats Per Minute are scaled from 0-100 based on the data available to us.",
            style={
                "margin-top": '16px',
                "color": "#ffffff"
            }
        ),
    ], width=6, style={
        "background-color": "#1c1c2e",
        "border-radius": "15px",
        "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
        "padding": "20px"
    }),
], id="polar-chart-display-area"),

# Additional row for popularity graph
dbc.Row([
    html.H4("🕰️ Curious about musical eras?", style={"marginTop": "30px", "color": "#1c1c2e"}),
    html.P(id='era-subtitle', style={"marginBottom": "10px", "color": "#1c1c2e"}),

    dcc.RangeSlider(
        id='year-slider',
        min=data['year'].min(),
        max=data['year'].max(),
        step=1,
        marks={str(year): str(year) for year in range(data['year'].min(), data['year'].max()+1, 5)},
        value=[data['year'].min(), data['year'].max()],
    ),
    dbc.Col([
        html.H2(
            'Popularity by Genre (Top 10 Most Popular)',
            className='text-center',
            style={
                "color": "#1c1c2e",
                "textAlign": "center",
                "marginTop": "20px"
            }
        ),
        dcc.Graph(id='popularity-by-genre-graph'),
    ], width=12)
]),
    dbc.Row([
        dbc.Col([
            html.H2('Popularity & Danceability', className='text-center',  style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            html.Label('Select Genre:'),
            dcc.Dropdown(
                id='genre_group-dropdown-graph',
                options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
                value=data['genre_group'].unique()[0],
                style={
                    'background-color': '#f8f8f0',   # cream/off-white background
                    'color': '#1c1c2e',              # text color (dark navy)
                    'border': '1px solid #2dd4bf',   # border color (teal) as accent
                    'border-radius': '4px',          # slight border rounding
                    'padding': '5px'                 # optional padding
                }
            ),
            dbc.Tooltip(
                "Select a genre to see its popularity trends!",
                target="genre_group-dropdown-graph" ,
                style={
                    'background-color': '#f8f8f0',   # cream/off-white background
                    'color': '#1c1c2e',              # text color (dark navy)
                    'border': '1px solid #2dd4bf',   # border color (teal) as accent
                    'border-radius': '4px',          # slight border rounding
                    'padding': '5px'                 # optional padding
                }
            ),
            
            dcc.Graph(id='popularity-graph'),
        ], width=12)
    ]),
    dbc.Row([
        html.H2("🔋 Energy Over Time by Genre", className='text-center', style={"color": "#1c1c2e", "marginTop": "40px"}),

        html.Label("Select Genre(s):"),
        dcc.Dropdown(
            id='energy-genre-dropdown',
            options=[{'label': genre, 'value': genre} for genre in sorted(data['genre_group'].unique())],
            multi=True,
            placeholder="Select one or more genres...",
            style={
                'background-color': '#f8f8f0',
                'color': '#1c1c2e',
                'border': '1px solid #2dd4bf',
                'border-radius': '4px',
                'padding': '5px'
            }
        ),

        dcc.Graph(id='energy-line-graph')

    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Top 10 Artists by Popularity', className='text-center',  style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),

            html.Label('Filter by Genre:'),
            dcc.Dropdown(
                id='artist-genre-dropdown',
                options=genre_group_options,
                placeholder='Select a genre (optional)',
                style={
                    'background-color': '#f8f8f0',
                    'color': '#1c1c2e',
                    'border': '1px solid #2dd4bf',
                    'border-radius': '4px',
                    'padding': '5px'
                }
            ),

            html.Label('Filter by Artist:'),
            dcc.Dropdown(
                id='artist-dropdown',
                options=[{'label': a, 'value': a} for a in sorted(data['artist'].unique())],
                placeholder = 'Select Specific Artists to see a Popularity Comparison',
                multi=True,
                style={
                    'background-color': '#f8f8f0',
                    'color': '#1c1c2e',
                    'border': '1px solid #2dd4bf',
                    'border-radius': '4px',
                    'padding': '5px'
                }
            ),

            dcc.Graph(id='artist-graph'),
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('Popular Tracks (Top 10)', className='text-center',  style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),    
            html.H5('Choose a genre:', className='text-center',  
                style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
            dbc.Row([
                dcc.Dropdown(
                    id='popular-song-genre-dropdown',
                    # options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
                    options=genre_group_options,
                    value=genre_group_options[0]['value'],
                    style={
                        'background-color': '#f8f8f0',   # cream/off-white background
                        'color': '#1c1c2e',              # text color (dark navy)
                        'border': '1px solid #2dd4bf',   # border color (teal) as accent
                        'border-radius': '4px',          # slight border rounding
                        'padding': '5px'                 # optional padding
                    }
                )
            ]),
            dbc.Row([
                dcc.Graph(id='popular-song-graph')  # ID for this plot
            ]),

    ]),

    dbc.Row([
        dbc.Col([
            html.H4("🎧 Ready to preview a song?", style={"marginTop": "30px", "color": "#1c1c2e"}),
            html.P("Use the dropdowns below to select a genre and pick a song to hear a short preview!", style={"marginBottom": "10px", "color": "#1c1c2e"}),
            html.Label('Select Genre for Previews:'),
            dcc.Dropdown(
                id='genre_group-dropdown-preview',
                options=[{'label': genre, 'value': genre} for genre in data['genre_group'].unique()],
                value=data['genre_group'].unique()[0],
                style={
                    'background-color': '#f8f8f0',   # cream/off-white background
                    'color': '#1c1c2e',              # text color (dark navy)
                    'border': '1px solid #2dd4bf',   # border color (teal) as accent
                    'border-radius': '4px',          # slight border rounding
                    'padding': '5px'                 # optional padding
                }
            ),

            html.Label('Preview Song from Genre:'),
            dcc.Dropdown(
                id='preview-dropdown', 
                placeholder='Select title...',
                style={
                    'background-color': '#f8f8f0',   # cream/off-white background
                    'color': '#1c1c2e',              # text color (dark navy)
                    'border': '1px solid #2dd4bf',   # border color (teal) as accent
                    'border-radius': '4px',          # slight border rounding
                    'padding': '5px'                 # optional padding
                }
            ),
            html.Div([
                html.Iframe(id='audio-player', style={'width': '100%', 'height': '80px', 'border': 'none'})
            ])
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H2('K-Means Clusters (PCA Projection)', className='text-center',  
                    style={'background-color': '#f8f8f0', "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.P('Investigate tracks that are similar to each other here!', className='text-center',  
                   style={'background-color': '#f8f8f0', "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.H5('Choose a genre:', className='text-center',  
                style={'background-color': '#f8f8f0', "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
            dbc.Row([
                dcc.Dropdown(
                    id='cluster-genre-dropdown',
                    options=genre_group_options,
                    value=genre_group_options[0]['value'],
                    style={
                        'background-color': '#f8f8f0',
                        'color': '#1c1c2e',
                        'border': '1px solid #2dd4bf',
                        'border-radius': '4px',
                        'padding': '5px'
                    }
                )
            ]),
            dbc.Row([
                dcc.Graph(id='kmeans-cluster-graph')
            ]),
        ]),
        dbc.Col([
            dbc.Row([
                html.H5('Choose a track:', className='text-center',  
                style={'background-color': '#f8f8f0', "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
            dbc.Row([
                dcc.Dropdown(
                    id='cluster-track-dropdown',
                    options=[{'label': i[0], 'value': i[1]} for i in zip(artist_title_labels, artist_title_values)],
                    placeholder="Select a track", 
                    style={
                        'background-color': '#f8f8f0',
                        'color': '#1c1c2e',
                        'border': '1px solid #2dd4bf',
                        'border-radius': '4px',
                        'padding': '5px'
                    }
                ),
            ]),
            dbc.Row([
                html.P(id='cluster-subtitle', style={
                    "color": "#1c1c2e",
                    "fontStyle": "italic",
                    "marginBottom": "10px",
                    "marginTop": "10px"
                }),
                dbc.Table(
                    id='cluster-table',
                    style={
                        'background-color': '#f8f8f0',
                        'color': '#1c1c2e',
                        'border': '1px solid #2dd4bf',
                        'border-radius': '4px',
                        'padding': '5px'
                    } 
                )
            ])
        ]),
    ]),
], 
    fluid=True,
    style={"background-color": "#f8f8f0", "min-height": "100vh", "padding": "20px"}
)

# Callbacks for interactivity

@app.callback(
    Output('polar-style-chart', 'figure'),
    Input('genre-polar-dropdown', 'value')
)
def update_polar_chart(selected_genre):
    
    if(selected_genre == None):
        selected_genre = "adult standards"
    scalable = ['beats_per_minute_bpm', 'loudness_db']

    # scale bpm and loudness based on full data
    minmax_scaler = MinMaxScaler(feature_range=(0, 100))
    minmax_scaler.fit(data[scalable])

    # create genre_data as a copy of data, then scale its features as needed
    genre_data = data.copy()
    scaled_features = minmax_scaler.transform(genre_data[scalable])
    scaled_features = pd.DataFrame(scaled_features, columns=scalable)
    genre_data['beats_per_minute_bpm'] = scaled_features['beats_per_minute_bpm']
    genre_data['loudness_db'] = scaled_features['loudness_db']

    # Filter data for the selected genre
    genre_data = genre_data[genre_data['genre_group'] == selected_genre]

    # Compute average features
    features = genre_data[[
        'beats_per_minute_bpm', 'energy', 'danceability', 'loudness_db',
        'liveness', 'valence', 'acousticness', 'speechiness', 'popularity'
    ]].mean()

    # Prepare data for polar chart
    categories = []
    for col in features.index:
        if col == 'beats_per_minute_bpm':
            categories.append('Beats Per Minute')
        elif col == 'loudness_db':
            categories.append('Loudness')
        else:
            categories.append(col.replace('_', ' ').title())


    values = features.tolist()

    fig = px.line_polar(
        r=values + [values[0]],  # repeat the first value to close the loop
        theta=categories + [categories[0]],
        line_close=True,
        title=f'Style Profile: {selected_genre}',
        template='plotly_white',
    )

    fig.update_traces(fill='toself', line_color='lime')
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      paper_bgcolor='#f8f8f0',  
                      plot_bgcolor="#f8f8f0"
                      )
    return fig

#callback that updates track preview dropdown when genre is selected
@app.callback(
    Output('preview-dropdown', 'options'),
    Input('genre_group-dropdown-preview', 'value'),
)
def update_preview_list(selected_genre):
    if(selected_genre == None):
        selected_genre = "adult standards"
    genre_filter = data[(data['genre_group'] == selected_genre)]
    labels = genre_filter['artist'] + " - " + genre_filter['title']

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

    if (artist_and_title == None):
        return current_src
    tags = artist_and_title.split(" - ")

    #note: %3A is HTML URL encoding for colon
    track='track:' + tags[1]        #e.g. track:"Love Me Tender"      
    artist=' artist:' + tags[0]     #e.g. artist:"Elvis Presley"
    query = track + artist

    #for URL encoding, replace spaces with +
    #query = query.replace(" ", "+")

    search_result = sp.search(q=query, type='track', limit=1)
    track_id = search_result["tracks"]["items"][0]["id"]
    src = "https://open.spotify.com/embed/track/" + track_id + "?utm_source=generator"

    #store src in current_src in case artist_and_title becomes None
    current_src = src
    return src


#callback that finds nearest neighbors of the track from the cluster-track-dropdown
@app.callback(     
    Output('cluster-table', 'children'),
    Input('cluster-track-dropdown', 'value'),
    prevent_initial_call=True
)
def get_track_nn(track_index):

    if(track_index==None):
        return None

    #number of results to display
    k = 10

    #list of quantitative columns
    numeric = ['year', 'beats_per_minute_bpm', 'energy', 'danceability', 'loudness_db', 
              'liveness', 'valence', 'length_duration', 'acousticness', 'speechiness', 'popularity']
    
    #create dataframe for results
    #this copy operation is probably performance-expensive; move elsewhere if needed
    nn_results = data.copy()

    #scale nn_results to ensure numeric features are equally represented
    mas_scaler = MaxAbsScaler()
    mas_scaler.fit(nn_results[numeric].to_numpy())
    nn_results.loc[:, numeric] = mas_scaler.transform(nn_results[numeric].to_numpy())
   
    #calculate distances between observation 'track_index' and all other points
    distances = cdist(nn_results[numeric], [nn_results[numeric].iloc[track_index]], metric='euclidean')

    nn_results['distance'] = pd.Series(distances[:, 0])

    #sort nn_results by distance
    nn_results.sort_values(by='distance', axis=0, inplace=True, ascending=True)
    nn_results = nn_results.iloc[1:k+1]
    nn_results.reset_index(drop=True, inplace=True)

    #generate children of the cluster-table component
    table_contents = [
        dbc.Row([
            html.H6('Tracks with similar metrics:', className='text-center',  
                style={"color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
        ]),
        dbc.Row([
            dbc.Col([
                html.P('Artist', className='text-center',  
                    style={"background-color": "#f8f8f0", "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
            dbc.Col([
                html.P('Title', className='text-center',  
                    style={"background-color": "#f8f8f0", "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
            dbc.Col([
                html.P('Distance', className='text-center',  
                    style={"background-color": "#f8f8f0", "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
            ]),
        ]),
    ]

    #append data to table_contents
    for i in range(0, k):
        table_contents.append(
            dbc.Row([
                dbc.Col([
                    html.P(nn_results.iloc[i]['artist'], className='text-center',  
                        style={"background-color": "#f8f8f0","color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
                ]),
                dbc.Col([
                    html.P(nn_results.iloc[i]['title'], className='text-center',  
                        style={"background-color": "#f8f8f0", "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
                ]),dbc.Col([
                    # truncate distance to three decimals
                    html.P(truncate(nn_results.iloc[i]['distance'], 3), className='text-center',  
                        style={"background-color": "#f8f8f0", "color": "#1c1c2e", "textAlign": "center", "marginTop": "20px"}),
                ]),
            ])
        )

    return table_contents

#callback that updates graph when genre is selected, or when year slider or plot type radio button are used
@app.callback(
    Output('popularity-graph', 'figure'),
    Input('genre_group-dropdown-graph', 'value'),
    Input('year-slider', 'value')
    )
def update_graph(selected_genre, year_range):

    if (selected_genre == None):
        selected_genre = 'adult standards'
    if (year_range == None):
        year_range = [data['year'].min(), data['year'].max()]

    filtered_data = data[(data['genre_group'] == selected_genre) &
                      (data['year'] >= year_range[0]) &
                      (data['year'] <= year_range[1])]
    
    fig = px.scatter(
        filtered_data,
        x='danceability',
        y='popularity',
        color='energy',
        color_continuous_scale='Viridis',
        labels={'popularity': 'Popularity', 'danceability': 'Danceability', 'energy': 'Energy'},
        hover_data={'artist': True, 'title': True, 'year': True, 'popularity': True},
    )

    fig.update_layout(
        template='ggplot2',
        font=dict(family='Helvetica, Arial, sans-serif', size=14, color='#333'),
        paper_bgcolor='#f8f8f0',
        plot_bgcolor="#f8f8f0"
    )
    
    return fig

@app.callback(
    Output('popularity-by-genre-graph', 'figure'),
    Input('year-slider', 'value')
)
def update_popularity_by_genre(year_range):
    start_year, end_year = year_range
    filtered_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

    # Use only the cleaned genre_group column to avoid obscure genres
    pop_genre = (
        filtered_data
        .groupby('genre_group')['popularity']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )


    fig = px.bar(
        pop_genre,
        x='popularity',
        y='genre_group',
        orientation='h',
        labels={'popularity': 'Average Popularity', 'genre_group': 'Genres'},
        color='popularity',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      template='ggplot2',
                      paper_bgcolor='#f8f8f0', 
                      plot_bgcolor="#f8f8f0",
                      font=dict(
                          family='Helvetica, Arial, sans-serif',
                          size=14,
                          color='#333'),
    )  # highest popularity on top

    return fig

@app.callback(
    Output('popular-song-graph', 'figure'),
    Input('popular-song-genre-dropdown', 'value')  # just to trigger once on app load
)
def update_popular_songs(genre):
    if (genre == None):
        genre = 'all'

    pop_songs = data.copy()
    if (genre != 'all'):
        pop_songs = pop_songs[pop_songs['genre_group'].isin([genre])]
    pop_songs = pop_songs.sort_values(by='popularity', ascending=False).head(10).reset_index()
    pop_songs['artist_title'] = pop_songs['artist'] + " - " + pop_songs['title']
    fig = px.bar(
        pop_songs,
        x='popularity',
        y='artist_title',
        orientation='h',
        labels={'popularity': 'Popularity', 'artist_title': 'Track'},
        color='popularity',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      template='ggplot2',  
                      paper_bgcolor='#f8f8f0', 
                      plot_bgcolor="#f8f8f0",
                      font=dict(
                          family='Helvetica, Arial, sans-serif',
                          size=14,
                          color='#333'),
    )  # highest popularity on top

    return fig

@app.callback(
    Output('kmeans-cluster-graph', 'figure'),
    Input('cluster-genre-dropdown', 'value'),
    Input('cluster-track-dropdown', 'value')
)
def update_kmeans_cluster_graph(genre, selected_track_index):
    if genre is None:
        genre = 'all'

    df = data.copy()
    if genre != 'all':
        df = df[df['genre_group'] == genre]

    df['highlight'] = 'Other Tracks'

    selected_cluster = None
    selected_point = None

    if selected_track_index is not None:
        selected_cluster = data.loc[selected_track_index, 'cluster']
        selected_point = data.loc[selected_track_index]
        df.loc[df['cluster'] == selected_cluster, 'highlight'] = f'Cluster {selected_cluster}'

    # Base scatter plot with colors
    fig = px.scatter(
        df,
        x='pca1',
        y='pca2',
        color='highlight',
        hover_data=['title', 'artist', 'genre_group', 'year'],
        color_discrete_map={
            'Other Tracks': 'lightgray',
            f'Cluster {selected_cluster}': 'royalblue',
            'Selected Track': 'crimson'
        } if selected_cluster is not None else {},
        labels={'pca1': 'PCA1', 'pca2': 'PCA2'},
        title='K-Means Clustering of Songs (PCA)'
    )

    # Emphasize selected track with larger size & different marker
    if selected_point is not None:
        fig.add_scatter(
            x=[selected_point['pca1']],
            y=[selected_point['pca2']],
            mode='markers+text',
            marker=dict(size=14, symbol='star', color='crimson', line=dict(width=2, color='black')),
            name='Selected Track',
            text=[selected_point['title']],
            textposition='top center',
            hovertext=f"{selected_point['artist']} - {selected_point['title']}",
            showlegend=True
        )

    fig.update_layout(
        height=600,
        width=900,
        template='ggplot2',
        font=dict(family='Helvetica, Arial, sans-serif', size=14, color='#333'),
        paper_bgcolor='#f8f8f0',
        plot_bgcolor='#f8f8f0'
    )

    return fig


app.layout.children.append(
    dbc.Row([
        dbc.Col(html.Footer('© 2025 Jeel Faldu, Ken Klabnik, Tirso Paneque  / Project. Data Source: Spotify', className='text-center text-muted py-2'), width=12)
    ])

)

@app.callback(
    Output('artist-graph', 'figure'),
    Input('artist-dropdown', 'value'),
    Input('artist-genre-dropdown', 'value'),
    Input('year-slider', 'value')
)
def update_artist_graph(selected_artists, selected_genre, year_range):

    if (year_range == None):
        year_range = [data['year'].min(), data['year'].max()]
    filtered_data = data.copy()

    # Filter by year range first
    filtered_data = filtered_data[
        (filtered_data['year'] >= year_range[0]) &
        (filtered_data['year'] <= year_range[1])
    ]

    # Then filter by genre if one is selected
    if selected_genre:
        filtered_data = filtered_data[filtered_data['genre_group'] == selected_genre]

    # Then filter by selected artists if any
    if selected_artists:
        filtered_data = filtered_data[filtered_data['artist'].isin(selected_artists)]

    top_artists = (
        filtered_data.groupby('artist')['popularity']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_artists,
        x='popularity',
        y='artist',
        color='popularity',
        title='Artists Popularity',
        orientation='h',
        color_continuous_scale='Viridis',
        labels={'popularity': 'Popularity', 'artist': 'Artists'}
    )
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      paper_bgcolor='#f8f8f0',  
                      plot_bgcolor="#f8f8f0")
    return fig


# Link this dropdown to the callback by syncing IDs
@app.callback(
    Output('artist-dropdown', 'options'),
    Input('artist-genre-dropdown', 'value'),
    Input('year-slider', 'value')
)
def update_artist_dropdown(genre, year_range):
    if (year_range == None):
        year_range = [data['year'].min(), data['year'].max()]

    filtered = data.copy()
    if genre:
        filtered = filtered[filtered['genre_group'] == genre]
    filtered = filtered[(filtered['year'] >= year_range[0]) & (filtered['year'] <= year_range[1])]
    options = [{'label': a, 'value': a} for a in sorted(filtered['artist'].unique())]
    return options

@app.callback(
    Output('era-subtitle', 'children'),
    Input('year-slider', 'value')
)
def update_era_subtitle(year_range):
    start_year, end_year = year_range
    era_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

    if era_data.empty:
        return "No popular songs found in this era. Maybe a hidden gem?"

    top_song = era_data.sort_values(by='popularity', ascending=False).iloc[0]
    return f"Oh, so you like the era of \"{top_song['title']}\" ruled by {top_song['artist']}!"

@app.callback(
    Output('energy-line-graph', 'figure'),
    [Input('year-slider', 'value'),
     Input('energy-genre-dropdown', 'value')]
)
def update_energy_line_graph(year_range, selected_genres):
    start_year, end_year = year_range
    filtered_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

    if not selected_genres:
        return px.line(title="Select at least one genre to display energy trends.")

    filtered_data = filtered_data[filtered_data['genre_group'].isin(selected_genres)]

    energy_over_time = (
        filtered_data
        .groupby(['year', 'genre_group'])['energy']
        .mean()
        .reset_index()
    )

    fig = px.line(
        energy_over_time,
        x='year',
        y='energy',
        color='genre_group',
        markers=True,
        title='Energy Trends Over Time',
        labels={'energy': 'Average Energy', 'year': 'Year', 'genre_group': 'Genre'}
    )

    fig.update_layout(
        template='ggplot2',
        paper_bgcolor='#f8f8f0',
        plot_bgcolor='#f8f8f0',
        font=dict(family='Helvetica, Arial, sans-serif', size=14, color='#333')
    )

    return fig

@app.callback(
    Output('cluster-subtitle', 'children'),
    Input('cluster-track-dropdown', 'value'),
    prevent_initial_call=True
)
def update_cluster_subtitle(track_index):
    if track_index is None:
        return ""
    return "These are the songs most similar in audio features to the selected track. They're determined using K-Means input features and proximity in the feature space."


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)