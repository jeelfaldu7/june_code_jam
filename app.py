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
from dash import Dash, html, dcc, callback, Output, Input

# Load the dataset and read the data correctly
data = pd.read_csv('datasets/spotify.csv')

# Clean column names: replace spaces with underscores, remove special characters, and convert to lowercase
data.columns = [re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', col)).lower() for col in data.columns]

# Strip commas from length_duration entries that have them
data['length_duration'] = data['length_duration'].str.replace(',', '')

# Convert 'length_duration' column to integer data type
data['length_duration'] = data['length_duration'].astype(int)

app = Dash()

# App layout
app.layout = html.Div([
    html.H1('Spotify Dashboard 🎵'),
    html.Label('Select Genre:'),
    dcc.Dropdown(
        id='top_genre-dropdown',
        options=[{'label': genre, 'value': genre} for genre in data['top_genre'].unique()],
        value=data['top_genre'].unique()[0]
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
    dcc.Graph(id='popularity-graph'),
    html.Label('Select Plot Type:'),
    dcc.RadioItems(
        id='plot_type',
        options=[
            {'label': 'Scatter Plot', 'value':'scatter'},
            {'label': 'Box Plot', 'value': 'box'}
        ],
        value='scatter', 
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
])

# Callbacks for interactivity
@app.callback(
    Output('popularity-graph', 'figure'),
    Input('top_genre-dropdown', 'value'),
    Input('year-slider', 'value'), 
    Input('plot_type', 'value')
)
def update_graph(selected_genre, year_range, plot_type):
    filtered_data = data[(data['top_genre'] == selected_genre) &
                      (data['year'] >= year_range[0]) &
                      (data['year'] <= year_range[1])]
    
    # Plot logic depending on the selected plot type
    if plot_type == 'scatter':
        fig = px.scatter(
            filtered_data,
            x='danceability',
            y='popularity',
            color='energy',
            hover_data=['artist', 'title', 'year']
        )
    elif plot_type == 'box':
        fig = px.box(
            filtered_data,
            x='year',
            y='popularity',
            color='energy',
            points='all',  # show all points
            hover_data=['artist', 'title', 'year']
        )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)