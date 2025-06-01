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

# Compute the correlation matrix
corr_matrix = data.corr(numeric_only=True)

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
@app.callback(
    Output('popularity-graph', 'figure'),
    Input('genre_group-dropdown', 'value'),
    Input('year-slider', 'value'), 
    Input('plot_type', 'value')
)
def update_graph(selected_genre, year_range, plot_type):
    print("Callback triggered!") 
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

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=7124)