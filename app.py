# Import necessary libraries
import re
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')

from dash import Dash, html, dcc, callback, Output, Input

import plotly.express as px

# Load the dataset and read the data correctly
data = pd.read_csv('datasets/spotify.csv')

# Clean column names: replace spaces with underscores, remove special characters, and convert to lowercase
data.columns = [re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', col)).lower() for col in data.columns]

# Strip commas from length_duration entries that have them
data['length_duration'] = data['length_duration'].str.replace(',', '')

# Convert 'length_duration' column to integer data type
data['length_duration'] = data['length_duration'].astype(int)

demo_app = Dash()

demo_app.layout = [
    html.H1(children='Soon-To-Be-Titled Spotify Analyzer', style={'textAlign':'center'}),
    dcc.Dropdown(data['year'].unique(), value=2004, id='dropdown-selection'),
    dcc.Graph(id='graph-content')
]

@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value')
)
def update_graph(value):
    dff = data[data['year']==value].sort_values(by='beats_per_minute_bpm', axis=0)
    return px.line(dff, x='beats_per_minute_bpm', y='popularity')

if __name__ == '__main__': 
    demo_app.run(debug=True, port=7124)