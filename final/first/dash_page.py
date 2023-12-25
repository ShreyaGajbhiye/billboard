from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

grammy_data_path = '../datasets/grammySongs_1999-2019.csv'
spotify_data_path = '../datasets/songAttributes_1999-2019.csv'
billboard_data_path = '../datasets/billboardHot100_1999-2019.csv'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def read_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def preprocess_grammy_data():
    grammy = read_data(grammy_data_path)
    grammy['Artist'] = grammy['Artist'].str.replace(" &", ",")
    grammy['Name'] = grammy['Name'].str.lower()
    grammy['Artist'] = grammy['Artist'].str.lower()
    return grammy


def preprocess_spotify_data():
    spotify = read_data(spotify_data_path)
    spotify = spotify.drop(columns='Unnamed: 0')
    spotify['Artist'] = spotify['Artist'].str.strip("[]")
    spotify['Artist'] = spotify['Artist'].str.replace("'", "").str.replace(" &", ",")
    spotify.rename(columns={'name': 'Name'}, inplace=True)
    spotify['Name'] = spotify['Name'].str.lower()
    spotify['Artist'] = spotify['Artist'].str.lower()
    return spotify


def preprocess_billboard_data():
    bboard = read_data(billboard_data_path)
    bboard = bboard.drop(columns='Unnamed: 0')
    bboard.rename(columns={'Artists': 'Artist'}, inplace=True)
    bboard['Artist'] = bboard['Artist'].str.replace(" &", ",")
    bboard['Name'] = bboard['Name'].str.lower()
    bboard['Artist'] = bboard['Artist'].str.lower()
    # bboard = bboard.reset_index(drop=True)
    return bboard

def grammy_spotify_billboard():
    grammy = preprocess_grammy_data()
    spotify = preprocess_spotify_data()
    billboard = preprocess_billboard_data()
    songs = spotify.groupby(['Name', 'Artist'], as_index=False).agg({'Acousticness': 'mean', 'Danceability': 'mean', 'Duration': 'mean', 'Energy': 'mean', 'Explicit': 'max', 'Instrumentalness': 'mean', 'Liveness': 'mean', 'Loudness': 'mean', 'Mode': 'max', 'Popularity': 'sum', 'Speechiness': 'mean', 'Tempo': 'mean', 'Valence': 'mean'})
    gr = grammy.merge(songs, on=['Name', 'Artist'])
    gr = gr.drop(columns = 'Unnamed: 0').drop(columns = 'X')
    bb1 = billboard.groupby(['Name', 'Artist', 'Week', 'Weekly.rank'], as_index = False).agg({'Weeks.on.chart' : 'max', 'Peak.position' : 'min', 'Genre' : 'first', 'Date':'first'})
    bb1 = bb1.merge(songs, on = ['Name', 'Artist'])
    bb2 = bb1.groupby(['Name','Artist'], as_index = False).agg({'Weeks.on.chart' : 'max', 'Peak.position' : 'min'})
    bb2 = bb2.dropna(subset = ['Peak.position', 'Weeks.on.chart'])
    bb3 = bb1.groupby(['Name','Artist'], as_index = False).agg({'Acousticness' : 'mean', 'Danceability' : 'mean', 'Duration' : 'mean', 'Energy' : 'mean', 'Explicit' : 'max', 'Instrumentalness' : 'mean',  'Liveness' : 'mean',  'Loudness' : 'mean',  'Mode' : 'max', 'Speechiness' : 'mean', 'Tempo' : 'mean', 'Valence' : 'mean'})
    songs['Name'] = songs['Name'].str.title()
    songs['Artist'] = songs['Artist'].str.title()
    gr['Name'] = gr['Name'].str.title()
    gr['Artist'] = gr['Artist'].str.title()
    bb1['Name'] = bb1['Name'].str.title()
    bb1['Artist'] = bb1['Artist'].str.title()
    bb2['Name'] = bb2['Name'].str.title()
    bb2['Artist'] = bb2['Artist'].str.title()
    bb3['Name'] = bb3['Name'].str.title()
    bb3['Artist'] = bb3['Artist'].str.title()
    bb3['Loudness'] = bb3['Loudness']/60 + 1
    songs['Loudness'] = songs['Loudness']/60 + 1
    gr['Loudness'] = gr['Loudness']/60 + 1
    return songs, gr, bb1, bb2, bb3


# Layout of the app
app.layout = html.Div([
    html.H1("Music Trend Analysis with Dash", className='mb-2', style={'textAlign':'center'}),
    html.Label("Select a Song Feature:"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='feature-dropdown',
                value='Acousticness',
                options=[
                {'label': 'Acousticness', 'value': 'Acousticness'},
                {'label': 'Danceability', 'value': 'Danceability'},
                {'label': 'Energy', 'value': 'Energy'},
                {'label': 'Popularity', 'value': 'Popularity'},
                {'label': 'Mode', 'value': 'Mode'},
                {'label': 'Speechiness', 'value': 'Speechiness'},
                {'label': 'Tempo', 'value': 'Tempo'},
                {'label': 'Valence', 'value': 'Valence'},
                {'label': 'Loudness', 'value': 'Loudness'},
                {'label': 'Liveness', 'value': 'Liveness'},
                {'label': 'Instrumentalness', 'value': 'Instrumentalness'},
                {'label': 'Explicit', 'value': 'Explicit'},
                {'label': 'Duration', 'value': 'Duration'}],
                # options=df.columns[1:]
                )], width=4)
        ]), dcc.Graph(id='boxplot'),

        dbc.Row([
        dbc.Col([
            html.Img(id='bar-graph-matplotlib')
        ], width=12)
    ]),
])

# Callback to update the boxplot based on the selected feature
@app.callback(
    Output(component_id='bar-graph-matplotlib', component_property='src'),
    Output('boxplot', 'figure'),
    Input('feature-dropdown', 'value')
)

def plot_data(selected_yaxis):
    songs, gr, bb1, bb2, bb3 = grammy_spotify_billboard()
    fig = plt.figure(figsize=(14, 5))
    genre_counts = gr['Genre'].value_counts()
    data = {
    'Genre': list(genre_counts.keys()),
    'Count': list(genre_counts.values)
    }
    genre_counts = pd.DataFrame(data)
    genre_counts.loc[genre_counts['Genre'].isin(['Dance/Electronic Music', 'American Roots Music', 'Gospel/Contemporary Christian Music']), 'Genre'] = 'Other'
    genre_counts = genre_counts.groupby('Genre')['Count'].sum().reset_index()
    plt.pie(genre_counts['Count'], labels=genre_counts['Genre'], autopct='%1.1f%%', startangle=80)
    plt.title('Distribution of Award-Winning Songs Across Different Genres')
    
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    return fig_bar_matplotlib

def update_boxplot(selected_feature):
    songs, gr, bb1, bb2, bb3 = grammy_spotify_billboard()
    top_genres = list(bb1.Genre.value_counts().sort_values(ascending=False).head(10).keys())
    filtered_df = bb1[bb1['Genre'].apply(lambda x: any(genre == x for genre in top_genres))]
    filtered_df = filtered_df[filtered_df.iloc[:, :2].duplicated() == False]
    fig = px.box(filtered_df, x='Genre', y=selected_feature, title=f'{selected_feature} Distribution by Genre')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)