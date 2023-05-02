import pandas as pd
import dash
import dash_core_components as dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import json
import plotly.graph_objects as go



# Read data

df = pd.read_csv('./dataset/movies_dataset.csv', sep=';')

with open('./dataset/author_movie_dict.json', 'r') as fp:
    author_movie_dict = json.load(fp)


person_list = df['person'].astype(str).unique().tolist()
person_list.sort()

years = df['movie_year'].dropna().sort_values().unique()
years = sorted(years.tolist(), reverse=True)

# Top rated list
top_genre_by_count = df[['movie_genre','movie']].drop_duplicates()
top_genre_by_count['count'] = top_genre_by_count.groupby(['movie_genre']).transform('count')
top_genre = top_genre_by_count.sort_values(by=['count'], ascending=False).iloc[0]['movie_genre']

most_movies_by_year = df[['movie_year','movie']].drop_duplicates()
most_movies_by_year['count'] = most_movies_by_year.groupby(['movie_year']).transform('count')
most_movies_by_year = most_movies_by_year.sort_values(by=['count'], ascending=False).iloc[0]['movie_year']



#fig1
fig = go.Figure()

fig.add_trace(go.Treemap(
    ids = df.ids,
    labels = df.label,
    parents = df.parents,
    maxdepth=3,
    root_color="lightgrey",
    branchvalues='total',
    values=df.rank_rev,
    marker=dict(
        colors=df.budget,
        colorscale='sunsetdark',
        line=dict(width=1, color='grey'),
        cmin=1000000,
        cmid=200000000,
        cmax=4000000000,
        showscale=True,
        colorbar=dict(
            title="Budget",
            exponentformat='B',
            labelalias = {'4B': '4000M',
                          '3.5B': '3500M',
                          '3B': '3000M',
                          '2.5B': '2500M',
                          '2B': '2000M',
                          '1.5B': '1500M',
                          '1B': '1000M',
                          '0.5B': '500M'},
            thickness=60,
            len=0.5,
            y=0.5,
            ypad=0,
            ticklen=0,
            tickfont=dict(
                size=16,
                color="black"
            ),
            titlefont=dict(
                size=18,
                color="black"
            )


            ))
))

fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.update_traces(textinfo = "label+value+percent parent+percent entry"
                  , hovertemplate = "<b>%{label}</b><br><br>" +
                                    "Parent: %{parent}<br>" +
                                    "Total budget: %{color}<br>" +
                                    "Percent popularity rank of parent: %{percentParent:.2%}<br>" +
                                    "Percent popularity rank of entry: %{percentEntry:.2%}<br>" +
                                    "<extra></extra>"
                  , textfont = dict(size = 30)
                    , textposition = "middle center"
                    , texttemplate = "%{label}",

                  )

fig.update_layout(
    title={
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(
        family="Curier New, monospace",
        size=22,
        color="white")

    )

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})

fig.update_layout(legend=dict(font=dict(color='white'),
                   title=dict(font=dict(color='white'))),
                  height=800 )

fig.update_traces(marker=dict(colorbar=dict(
            title_font_size=18,
            tickfont_size=16,
)))





# Fig 2
df_movie = df[['movie', 'movie_year', 'movie_genre', 'budget', 'rank']].drop_duplicates()
df_budget = df[['movie','budget']].drop_duplicates()

budget_dict = dict(zip(df_budget.movie, df_budget.budget))

def find_budget(name):
    try:
        return budget_dict[name]

    except:
        return 0


df_movie['budget'] = df_movie['movie'].apply(find_budget)
df_top50 = df_movie.sort_values(by=['rank'], ascending=True).drop_duplicates('rank').iloc[:50]
df_sorted = df_top50.sort_values(by=['movie_genre'], ascending=True)
fig2 = px.scatter(df_sorted, x='movie_year', y='rank', color='movie_genre', size='budget', hover_data=['movie', 'budget', 'rank'], text='movie', size_max=90)


fig2.update_layout(
    title={
        'text': "Top 50 Movies by Rank (Budget vs Rank)",
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(
        family="Curier New, monospace",
        size=14,
        color="RebeccaPurple"

    ),
    height=800


)

fig2.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 0.95)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',

    },font=dict(
        family="Courier New, monospace",
        size=14,
        color="black"
)

)

fig2.update_layout(yaxis=dict(color='white'),
                   xaxis=dict(color='white'),
                   legend=dict(font=dict(color='white',size=20)),
                   title=dict(font=dict(color='white')))


fig2.update_layout(hoverlabel=dict(
    bgcolor="white",
    font_size=20,
    font_family="Curier New, monospace"
))





# Fig 3
genres_highst_rate_avg = df[['movie_genre','movie','rating']].drop_duplicates(subset=['movie', 'movie_genre']).dropna(subset=['movie', 'movie_genre'])
genres_highst_rate_avg['rating'] = genres_highst_rate_avg['rating'].astype(float)
genres_highst_rate_avg['avg_rating'] = genres_highst_rate_avg.groupby(['movie_genre'])['rating'].transform('mean')
genre_fig = genres_highst_rate_avg['movie_genre'].value_counts().to_frame().reset_index()

fig3 = px.pie(genre_fig, values='count', names='movie_genre', title=f'Movies Distribution by Genre',
              color_discrete_sequence=px.colors.sequential.RdBu,
              hover_data=['count', 'movie_genre'],
              labels={'index': 'Genre', 'movie_genre': 'Count'},
              hole=.2,
              height=500,
              width=500,
              opacity=0.8,
              custom_data=['movie_genre', 'count']
                )
fig3.update_traces(textposition='inside', textinfo='percent+label',showlegend=False)
fig3.update_layout(
    title={
        'text': "Movies Distribution by Genre",
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            family="Curier New, monospace",
            size=22,
            color="lightgrey"
        )},
    font=dict(
        family="Curier New, monospace",
        size=20,
        color="RebeccaPurple"
    )
)

fig3.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})



# Helper functions
def split_runtime_hours(df_sub):
    """
    Split runtime into hours and minutes
    """
    df_sub = str(df_sub)
    if len(df_sub.split(' ')) == 1:
        return 0
    else:
        runtime_hours = df_sub.split(' ')[0]
        runtime_hours = runtime_hours.replace('h', '')
        return runtime_hours


def split_runtime_minutes(df_sub):
    """
    Split runtime into hours and minutes
    """
    df_sub = str(df_sub)
    if len(df_sub.split(' ')) == 1:
        return 0
    else:
        runtime_minutes = df_sub.split(' ')[1]
        runtime_minutes = runtime_minutes.replace('m', '')
        return runtime_minutes

def movie_length(df):
    """
    Split runtime into hours and minutes
    """

    if df['total_minutes'] < 90:
        return 'below_90'
    elif (df['total_minutes'] >= 90) & (df['total_minutes'] < 120):
        return '90_120'
    elif (df['total_minutes'] >= 120) & (df['total_minutes'] < 150):
        return '120-150'
    elif (df['total_minutes'] >= 150) & (df['total_minutes'] < 180):
        return '150-180'
    else:
        return 'above-180'



# fig 4

df_movies_unique = df[['movie', 'movie_genre', 'run_time','rating', 'rank', 'rank_rev']].reset_index().drop_duplicates()
df_movies_unique['hour'] = df_movies_unique['run_time'].apply(split_runtime_hours)
df_movies_unique['minutes'] = df_movies_unique['run_time'].apply(split_runtime_minutes)
df_movies_unique['hour'].replace('Not', 0, inplace=True)
df_movies_unique['minutes'].replace('Available', 0, inplace=True)
df_movies_unique['total_minutes'] = df_movies_unique['hour'].astype(int)*60 + df_movies_unique['minutes'].astype(int)

df_movies_unique['below_90'] = df_movies_unique['total_minutes'] < 90
df_movies_unique['90_120'] = (df_movies_unique['total_minutes'] >= 90) & (df_movies_unique['total_minutes'] < 120)
df_movies_unique['120-150'] = (df_movies_unique['total_minutes'] >= 120) & (df_movies_unique['total_minutes'] < 150)
df_movies_unique['150-180'] = (df_movies_unique['total_minutes'] >= 150) & (df_movies_unique['total_minutes'] < 180)
df_movies_unique['above-180'] = df_movies_unique['total_minutes'] >= 180
df_movies_unique['movie_length'] = df_movies_unique.apply(movie_length, axis=1)

df_movies_unique['movie_length'].replace('below_90', '<90min', inplace=True)
df_movies_unique['movie_length'].replace('90_120', '90-120min', inplace=True)
df_movies_unique['movie_length'].replace('120-150', '120-150min', inplace=True)
df_movies_unique['movie_length'].replace('150-180', '150-180min', inplace=True)
df_movies_unique['movie_length'].replace('above-180', '>180min', inplace=True)

movie_len_cat = df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length', ascending=False)

fig4 = go.Figure(data=go.Scatterpolar(
  r=[i for i in df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length', ascending=False)['count']],
  theta=[i for i in df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length', ascending=False)['movie_length']],
  fill='toself',
    mode='markers+text',
    text=[i for i in df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length', ascending=False)['count']],
    marker=dict(
            color='grey',
            size=22),
        opacity=0.8,
        hovertemplate = '<b>Movie Length</b>: %{theta}<br><b>Movie count</b>: %{r}<extra></extra>',
        hoverlabel=dict(
            bgcolor="white",
            font_size=25,
            font_family="Curier New, monospace"
        ),
    name='Distribution of movies by length',

))

fig4.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=False

    ),
  ),
  showlegend=False
)
fig4.update_layout(
    title={
        'text': "Movies Distribution by Length",
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            family="Curier New, monospace",
            size=25,
            color="lightgrey"
        )},
    font=dict(
        family="Curier New, monospace",
        size=20,
        color="RebeccaPurple"
    )
)

fig4.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})


# fig 5
df_movies_unique_high_rate = df_movies_unique[df_movies_unique['rating'] == df_movies_unique['rating'].max()]
highest_rating = df_movies_unique_high_rate[['movie','rating','rank']].reset_index().sort_values(by='rank', ascending=True)
avg_rating = df_movies_unique['rating'].mean().round(1)

fig5 = go.Figure()

fig5.add_trace(go.Indicator(
    mode = "number+delta",
    value =highest_rating['rating'][0],
    domain = {'row': 2, 'column': 1}))

movie_name = f'{highest_rating["movie"][0]}<br><span style="font-size:0.6em;color:gray">Highest rated Movie VS Average</span><br>'
fig5.update_layout(
    grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'title': {"text": movie_name, 'font': {'size': 45}},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': avg_rating}}]
                         }})

fig5.update_layout(

    font=dict(
        family="Curier New, monospace",
        size=70,
        color="white"
    )
)

fig5.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})




# Create app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Create layout
app.layout = html.Div([

    dbc.Row([dbc.Col(
                    html.Div(html.H1('IMDB Top250 Movies Dataset - a visualization',
                             style={'text-align': 'center', 'color': 'white', 'padding-top': '10px','fontfamily': 'Curier New, monospace','size': '20px'})),md=10

    ),
        dbc.Col(
            html.Div(html.H1('',
                             style={'text-align': 'center', 'color': 'white', 'padding-top': '10px',
                                    'fontfamily': 'Curier New, monospace'})),md=2

        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([

                dbc.Col([
                    dcc.Loading(dcc.Graph(id='radar', figure=fig4, style={'width': '100%', 'height': '100%','background-color': 'black','padding-top': '100px'},
                                          config={
                                                'staticPlot': True, # True, False
                                          }
                                          ))
                ]),
                dbc.Col([
                    dbc.Alert('No movie available for this year', color='danger', dismissable=True, is_open=False, id='error_msg'),
                    dcc.Loading(dcc.Graph(id='top_movie', figure=fig5, style={'width': '100%', 'height': '100%','background-color': 'black','padding-top': '200px'}))
                ]),
                dbc.Col([
                    dcc.Loading(dcc.Graph(id='pie', figure=fig3, style={'width': '100%', 'height': '100%','background-color': 'black','padding-top': '100px'})),
                   ]),
                dbc.Row([
                    dbc.Row(dcc.Loading(dcc.Graph(id='treemap', figure=fig, style={'width': '100%', 'height': '100%'}))),
                    dbc.Row(dcc.Loading(dcc.Graph(id='bubble-graph', figure=fig2, style={'width': '100%', 'height': '100%'})))
            ]),
            ]),

            # dbc.Row([
            #     dbc.Col([
            #         html.Div(html.H1('Movie Dashboard',
            #             style={'text-align': 'center', 'color': 'white', 'padding-top': '10px'}))]),
            #     dbc.Col([
            #         html.Div(html.H1('Movie Dashboard',
            #                          style={'text-align': 'center', 'color': 'white', 'padding-top': '10px'}))]),
            # ]),

        ],md=10),
        dbc.Col([
            dbc.Row(html.Div(html.H1('Filter dataset',
                    style={'text-align': 'center', 'color': 'white', 'padding-top': '10px','fontfamily': 'Curier New, monospace','size': '20px'}))),
            dbc.Row([
                html.Div(html.H4('Select a person',
                    style={'text-align': 'left', 'color': 'white', 'padding-top': '10px','fontfamily': 'Curier New, monospace'})),
                dcc.Dropdown(
                id='dropdown',
                style={'width': '100%', 'font-size': '20px', 'color': 'grey','fontfamily': 'Curier New, monospace'},
                options=[i for i in person_list],
                placeholder="Select a person, default None",
                value=None,
                multi=False

            )]),
            dbc.Row([
                html.Div(html.H4('Select a years',
                    style={'text-align': 'left', 'color': 'white', 'padding-top': '10px','fontfamily': 'Curier New, monospace'})),
                dcc.Dropdown(
                id='dropdown2',
                style={'width': '100%', 'font-size': '20px', 'color': 'grey','fontfamily': 'Curier New, monospace'},
                options=[i for i in years],
                placeholder="Select a years, default All",
                value=None,
                multi=True

            )]),
            # dbc.Row([
            #     html.Div(html.H4('Publish tree map by: ',
            #         style={'text-align': 'left', 'color': 'white', 'padding-top': '350px'})),
            #     dcc.RadioItems(
            #     id='radio',
            #     style={'width': '100%', 'font-size': '20px', 'color': 'grey'},
            #     options=['genre', 'year', 'position'],
            #     value='genre'
            #
            #     )]),
        ],md=2,style={'text-align': 'left', 'color': 'white', 'padding-top': '100px'}),

], style={'width': '100%', 'height': '1400px' ,'background-color': 'black'})

],style={'background-color': 'black','height': '2500px'})

# Create callback
@app.callback(
    [Output('treemap', 'figure'),
     Output('bubble-graph', 'figure'),
     Output('pie', 'figure'),
     Output('radar', 'figure'),
     Output('top_movie', 'figure'),
     Output('error_msg', 'is_open'),
     Output('dropdown2', 'options'),
     Output('dropdown', 'options'),
    Output('dropdown', 'value'),
    Output('dropdown2', 'value')],

    [Input('dropdown', 'value'),
     Input('dropdown2', 'value'),
     Input('dropdown', 'options'),
    Input('dropdown2', 'options')])




def update_graph(person, year, person_list_in, years_list_in):


    df_sub = df
    error_msg = False
    select_years = years_list_in
    person_list_update = person_list_in
    person_value = person
    year_list_person = None
    year_value = year
    # if radio == 'genre':
    #     df_sub = df.groupby(['genre', 'year', 'name','position', 'budget','person','rank_rev'])

    # path = {'genre': ['genre', 'year', 'name', 'position'],
    #         'year': ['year', 'genre', 'name', 'position'],
    #         'position': ['position', 'genre', 'year', 'name']}

    # click_data = {}
    # if clickData is not None:
    #     click_data = clickData['points'][0]

    if person is None or person == []:
        select_years = years
        select_person = person_list
        # path = {'genre': ['genre', 'year', 'name', 'person', 'position'],
        #         'year': ['year', 'genre', 'name', 'person', 'position'],
        #         'position': ['position', 'genre', 'year', 'name', 'person']}
        df_sub = df
    else:
        person_name = person

        genre_list_person = set()
        year_list_person = set()
        movie_list_person = set()
        for i in author_movie_dict[person_name]:
            genre_list_person.add(i['movie_genre'])
            year_list_person.add(i['movie_year'])
            movie_list_person.add(i['movie'])

        print(genre_list_person)
        print(year_list_person)
        print(movie_list_person)
        if year is None or year == []:
            df_movie_filter = df_sub[df_sub['movie_genre'].isin(genre_list_person) &
                                 (df_sub['movie_year'].isin(year_list_person) |
                                  df_sub['movie_year'].isnull()) &
                                 (df_sub['movie'].isin(movie_list_person) |
                                  df_sub['movie'].isnull())]

        else:
            df_movie_filter = df_sub[df_sub['movie_genre'].isin(genre_list_person) &
                                 (df_sub['movie_year'].isin(year) |
                                  df_sub['movie_year'].isnull()) &
                                 (df_sub['movie'].isin(movie_list_person) |
                                  df_sub['movie'].isnull())&(
                                    df_sub['movie'].isin(year)
                                     )]


        select_years = list(year_list_person)

        # Update rank-rev values to fit screen
        for i in genre_list_person:
            try:
                df_movie_filter.loc[df_movie_filter['ids'] == i, 'rank_rev'] = \
                df_movie_filter.loc[df_movie_filter['ids'] == i, 'rank_rev'].values[0] / 1500
            except:
                pass


        df_sub = df_movie_filter
        #df_sub.to_csv('df_sub.csv', sep=';',index=False)
        #df_sub = df_sub[df_sub['person'] == person]

    if year is None or year == []:
        if person is None or person == []:
            select_years = years
        df_sub = df_sub

    else:
        df_year_filter = df[df['movie_year'].isin(year) | df['movie_year'].isnull()]

        if person is None or person == []:
            select_years = years
            df_sub = df_year_filter
        else:
            person_name = person

            genre_list_person = set()
            year_list_person = set()
            movie_list_person = set()
            for i in author_movie_dict[person_name]:
                genre_list_person.add(i['movie_genre'])
                year_list_person.add(i['movie_year'])
                movie_list_person.add(i['movie'])

            select_years = list(year_list_person)
            print(genre_list_person)
            print(year_list_person)
            print(movie_list_person)

            df_movie_filter = df_year_filter[df_year_filter['movie_genre'].isin(genre_list_person) &
                                     (df_year_filter['movie_year'].isin(year_list_person) |
                                      df_year_filter['movie_year'].isnull()) &
                                     (df_year_filter['movie'].isin(movie_list_person) |
                                      df_year_filter['movie'].isnull())]


            df_sub = df_movie_filter





    fig = go.Figure()

    fig.add_trace(go.Treemap(
        ids=df_sub.ids,
        labels=df_sub.label,
        parents=df_sub.parents,
        maxdepth=3,
        root_color="lightgrey",
        branchvalues='total',
        values=df_sub.rank_rev,


        marker=dict(
            colors=df_sub.budget,
            colorscale='sunsetdark',
            line=dict(width=1, color='grey'),
            cmin=1000000,
            cmid=200000000,
            cmax=4000000000,
            showscale=True,
            colorbar=dict(
                title="Budget",
                exponentformat='B',
                labelalias={'4B': '4000M',
                            '3.5B': '3500M',
                            '3B': '3000M',
                            '2.5B': '2500M',
                            '2B': '2000M',
                            '1.5B': '1500M',
                            '1B': '1000M',
                            '0.5B': '500M'},
                thickness=60,
                len=0.5,
                y=0.5,
                ypad=0,
                ticklen=0,
                tickfont=dict(
                    size=20,
                    color="white"
                ),
                titlefont=dict(
                    size=30,
                    color="white"
                )

            ))
    ))

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.update_traces(textinfo="label+value+percent parent+percent entry"
                      , hovertemplate="<b>%{label}</b><br><br>" +
                                      "Parent: %{parent}<br>" +
                                      "Total budget: %{color}<br>" +
                                      "Percent popularity rank of parent: %{percentParent:.2%}<br>" +
                                      "Percent popularity rank of entry: %{percentEntry:.2%}<br>" +
                                      "<extra></extra>"
                      , textfont=dict(size=30)
                      , textposition="middle center"
                      , texttemplate="%{label}",

                      )
    fig.update_traces(pathbar = {'visible': True, 'thickness': 20, 'textfont': {'size': 20, 'color': 'black'}})

    fig.update_traces(marker=dict(colorbar=dict(
        title_font_size=22,
        tickfont_size=20,
    )))

    if person is None or person == []:
        fig.update_layout(title_text='The Movie Shelf: A Visualisation of the Most Popular Movies of All Time by Genre and Year',
                          font=dict(
                              family="Courier New, monospace",
                              size=25,
                              color="white"
                          )
                          )
    else:
        fig.update_layout(title_text=f'The Movie Shelf: A Visualisation of the Most Popular Movies of All Time by Genre and Year for {person_name}',
                          font=dict(
                                family="Courier New, monospace",
                                size=22,
                                color="white"
                          )
                          )
    fig.update_layout(
        title={
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    fig.update_layout(legend=dict(font=dict(color='white')),
                      title=dict(font=dict(color='white')),
                      height=800)

    fig.update_traces(marker_depthfade=False, selector=dict(type='treemap'),
                      insidetextfont=dict(size=30, color='black'),
                      )



    # fig 2 bubble chart

    df_movie = df_sub[['movie', 'movie_year', 'movie_genre', 'budget', 'rank']].drop_duplicates()
    df_budget = df[['movie', 'budget']].drop_duplicates()

    budget_dict = dict(zip(df_budget.movie, df_budget.budget))

    def find_budget(name):
        try:
            return budget_dict[name]

        except:
            return 0

    df_movie['budget'] = df_movie['movie'].apply(find_budget)

    df_top50 = df_movie.sort_values(by=['rank'], ascending=True).drop_duplicates('rank').iloc[:50]
    df_sorted = df_top50.sort_values(by=['movie_genre'], ascending=True)
    fig2 = px.scatter(df_sorted, x='movie_year', y='rank', color='movie_genre', size='budget',
                      hover_data=['movie', 'budget', 'rank'], text='movie', size_max=90)


    if person is None or person == []:
        fig2.update_layout(title_text=f'Top 50 Movies by Rank (Budget vs Rank) of All Time',
                           font=dict(
                               family="Courier New, monospace",
                               size=22,
                               color="white"
                           )
                           )
        fig2.update_yaxes(range=[0, 60])
    else:
        fig2.update_layout(title_text=f'Top Movies by Rank (Budget vs Rank) for {person} of All Time ',
                           font=dict(
                               family="Courier New, monospace",
                               size=22,
                               color="white"
                           )
                           )
        fig2.update_yaxes(range=[0, 252])

    if year is None or year == [] :
        fig2.update_yaxes(range=[0, 60])
    else:
        fig2.update_yaxes(range=[0, 252])

    if year is None or year == [] and person is None or person == []:
        fig2.update_yaxes(range=[0, 60])
    else:
        fig2.update_yaxes(range=[0, 252])


    fig2.update_layout(
        title={
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        height=800

    )

    fig2.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 0.95)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',

    }, font=dict(
        family="Courier New, monospace",
        size=18,
        color="black"
    )

    )

    fig2.update_layout(yaxis=dict(color='white', title='Rank', titlefont=dict(color='white', size=30)),
                       xaxis=dict(color='white', title='Year', titlefont=dict(color='white', size=30)),
                       legend=dict(font=dict(color='white',size=18),title='Genre'),
                       title=dict(font=dict(color='white',size=40)))

    fig2.update_layout(hoverlabel=dict(
        bgcolor="white",
        font_size=20,
        font_family="Curier New, monospace"
    ))




    # fig 3 genres pie chart

    genres_highst_rate_avg = df_sub[['movie_genre', 'movie', 'rating']].drop_duplicates(subset=['movie', 'movie_genre']).dropna(subset=['movie', 'movie_genre'])
    genres_highst_rate_avg['rating'] = genres_highst_rate_avg['rating'].astype(float)
    genres_highst_rate_avg['avg_rating'] = genres_highst_rate_avg.groupby(['movie_genre'])['rating'].transform('mean')
    genre_fig = genres_highst_rate_avg['movie_genre'].value_counts().to_frame().reset_index()

    fig3 = px.pie(genre_fig, values='count', names='movie_genre', title=f'Movies Distribution by Genre',
                  color_discrete_sequence=px.colors.sequential.RdBu,
                  hover_data=['count', 'movie_genre'],
                  labels={'index': 'Genre', 'movie_genre': 'Count'},
                  hole=.2,
                  height=500,
                  width=500,
                  opacity=0.8,
                  custom_data=['movie_genre', 'count']
                  )
    fig3.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
    fig3.update_layout(
        title={
            'text': "Movies Distribution by Genre",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=22,
                color="grey"
            )},
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="RebeccaPurple"
        )
    )

    fig3.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # fig 4
    df_movies_unique = df_sub.drop_duplicates(subset=['rank']).dropna(subset=['rank'])
    df_movies_unique['hour'] = df_movies_unique['run_time'].apply(split_runtime_hours)
    df_movies_unique['minutes'] = df_movies_unique['run_time'].apply(split_runtime_minutes)
    df_movies_unique['hour'].replace('Not', 0, inplace=True)
    df_movies_unique['minutes'].replace('Available', 0, inplace=True)
    df_movies_unique['total_minutes'] = df_movies_unique['hour'].astype(int) * 60 + df_movies_unique['minutes'].astype(
        int)

    df_movies_unique['below_90'] = df_movies_unique['total_minutes'] <= 90
    df_movies_unique['90_120'] = (df_movies_unique['total_minutes'] > 90) & (df_movies_unique['total_minutes'] < 120)
    df_movies_unique['120-150'] = (df_movies_unique['total_minutes'] >= 120) & (df_movies_unique['total_minutes'] < 150)
    df_movies_unique['150-180'] = (df_movies_unique['total_minutes'] >= 150) & (df_movies_unique['total_minutes'] < 180)
    df_movies_unique['above-180'] = df_movies_unique['total_minutes'] >= 180
    df_movies_unique['movie_length'] = df_movies_unique.apply(movie_length, axis=1)

    df_movies_unique['movie_length'].replace('below_90', '<90min', inplace=True)
    df_movies_unique['movie_length'].replace('90_120', '90-120min', inplace=True)
    df_movies_unique['movie_length'].replace('120-150', '120-150min', inplace=True)
    df_movies_unique['movie_length'].replace('150-180', '150-180min', inplace=True)
    df_movies_unique['movie_length'].replace('above-180', '>180min', inplace=True)

    movie_len_cat = df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(
        by='movie_length', ascending=False)

    fig4 = go.Figure(data=go.Scatterpolar(
        r=[i for i in
           df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length',
                                                                                                ascending=False)[
               'count']],
        theta=[i for i in
               df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length',
                                                                                                    ascending=False)[
                   'movie_length']],
        fill='toself',
        mode='markers+text',
        text=[i for i in
              df_movies_unique['movie_length'].value_counts().to_frame().reset_index().sort_values(by='movie_length',
                                                                                                   ascending=False)[
                  'count']],
        marker=dict(
            color='grey',
            size=25),
        opacity=0.8,
        hovertemplate='<b>Movie Length</b>: %{theta}<br><b>Movie count</b>: %{r}<extra></extra>',
        hoverlabel=dict(
            bgcolor="white",
            font_size=22,
            font_family="Curier New, monospace"
        ),
        name='Distribution of movies by length',
    ))

    fig4.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False

            ),
        ),
        showlegend=False
    )

    fig4.update_layout(
        title={
            'text': "Movies Distribution by Length",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Courier New, monospace",
                size=22,
                color="grey"
            )},
        font=dict(
            family="Courier New, monospace",
            size=20,
            color="lightgrey"
        )
    )

    fig4.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # fig 5
    df_movies_unique = df_sub.drop_duplicates(subset=['rank'])
    df_movies_unique['rating'] = df_movies_unique['rating'].astype(float)
    df_movies_unique_high_rate = df_movies_unique[df_movies_unique['rating'] == df_movies_unique['rating'].max()]
    highest_rating = df_movies_unique_high_rate[['movie','rating','rank','person']].reset_index().sort_values(by='rank', ascending=True)
    avg_rating = df_movies_unique['rating'].mean()



    fig5 = go.Figure()

    fig5.add_trace(go.Indicator(
        mode="number+delta",
        value=highest_rating['rating'][0],
        name=highest_rating["movie"][0],
        domain={'row': 1, 'column': 1}))
    try:
        if person is not None and person != []:
            m_name = highest_rating["movie"][0]
            if len(m_name) > 20:
                m_name = f'<span style="font-size:0.6em;color:gray">{m_name}</span>' \
                         f'<br><span style="font-size:0.4em;color:gray">Highest rated Movie VS Average of all\n{person} Movies</span><br><span style="font-size:0.6em;color:gray"></span>'
            else:
                m_name = highest_rating["movie"][0]
            movie_name = m_name
        else:
            movie_name = f'{highest_rating["movie"][0]}<br><span style="font-size:0.6em;color:gray">Highest rated Movie VS Average of all Movies</span><br>'
    except:
        movie_name = f'No Movie with Highest Rating<br><span style="font-size:0.4em;color:gray">Highest rated Movie VS Average</span><br>'
    fig5.update_layout(
        grid={'rows': 1, 'columns': 1, 'pattern': "independent"},
        template={'data': {'indicator': [{
            'title': {"text": movie_name, 'font': {'size': 45}},
            'mode': "number+delta+gauge",
            'customdata': [highest_rating["movie"][0]],
            'delta': {'reference': avg_rating}}]
        }})

    fig5.update_layout(

        font=dict(
            family="Courier New, monospace",
            size=70,
            color="white"
        )
    )

    fig5.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })





    if years_list_in is not years:
        if years_list_in is not []:
            movie_list_year = df_sub[df_sub['year'].isin(years_list_in)]['movie'].unique()
            person_movie_year = df_sub[df_sub['movie'].isin(movie_list_year)]['person'].dropna().unique().tolist()
            person_list_update = sorted(person_movie_year)
            select_years = sorted(years_list_in,reverse=True)
        else:
            person_list_update = sorted(person_list)
            select_years = sorted(years,reverse=True)

        if person_list_in is not person and years_list_in is not years:

            if person is not [] and person is not None:
                person_list_update = sorted(person_list_in)

                person_name = person
                genre_list_person = set()
                year_list_person = set()
                movie_list_person = set()
                for i in author_movie_dict[person_name]:
                    genre_list_person.add(i['movie_genre'])
                    year_list_person.add(i['movie_year'])
                    movie_list_person.add(i['movie'])

                print(genre_list_person)
                print(year_list_person)
                print(movie_list_person)
                if year is None or year == []:
                    df_movie_filter = df_sub[df_sub['movie_genre'].isin(genre_list_person) &
                                             (df_sub['movie_year'].isin(year_list_person) |
                                              df_sub['movie_year'].isnull()) &
                                             (df_sub['movie'].isin(movie_list_person) |
                                              df_sub['movie'].isnull())]

                else:
                    df_movie_filter = df_sub[df_sub['movie_genre'].isin(genre_list_person) &
                                             (df_sub['movie_year'].isin(year) |
                                              df_sub['movie_year'].isnull()) &
                                             (df_sub['movie'].isin(movie_list_person) |
                                              df_sub['movie'].isnull()) & (
                                                 df_sub['movie'].isin(year)
                                             )]

                select_years = list(year_list_person)

    if person_value not in person_list and year is not years:

        if year is None:
            pass
        else:
            if year is []:

                if person == person_value:
                    print('Detexcted_not known')
                    select_years = sorted(years,reverse=True)

                else:
                    select_years = sorted(years_list_in, reverse=True)
            select_years = sorted(years, reverse=True)
            #person_list_update = sorted(person_list)
            print('Inside')



    if year == [] :
        print('Inside none none0')
        person_list_update = sorted(person_list)

    if year == []:
        if person is None:
            fig.update_traces(branchvalues='total')
            fig2.update_yaxes(range=[0, 60])
        else:
            fig.update_traces(branchvalues='remainder')
        if person == []:
            fig.update_traces(branchvalues='remainder')

    if year is None:
        if person is None:
            fig.update_traces(branchvalues='total')
            fig2.update_yaxes(range=[0, 60])

        else:
            fig.update_traces(branchvalues='remainder')

    if year is None and person is None:
        select_years = sorted(years, reverse=True)
        person_list_update = sorted(person_list)

    if type(person) is str:
        fig.update_traces(branchvalues='remainder')
        fig2.update_yaxes(range=[0, 252])

    # if person is None :
    #     if year == []:
    #         fig.update_traces(branchvalues='total')
    #         fig2.update_yaxes(range=[0, 252])
    #     else:
    #         fig.update_traces(branchvalues='remainder')

    # if person is []:
    #     fig.update_traces(branchvalues='total')
    #     fig2.update_yaxes(range=[0, 252])
    # else:
    #     fig.update_traces(branchvalues='remainder')

    # if year is None and person is None:
    #     fig.update_traces(branchvalues='total')
    #     fig2.update_yaxes(range=[0, 60])



    print(year,person, person_value)
    return fig,fig2,fig3,fig4,fig5,error_msg,select_years,person_list_update,person_value,year_value


if __name__ == '__main__':
    app.run_server(debug=True)