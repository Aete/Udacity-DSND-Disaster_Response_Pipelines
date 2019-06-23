import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    categories = df.columns[4:]
    categories_top10 = df[categories].sum(axis=0).sort_values(ascending=False).index[:10] 
    df_top10_counts_direct = df[df['genre']=='direct'][categories_top10].sum(axis=0).sort_values(ascending=False).values[:10] 
    df_top10_counts_news = df[df['genre']=='news'][categories_top10].sum(axis=0).sort_values(ascending=False).values[:10] 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph_1 = {
            'data': [
                Bar(
                    name='Direct',
                    x=categories_top10,
                    y=df_top10_counts_direct,
                    text=["{:,}".format(x) for x in df_top10_counts_direct],
                    textposition = "outside",
                    hoverinfo = 'text',
                    textfont = dict(family='Arial, sans-serif', size=12,color='black'),
                    marker=dict(color='rgb(244,67,54)')
                ),
                Bar(
                    name='News',
                    x=categories_top10,
                    y=df_top10_counts_news,
                    text=["{:,}".format(x) for x in df_top10_counts_news],
                    textposition = "outside",
                    textfont = dict(family='Arial, sans-serif', size=12,color='black'),
                    marker=dict(color='rgb(33,150,243)')
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Categories (Top 10)',
                'titlefont':dict(family='Arial, sans-serif', size=18,color='black') ,
                'height':600,
                'yaxis': {
                    'tickfont': dict(family='Arial, sans-serif', size=14,color='black'),
                    'showgrid' : True,
                    'dtick':2000,
                    'range':[0, 12000]
                },
                'xaxis': {
                    'tickangle' : 0,
                    'tickfont': dict(family='Arial, sans-serif', size=14,color='black')
                }
            }
        }
    
    graph_2 = {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres (Pie chart)',
                'titlefont':dict(family='Arial, sans-serif', size=18,color='black') 
 
            }
        }
    graph_3 = {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres(Bar chart)',
                'titlefont':dict(family='Arial, sans-serif', size=18,color='black') ,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    
    graphs = [graph_2,graph_3,graph_1]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host=app.config.get("HOST", "localhost"),port=app.config.get("PORT", 9000))

if __name__ == '__main__':
    main()
