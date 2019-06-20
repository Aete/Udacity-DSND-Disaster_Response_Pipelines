import sys
import pandas as pd
import numpy as np

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD 


import pickle
import bz2

from sqlalchemy import create_engine



def load_data(database_filepath):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    df.head()


def tokenize(text):
    # normalize text and strip punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    token_list = word_tokenize(text) 
    token_list = [w for w in token_list if w not in stopwords.words("english")]
    
    # lemmatize and return token list
    token_list = [WordNetLemmatizer().lemmatize(w) for w in token_list]
    return token_list


def build_model():
    pipeline = Pipeline([('features',FeatureUnion(
                      [('text_pipeline1', Pipeline([
                                          ('vect', vect),
                                          ('tfidf', tfidf),
                                          ('T_SVD',TruncatedSVD())])),
                       ('text_pipeline2', Pipeline([
                                          ('vect2',TfidfVectorizer()),
                                          ('T_SVD', TruncatedSVD()) ])) 
                      ])),

                    ('Multi_classifier',MultiOutputClassifier(LGBMClassifier(device='gpu', boosting_type='dart',
                                                                             gpu_platform_id = 0, gpu_device_id = 0)))
                    ])
    parameters_grid =  {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'Multi_classifier__estimator__boosting_type' : ['dart'],
        'Multi_classifier__estimator__learning_rate' : (0.08,0.1)
        }
    GridSearchCV(pipeline, parameters_grid, cv = 3,  verbose = 20, n_jobs=-1)
    pipeline = grid_cv.best_estimator_
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print(col)
        print(classification_report(y_test[col].values, y_pred[:,i]))


def save_model(model, model_filepath):
    url = model_filepath
    f = bz2.BZ2File(url, 'wb')
    pickle.dump(model, f, protocol=2)
    f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()