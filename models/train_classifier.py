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

from lightgbm import LGBMClassifier

import pickle
import bz2

from sqlalchemy import create_engine



def load_data(database_filepath):
    
    '''
    Function for load database from sql file. It divides database to two different sets (X and Y).
    
    Args:
        database_filepath (string): filepath of target database
    
    Returns:
        X (pandas.DataFrame): dataframe for train pipeline (features)
        Y (pandas.DataFrame): dataframe for train pipeline (target)
        category_name_list (list): list of disaster categories
    '''
    
    
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_name_list = df.columns[4:].tolist()
    return X, Y, category_name_list

def tokenize(text):
        
    '''
    Function for tokenizing text. Through the function, punctuations and url will removed from the text.
    Then, text will be tokenized as list
    
    Args:
        text (string): target text to tokenize
    
    Returns:
        token_list (list): list of words from tokenized text
    
    '''
    
    # normalize text and strip punctuation
    text = text.strip()
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    
    # replace url to urlplaceholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # tokenize text
    token_list = word_tokenize(text) 
    
    # lemmatize and return token list
    token_list = [WordNetLemmatizer().lemmatize(w) for w in token_list if w not in stopwords.words("english")]
    return token_list


def build_model():
    
    '''
    Function for making new pipeline with different an estimator. It optimizes pipeline with gridsearchCV and returns optimized pipeline
    
    Args:
        vect: (Scikit-learn) CountVectorizer
        tfidf: (Scikit-learn) TfidfTransformer
    
    Returns:
        pipeline (Pipeline): pipeline which is consisted with CountVectorizer, TfidfTransformer, TfidfVectorizer, TruncatedSVD
                             and MultiOutputClassifier with LGBMClassifier.
    
    '''
    pipeline = Pipeline([('features',FeatureUnion(
                      [('text_pipeline1', Pipeline([
                                          ('vect', CountVectorizer(tokenizer=tokenize, ngram_range = (1,2), max_df = 0.75)),
                                          ('tfidf', TfidfTransformer()),
                                          ('T_SVD',TruncatedSVD())])),
                       ('text_pipeline2', Pipeline([
                                          ('vect2',TfidfVectorizer()),
                                          ('T_SVD', TruncatedSVD()) ])) 
                      ])),

                    ('Multi_classifier',MultiOutputClassifier(LGBMClassifier(device='gpu', boosting_type='dart',
                                                                             gpu_platform_id = 0, gpu_device_id = 0)))
                    ])
    
    parameters_grid =  {
        'vect__ngram_range': [(1, 2)],
        'vect__max_df': (0.75, 1.0),
        'Multi_classifier__estimator__boosting_type' : ['dart'],
        'Multi_classifier__estimator__learning_rate' : (0.08,0.1)
        }


    grid_cv = GridSearchCV(pipeline, parameters_grid, cv = 3,  verbose = 20, n_jobs=-1)
    grid_cv.fit(X_train,y_train) 
    pipeline = grid_cv.best_estimator_
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_name_list):
    '''
    Function evaluate pipeline with predicted y value from pipeline 
    It shows classification_report by each category.
    
    Args:
        model (pipeline): machine learning pipeline
        X_test (pandas.DataFrame): dataframe of features to test
        Y_test (pandas.DataFrame): dataframe of target to test
        category_name_list (list): list of disaster categories
    
    '''
    
    Y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col].values, Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Function that saves a pipeline as pickle file.
    
    Args:
        model_name: name(version) of the pipeline
        pipeline: target pipeline to save
        
    '''
    
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