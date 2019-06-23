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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_name_list = df.columns[4:].tolist()
    return X, Y, category_name_list

def tokenize(text):
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
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col].values, Y_pred[:,i]))


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