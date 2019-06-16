# Udacity-DSND-Disaster_Response_Pipelines
Analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

## Overview (from Udacity DSND program) 
1. app
    - template
      - master.html: main page of web app
      - go.html: classification result page of web app
    - run.py: Flask file that runs app

2. data
    - disaster_categories.csv: data to process 
    - disaster_messages.csv: data to process
    - process_data.py
    - InsertDatabaseName.db: database to save clean data to

3. models
    - train_classifier.py
    - classifier.pkl: saved model 

- README.md


## Notebooks (from Udacity DSND program) 

### 1. ETL Pipeline Preparation.ipynb
The first part of data pipeline is the Extract, Transform, and Load process.
This notebook is for reading the dataset, clean the data, and then store it in a SQLite database.
To load the data into an SQLite database, this notebook use the pandas dataframe .to_sql() method, which you can use 
with an SQLAlchemy engine.

### 2. ML Pipeline Preparation.ipynb
For the machine learning portion, this notebook split the data into a training set and a test set.
Then, this will build a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output
a final model that uses the message column to predict classifications for 36 categories (multi-output classification).
Finally, this will export your model to a pickle file. 
