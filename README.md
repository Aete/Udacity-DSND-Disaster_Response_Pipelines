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


## How to Run

### Requirement

1. python: 3.5+
2. Data Science Library : pandas, sqlalchemy, numpy, scikit-learn, lightgbm (gpu), NLTK, plotly
3. Web App Library : flask

### 1. Install
Clone this repository

`git clone https://github.com/Aete/Udacity-DSND-Disaster_Response_Pipelines`

### 2. Load and Clean Database

Set the workspace as 

`Udacity-DSND-Disaster_Response_Pipelines/data`

and run code below

`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

### 3. Build and Train a Pipeline

Set the workspace as 

`Udacity-DSND-Disaster_Response_Pipelines/models`

and run code below

`python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

### 4. Run a Web App

Set the workspace as 

`Udacity-DSND-Disaster_Response_Pipelines/app`

run code below and open localhost:9000

`python run.py`


