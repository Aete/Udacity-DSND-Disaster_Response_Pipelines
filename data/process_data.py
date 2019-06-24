import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
        
    '''
    Function for load database from sql file. It divides database to two different sets (X and Y).
    
    Args:
        messages_filepath (string): filepath of message data
        categories_filepath (string): filepath of disaster categories data
    
    Returns:
        df (pandas.DataFrame): merged dataframe from massages and categories data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath,sep=',')
    df = messages.merge(categories,on='id')
    return df

def clean_data(df):
            
    '''
    Function for cleaning database. 
    
    Args:
        df (pandas.DataFrame): target dataframe
    
    Returns:
        df (pandas.DataFrame): cleaned dataframe
    '''
    
    categories=df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    categories['related'] = categories['related'].replace(2,0)
    categories=categories.drop('child_alone',axis=1)
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    dupulicated_index_list = df[df.duplicated()].index.tolist()
    df = df.drop(dupulicated_index_list,axis=0)
    return df
    
def save_data(df, database_filename):
    '''
    Function that saves a cleaned database.
    
    Args:
        df (pandas.DataFrame): target dataframe to save
        database_filename (string): filename of database
        
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()