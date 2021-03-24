import sys

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads both messages and categories table,
    drop duplicate records from each one and merge them.
    Input: Two dataframes from messages_filepath, categories_filepath
    Output: One Dataframe df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Drop Duplicates records from each table
    messages.drop_duplicates(inplace=True)
    categories.drop_duplicates(inplace=True)
    
    # Merge both tables on id
    df = messages.merge(categories, on='id', how='left', copy=False)
    return df


def clean_data(df):
    '''
    This function cleans df by dropping duplicates and splitting the 
    categories column and further cleaning it.
    Input: One dataframe df
    Output: One dataframe df
    '''
    # Split Categories column on ; into a new dataframe
    categories = df.categories.str.split(';', expand=True)
    # Rename the columns accordingly
    categories.columns = [_[:-2] for _ in df.categories[1].split(';')]
    # Extract 0 or 1 from each entry in the table
    for column in categories:
        categories[column] = categories[column].astype("str").str[-1:].astype("int64")
    
    # Concatenate the original df with new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    # drop the original categories column from `df`
    df.drop(columns="categories", inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace =True)
    df.loc[df.related==2, "related"] = 1
    return df


def save_data(df, database_filename):
    '''
    This function saves df into an splite database
    Input: Dataframe and Database filename
    Output: None
    '''
    engine = create_engine(os.path.join('sqlite:///',database_filename))
    df.to_sql('CleanMessages', engine, index=False)


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