# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    '''
    function loads data for project
    INPUT:  location for messages file, location for categories file
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df =  pd.merge(messages, categories, on=['id'])
    return df
    
def clean_data(df):
    '''
    Function cleans up data for processing and modeling
    INPUT dataframe
    OUTPUT clean dataframe
    '''
   # split categories into separate columns
    categories = df.categories.str.split(";", expand=True)
    # use the first row to define the categories
    row = categories.iloc[0].tolist()
    # remove the last 2 digits
    z = slice(0,-2)
    category_colnames = []
    for n in row:
        category_colnames.append(n[z])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('float64') 
    # drop the other category
    df.drop(['categories'], axis=1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #clean up the df to remove duplicates
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    # clean up to remove any data that is not a 0 or 1, in order to model
    df = df[df.related.isin([0,1])]
    # remove any columns that have more than 2 different values for modeling
    for col in categories.columns:
        if len(df[col].unique()) < 2:
            df.drop([col], axis = 1, inplace = True)
        else:
            pass
        
    return df 

def save_data(df, database_filename):
    ''''
    Function saves data to databsae
    INPUT: dataframe, database location to save it
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages_final', engine, if_exists='replace', index=False)  

  
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