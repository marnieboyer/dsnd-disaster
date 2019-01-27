import sys
# import libraries
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    ''' 
    Function loads data from sql lite db
    INPUT: filepath
    OUTPUT: X (independent variables), y (dependent variable to be predicted)  and labels
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_final',engine)
    labels = df.columns[4:].tolist()
    X = df['message']
    y = df[labels].values
    return X,y,labels

def tokenize(text):
    '''
    Function tokenizes the message text
    INPUT: text, string
    OUTPUT: clean, lemmatized tokens
    '''
    words =  word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for word in words:
        clean = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean)
        
    return clean_tokens


def build_model():
    '''
    Create pipeline for model
    '''
    scorer=make_scorer(f1_score, average = 'micro')
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, stop_words = 'english')),
    ('tfidf', TfidfTransformer()),
        ## Tried several models:
    ###('clf', MultiOutputClassifier(LinearSVC()))  
    ('clf',MultiOutputClassifier(LogisticRegression()))
    ])
 
    parameters = {
        'clf__estimator__C': [0.1,0.5,1],
     #   'clf__estimator__multi_class': ['ovr']
        'clf__estimator__random_state': [25],
        'clf__estimator__solver':['lbfgs','liblinear']
    }

    model = GridSearchCV(pipeline, param_grid = parameters, scoring = scorer)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Run model with report 
    INPUT: ML model, X and Y data, list of cats
    OUTPUT: classification report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names = category_names))
 


def save_model(model, model_filepath):
    '''
    Function used to save the model to disk
    INPUT: model name, model location
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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