import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy as db
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    engine = create_engine(os.path.join('sqlite:///',database_filepath))
    connection = engine.connect()
    df = pd.read_sql('CleanMessages', connection)
    X = df.message
    category_names = df.columns[4:].tolist()
    Y = df.drop(columns= ['id','original','message','genre']).values
    return X, Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    return tokens


def build_model():
    parameters = {'vect__ngram_range': ((1, 1), (1, 2))
#               'vect__max_df': (0.5, 0.75, 1.0)
#               'vect__max_features': (None, 5000, 10000)
#               'tfidf__use_idf': (True, False),
#               'tfidf__smooth_idf': (True, False),
#               'clf__estimator__n_estimators': [50, 100, 200],
#               'clf__estimator__min_samples_split': [2, 3, 4]
             }
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',MultiOutputClassifier(RandomForestClassifier()))
        ])

    model = GridSearchCV(pipeline, param_grid = parameters)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
#     labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
#     accuracy = (y_pred == y_test).mean()
    
#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
#     print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    for i in range(y_test.shape[1]):
        print(category_names[i])
        print(classification_report(y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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