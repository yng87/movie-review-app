import pandas as pd
import numpy as np
import os
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download("stopwords")
stop = stopwords.words("english")
porter = PorterStemmer()


def preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|-)(?:-)?(?:\)|\(|D|P)", text)
    text = (re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", ""))
    return text

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(w) for w in text.split()]

def train(save_path, movie_data_path, train_size, random_seed, n_jobs, debug):
    df = pd.read_csv(movie_data_path, encoding="utf-8")
    df.review = df.review.map(preprocessor)

    X_train, y_train = df.loc[:train_size, "review"].values, df.loc[:train_size, "sentiment"].values
    X_test, y_test = df.loc[train_size:, "review"].values, df.loc[train_size:, "sentiment"].values

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    param_grid = [
        {
            'vect__ngram_range': [(1,1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'clf__penalty': ["l1", "l2"],
            'clf__C': [1., 10., 100.]
        },
        {
            'vect__ngram_range': [(1,1)],
            'vect__stop_words': [stop, None],
            'vect__tokenizer': [tokenizer, tokenizer_porter],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ["l1", "l2"],
            'clf__C': [1., 10., 100.]
        }
    ]

    lr_tfidf = Pipeline([
        ('vect', tfidf),
        ('clf', LogisticRegression(random_state=random_seed, solver='liblinear'))
    ])

    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=n_jobs)

    if debug:
        X_train = X_train[:50]
        y_train = y_train[:50]
    
    gs_lr_tfidf.fit(X_train, y_train)

    print(f"Best parameter set: {gs_lr_tfidf.best_params_}")
    
    cv_best_score = gs_lr_tfidf.best_score_
    print(f"CV Accuracy: {cv_best_score}")

    test_score = gs_lr_tfidf.best_estimator_.score(X_test, y_test)
    print(f"Test Accuracy: {test_score}")

    if debug:
        save_path = save_path + ".dbg"
    print(f"Save model to {save_path}")
    joblib.dump(gs_lr_tfidf.best_estimator_, save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--movie_data_path", type=str, default="./csv/movie_data.csv")
    parser.add_argument("--train_size", type=int, default=25000)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args_dict = vars(args)

    train(**args_dict)