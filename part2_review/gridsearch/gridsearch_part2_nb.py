from __future__ import print_function

import os
import joblib

from pprint import pprint
from time import time
import logging

import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

nltk.download("stopwords")

def load_movies(path="./data/raw/movies/"):
    corpus = []
    classes = []
    label = 0
    for cl in os.listdir(path):  # parcours des fichiers d'un répertoire
        for f in os.listdir(path + cl):
            txt = open(path + cl + "/" + f).read()
            corpus.append(txt)
            classes.append(label)
        label += 1  # changer de répertoire <=> changement de classe
    return corpus, classes

X, y = load_movies()

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("nb", MultinomialNB()),
    ]
)

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    "vect__min_df": (0.05, 0.15, 1),
    "vect__max_df": (0.5, 0.75, 1.0),
    "vect__max_features": (None, 20000, 200000),
    "vect__lowercase": (False, True),
    "vect__strip_accents": (None, "unicode"),
    "vect__stop_words": (None, stopwords.words("english")),
    "vect__ngram_range": ((1, 1), (1, 2), (2, 2)),  # unigrams or bigrams
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": (None, "l1", "l2"),
    "tfidf__smooth_idf": (True, False),
    "tfidf__sublinear_tf": (False, True),
    "nb__alpha": (0.5, 0.6, 0.7),
    "nb__fit_prior": (True, False),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        n_jobs=-1,
        verbose=10,
        scoring="roc_auc",
    )

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    joblib.dump(grid_search, 'part2_nb.pkl')

