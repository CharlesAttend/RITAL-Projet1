from __future__ import print_function

import codecs
import re
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

def load_speaker(path="./data/raw/corpus.tache1.learn.utf8"):
    corpus = []
    classes = []
    f = codecs.open(path, "r", "utf-8")  # pour régler le codage
    while True:
        texte = f.readline()
        if (len(texte)) < 5:
            break
        label = re.sub(r"<\d*:\d*:(.)>.*", "\\1", texte)
        texte = re.sub(r"<\d*:\d*:.>(.*)", "\\1", texte)
        if label.count("M") > 0:
            classes.append(-1)
        else:
            classes.append(1)
        corpus.append(texte)
    return corpus, classes


X, y = load_speaker()

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
        ("lr", LogisticRegression(solver="saga")),
    ]
)

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    "vect__min_df": (0.05, 0.1, 0.2, 0.5),
    "vect__max_df": (0.5, 0.75, 1.0),
    "vect__max_features": (None, 5000, 10000, 50000),
    "vect__lowercase": (False, True),
    "vect__strip_accents": (None, "unicode"),
    "vect__stop_words": (None, stopwords.words("french")),
    "vect__ngram_range": ((1, 1), (1, 2), (2, 2)),  # unigrams or bigrams
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": (None, "l1", "l2"),
    "tfidf__smooth_idf": (True, False),
    "tfidf__sublinear_tf": (False, True),
    "lr__penalty": ("l1", "l2", "elasticnet", None),
    "lr__class_weight": (None, "balanced"),
    "lr__max_iter": (100, 1000, 10000)
    
    # "svm__class_weight": (None, "balanced"),
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
        verbose=3,
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

    joblib.dump(grid_search, 'part1_nb.pkl')

