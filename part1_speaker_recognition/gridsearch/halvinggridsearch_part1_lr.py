from __future__ import print_function

from pprint import pprint
from time import time
import logging
import codecs
import re
import joblib

import nltk
from nltk.corpus import stopwords

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

nltk.download("stopwords")

def load_speaker(path="../data/raw/corpus.tache1.learn.utf8"):
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
        ("lr", LogisticRegression(max_iter=50000)),
    ]
)

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    "vect__min_df": (0.05, 0.15, 1),
    "vect__max_df": (0.5, 0.75, 1.0),
    "vect__max_features": (None, 10000, 20000, 50000, 100000),
    "vect__lowercase": (False, True),
    "vect__strip_accents": (None, "unicode"),
    "vect__stop_words": (None, stopwords.words("french")),
    "vect__ngram_range": ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3)),
    "vect__binary": (True, False),
    "tfidf__use_idf": (True, False),
    "tfidf__norm": (None, "l1", "l2"),
    "tfidf__smooth_idf": (True, False),
    "tfidf__sublinear_tf": (False, True),
    "lr__penalty": ("l2", None),
    "lr__class_weight": (None, "balanced"),
    "lr__C": (.001, .01, .1, 1, 10, 100, 1000),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block



    # find the best parameters for both the feature extraction and the
    # classifier
    searchcv = HalvingGridSearchCV(
        pipeline,
        parameters,
        n_jobs=-1,
        verbose=10,
        scoring="roc_auc",
    )

    # callback handler
    def on_step(optim_result):
        score = searchcv.best_score_
        print("best score: %s" % score)
        if score >= 0.98:
            print('Interrupting!')
            return True

    t0 = time()
    searchcv.fit(X, y, callback=on_step)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % searchcv.best_score_)
    print("Best parameters set:")
    best_parameters = searchcv.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    joblib.dump(searchcv, 'part1_lr.pkl')

