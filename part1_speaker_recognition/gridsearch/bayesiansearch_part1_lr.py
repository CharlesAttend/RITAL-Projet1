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


class CategoricalList(Categorical):
    def __init__(self, categories, **categorical_kwargs):
        super().__init__(self._convert_hashable(categories), **categorical_kwargs)

    def _convert_hashable(self, list_of_lists):
        return [self._HashableListAsDict(list_) for list_ in list_of_lists]

    class _HashableListAsDict(dict):
        def __init__(self, arr):
            self.update({i: val for i, val in enumerate(arr)})

        def __hash__(self):
            return hash(tuple(sorted(self.items())))

        def __repr__(self):
            return str(list(self.values()))

        def __getitem__(self, key):
            return list(self.values())[key]


# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    "vect": Categorical(
        CountVectorizer(ngram_range=(1, 1)),
        CountVectorizer(ngram_range=(1, 2)),
        CountVectorizer(ngram_range=(2, 2)),
        CountVectorizer(ngram_range=(2, 3)),
        CountVectorizer(ngram_range=(3, 3)),
    ),
    "vect__min_df": Real(0.05, 1),
    "vect__max_df": Real(0.3, 1.0),
    "vect__max_features": Categorical([None, 10000, 20000, 50000, 100000]),
    "vect__lowercase": Categorical([False, True]),
    "vect__strip_accents": Categorical([None, "unicode"]),
    "vect__stop_words": CategoricalList([[None], stopwords.words("french")]),
    "vect__binary": Categorical([True, False]),
    "tfidf__use_idf": Categorical([True, False]),
    "tfidf__norm": Categorical([None, "l1", "l2"]),
    "tfidf__smooth_idf": Categorical([True, False]),
    "tfidf__sublinear_tf": Categorical([False, True]),
    "lr__penalty": Categorical(["l2", None]),
    "lr__class_weight": Categorical([None, "balanced"]),
    "lr__C": Real(1e-6, 1e6, prior="log-uniform"),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    searchcv = BayesSearchCV(
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
            print("Interrupting!")
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

    joblib.dump(searchcv, "part1_lr.pkl")
