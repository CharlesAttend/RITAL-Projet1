from data import *

import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords

# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from skopt.plots import plot_objective, plot_histogram

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.fixes import loguniform
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV


# nltk.download("stopwords")

X, y = load_data_part1("../part1_speaker_recognition/data/raw/corpus.tache1.learn.utf8")
# X, y = load_data_part2("../part2_review/data/raw/")

classes = np.unique(y)
weights = {
    classes[0]: len(y[y == classes[0]]) / len(X),
    classes[1]: len(y[y == classes[1]]) / len(X),
}

## Define pipelines, default one combines text features extractors
def make_default_pipeline():
    return Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer())])


## We add the classifiers we want to tryout on our pipeline
def make_pipeline(pipeline, classifier):
    if classifier.__name__ == "LogisticRegression":
        pipeline.steps.append(["model", classifier(solver="liblinear")])
    else:
        pipeline.steps.append(["model", classifier()])


## parameters used to explore and optimize our models
## default ones for the bag-of-words model
def make_default_parameters():
    return {
        # "vect__min_df": (0.05, 0.15, 1),
        "vect__max_df": (0.5, 0.75, 1.0),
        "vect__max_features": (
            None,
            5000,
            10000,
            20000,
            30000,
            40000,
            50000,
            75000,
            100000,
            150000,
            200000,
            250000,
            500000,
        ),
        "vect__lowercase": (False, True),
        "vect__strip_accents": (None, "unicode"),
        "vect__stop_words": (None, stopwords.words("french")),
        "vect__ngram_range": ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3)),
        "vect__binary": (True, False),
        "tfidf__use_idf": (True, False),
        "tfidf__norm": (None, "l1", "l2"),
        "tfidf__smooth_idf": (True, False),
        "tfidf__sublinear_tf": (False, True),
    }


def make_parameters(parameters, classifier):
    param_nb = {
        "model__alpha": loguniform(1e-10, 1000),
        "model__fit_prior": (True, False),
    }

    param_lr = {
        "model__penalty": ("l2", "l1"),
        "model__class_weight": (None, "balanced", weights),
        "model__C": loguniform(1e-6, 1e6),
    }

    param_svc = {
        "model__class_weight": (None, "balanced", weights),
        "model__loss": ("hinge", "squared_hinge"),
        "model__C": loguniform(1e-6, 1e6),
    }

    param_ridge = {
        "model__class_weight": (None, "balanced", weights),
        "model__alpha": loguniform(1e-6, 1e6),
    }

    param_sgd = {
        "model__class_weight": (None, "balanced", weights),
        "model__loss": (
            "hinge",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ),
        "model__alpha": loguniform(1e-6, 1e6),
    }
    if classifier.__name__ == "MultinomialNB":
        parameters.update(param_nb)
    elif classifier.__name__ == "LogisticRegression":
        parameters.update(param_lr)
    elif classifier.__name__ == "LinearSVC":
        parameters.update(param_svc)
    elif classifier.__name__ == "RidgeClassifier":
        parameters.update(param_ridge)
    elif classifier.__name__ == "SGDClassifier":
        parameters.update(param_sgd)


classifiers = [
    MultinomialNB,
    LogisticRegression,
    LinearSVC,
    RidgeClassifier,
    SGDClassifier,
]

from sklearn.metrics import f1_score, make_scorer

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block
    scores = {}

    f1_scorer = make_scorer(f1_score, average="binary", pos_label=-1)

    for model in classifiers:
        pipeline = make_default_pipeline()
        make_pipeline(pipeline, model)
        parameters = make_default_parameters()
        make_parameters(parameters, model)
        searchcv = HalvingRandomSearchCV(
            pipeline,
            parameters,
            n_jobs=-1,
            verbose=10,
            scoring=f1_scorer,
        )
        searchcv.fit(X, y)
        scores[model.__name__] = {
            "score": searchcv.best_score_,
            "params": searchcv.best_params_,
        }
        ## On save
        path = "../part1_speaker_recognition/gridsearch/results/part1_hrscv_"
        # path = "../part2_review/gridsearch/results/part2_hrscv_"
        filename = path + model.__name__ + ".pkl"
        # joblib.dump(searchcv, filename)

    print()
    print(scores)
