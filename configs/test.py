import string


def lemmatizer():
    pass


def stemmer():
    pass


config = {
    "base_vectorizer_param": {
        "strip_accents": None,
        "lowercase": True,
        "stop_words": None,
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_df": 1.0,
        "min_df": 1,
        "max_features": None,
        "binary": False,
    },
    "number": None,  # Dictionnaire ou None
    "ponctuation": None,  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    "stemmer": None,
    "lemmatizer": None,
}
