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
        "ngram_range": (1, 1),
        "max_df": 1.0,
        "min_df": 1,
        "max_features": None, 
        "binary": False,
    },
    "number": None,  # Pas encore traité 
    "ponctuation": "",  # Pas encore traité 
    "stemmer": None,
    "lemmatizer": None,
}