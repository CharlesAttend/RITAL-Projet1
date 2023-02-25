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
        "ngram_range": (1, 1),
        "max_df": 1.0,
        "min_df": 1,
        "max_features": None, 
        "binary": False,
    },
    "number": {
        "regex": "\-?\d{1,10}(?:\.\d{1,10})?",    # plus simple je prends
        "replacement": "NUMBER",
    },  # Dictionnaire ou None
    "ponctuation": string.punctuation, # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    "stemmer": None,
    "lemmatizer": None,
}