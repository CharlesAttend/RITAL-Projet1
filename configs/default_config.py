import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer(object):
    def init(self):
        self.wnl = WordNetLemmatizer()
    def call(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
        
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
        "regex": "-?\d+((,|\.)\d+)?", 
        "replacement": "NUMBER",
    },  # Dictionnaire ou None
    "ponctuation": string.punctuation,  # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    "stemmer": None,
    "lemmatizer": None,
}