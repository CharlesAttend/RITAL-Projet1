import os, re, codecs
import importlib
import numpy as np
from utils import DotDict
from types import FunctionType
from sklearn.feature_extraction.text import CountVectorizer


def load_data_part1(
    path="./part1_speaker_recognition/data/raw/corpus.tache1.learn.utf8",
):
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
    return np.array(corpus), np.array(classes)


def load_data_part2(path="./part2_review/data/raw/"):
    corpus = []
    classes = []
    label = 0
    try:
        for cl in os.listdir(path):  # parcours des fichiers d'un répertoire
            for f in os.listdir(path + cl):
                txt = open(path + cl + "/" + f).read()
                corpus.append(txt)
                classes.append(label)
            label += 1  # changer de répertoire <=> changement de classe
        return np.array(corpus), np.array(classes)
    except FileNotFoundError:
        print(os.listdir("."))
        raise FileNotFoundError


class CustomAnalyzer:
    def __init__(self, config_name) -> None:
        """
        Build an analyser from a config template.

        This analyser will be plug into one of the sklearn.feature_extraction.text
        vectorizer through the parameter "analyzer=MixedAnalyzer"

        Parameters
        ----------
        config_name : str, default=None
            The config file name to load
        """
        self.config = DotDict(importlib.import_module(f"configs.{config_name}").config)
        self.base_analyzer = CountVectorizer(
            **self.config.base_vectorizer_param
        ).build_analyzer()

    def __call__(self, doc):
        """
        Preprocess, tokenize and stemming with nltk.
        Look at stopwords.

        Preprocessing :
            Takes an entire document as input (as a single string), and returns a
            possibly transformed version of the document, still as an entire string.
            This can be used to remove HTML tags, lowercase the entire document, etc.
            preprocess_doc = preprocess(doc)

        Tokenizer :
            a callable that takes the output from the preprocessor
            and splits it into tokens, then returns a list of these.

        N-gram extraction and stop word filtering take place at the analyzer level
        """
        # Ici il preprocess + tokenise à partir du baseVectorizer
        # * strip accent
        # * Stop word
        if self.config.number is not None:
            doc = re.sub(self.config.number.regex, self.config.number.replacement, doc)
        if self.config.ponctuation is not None:
            doc = doc.strip(self.config.ponctuation)
        tokenized_list = self.base_analyzer(doc)

        # Stemming
        if callable(self.config.stemmer):
            return [self.config.stemmer(w) for w in tokenized_list]
        if callable(self.config.lemmatizer):
            return [self.config.lemmatizer(w) for w in tokenized_list]
        return tokenized_list
