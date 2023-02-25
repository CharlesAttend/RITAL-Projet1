import os, re, codecs
import importlib
from src.utils import DotDict
from types import FunctionType
from sklearn.feature_extraction.text import CountVectorizer

def load_data_part2(path2data="./part2_review/data/raw/"): # 1 classe par répertoire
    alltxts = [] # init vide
    labs = []
    cpt = 0
    try:
        for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
            for f in os.listdir(path2data+cl):
                txt = open(path2data+cl+'/'+f).read()
                alltxts.append(txt)
                labs.append(cpt)
            cpt+=1 # chg répertoire = cht classe
        return alltxts,labs
    except FileNotFoundError:
        print(os.listdir('.'))
        raise FileNotFoundError

def load_data_part1(path="./part1_speaker_recognition/data/raw/corpus.tache1.learn.utf8"):
    alltxts = []
    alllabs = []
    s = codecs.open(path, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts, alllabs

class Custom_analyzer():
    def __init__(self, config_name) -> None:
        """
        Build an analyser from a config template. 

        This analyser will be plug into one of the sklearn.feature_extraction.text vectorizer with the param "analyzer = Mixed_anayzer"

        Parameters
        ----------
        config_name : str, default=None
            The config file name to load 
        """
        self.config = DotDict(importlib.import_module(f'configs.{config_name}').config)
        self.base_analyzer = CountVectorizer(**self.config.base_vectorizer_param).build_analyzer()

    def __call__(self, doc):
        """
        Stem avec nltk + regarde les stopwords
        """
        # Preprocessing : 
        #   Takes an entire document as input (as a single string), and returns a possibly transformed version of the document, still as an entire string. 
        #   This can be used to remove HTML tags, lowercase the entire document, etc.
        # preprocess_doc = preprocess(doc)
    
        # Tokenizer :
        #   a callable that takes the output from the preprocessor 
        #   and splits it into tokens, then returns a list of these.
        
        # N-gram extraction and stop word filtering take place at the analyzer level
        
        # Ici il preprocess + tokenise à partir du baseVectorizer
        # * strip accent
        # * Stop word
        if self.config.number != None:
            doc = re.sub(self.config.number.regex, self.config.number.replacement, doc)
        if self.config.ponctuation != None:
            doc = doc.strip(self.config.ponctuation)
        tokenized_list = self.base_analyzer(doc)

        # Stemming
        if callable(self.config.stemmer):
            return [self.config.stemmer(w) for w in tokenized_list]
        elif callable(self.config.lemmatizer):
            return [self.config.lemmatizer(w) for w in tokenized_list]
        else:
            return tokenized_list