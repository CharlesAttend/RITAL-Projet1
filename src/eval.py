from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from src.data import Custom_analyzer
import pandas as pd
import json, os

def print_score(y_test, pred, name):
    macro_f1 = f1_score(y_test, pred, average='macro')
    micro_f1 = f1_score(y_test, pred, average='micro')
    macro_auc = roc_auc_score(y_test, pred)
    micro_auc = roc_auc_score(y_test, pred, average="weighted")
    classif_report = classification_report(y_test, pred, output_dict=True)

    reports = f"""
    {name} :
    =====
    Macro F1-score : {macro_f1}
    Micro F1-score : {micro_f1}
    Macro ROC-AUC: {macro_auc}
    Weighted ROC-AUC: {micro_auc}
    Classification report :
    {classification_report(y_test, pred)}
    =====
    """
    print(reports)

    classif_report['macro_auc'] = macro_auc
    classif_report['micro_auc'] = micro_auc
    return classif_report

def fit_eval(X_train, y_train, X_test, y_test, balanced=None):
    """
    Fit et évalue les algo classique de classification sur les jeux de données envoyé en paramètre

    Parameters
    ----------
    X_train: Sparce Matrix
        Important : Une matrice BoW est attendu

    X_test: Sparce Matrix
        Important : Une matrice BoW est attendu

    y_train: list
        Label du train

    y_test: list 
        Label du test
    
    balanced: bool

    """
    #Naïve Bayes
    if balanced==None:
        nb_clf = MultinomialNB()
    else:
        balanced = "balanced"
        nb_clf = MultinomialNB(fit_prior=True)
    nb_clf.fit(X_train, y_train)

    #Logistic Regression
    lr_clf = LogisticRegression(random_state=0, solver='lbfgs',n_jobs=-1, max_iter=10000, class_weight=balanced)
    lr_clf.fit(X_train, y_train)

    #Linear SVM
    svm_clf = LinearSVC(random_state=0, tol=1e-5, max_iter=20000, class_weight=balanced)
    svm_clf.fit(X_train, y_train)

    pred_nb = nb_clf.predict(X_test)
    pred_lr = lr_clf.predict(X_test)
    pred_svm = svm_clf.predict(X_test)

    # Ridge Classifier ?
    results = (print_score(y_test, pred_nb, 'Naïve Bayes'),\
        print_score(y_test, pred_lr, 'Logistic Regression'),\
        print_score(y_test, pred_svm, 'SVM'))
    algo_names = ['Naïve Bayes', 'Logistic Regression', 'SVM']

    return results, algo_names

def eval_from_config(X_train, X_test, y_train, y_test, config_name, vectorizer_class=CountVectorizer, saveName=None, path=None):
    """
    Evalue avec les prétraitement de la config sur les sets de train et test fournis en paramètre

    Parameters
    ----------
    X_train: list of string
        Important : Fournir les données bruts pour pouvoir prétraiter (pas de BoW)

    X_test: list of string
        Important : Fournir les données bruts pour pouvoir prétraiter (pas de BoW)

    y_train: list
        Label du train

    y_test: list 
        Label du test

    config_name : str, default=None
        The config file name to load 
    
    vectorizer_class: Submodule of sklearn.feature_extraction.text => {CountVectorizer, TfidfVectorizer, HashingVectorizer}
        This analyser will be plug into one of the sklearn.feature_extraction.text vectorizer with the param "analyzer = Mixed_anayzer"
    
    saveName: str
        If not None, save result and config under this name
    """
    custom_ana = Custom_analyzer(config_name)
    vectorizer = vectorizer_class(analyzer = custom_ana)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    results, algo_names = fit_eval(X_train, y_train, X_test, y_test)
    if saveName != None:
        if path == None:
            raise ValueError('Path needed to save results')
        config = custom_ana.config
        save_eval_and_config(config, results, algo_names, saveName, path)

def save_eval_and_config(config, results, algo_names, saveName, path):
    """
    Ajoute les résultat à la dataframe au format long situé dans path+"stats.csv"
    Save la config sous le saveName fournis en param

    Parameters
    ----------
    config: dict
        La config du CustomAnalyzer 
    
    results: tuple of dict
        Résultat renvoyé par fit_eval() avec toutes les metrics, il est possible d'en ajouter comme cela fut fait avec le AUC score
    
    algo_names: list of strings
        Comme results est un tuple de résultat pour chaque algo, ce paramètre permet de nommer les algos dans l'ordre

    saveName:
        Le nom court pour la config associé, il doit être unique par rapport à un précédent, sous peine de remplacer la config de l'ancien
    
    path: Optional string
        Optional path
    """
    # Sanity check
    assert len(results) == len(algo_names)
    assert os.path.isdir(path), "Wrong path to save configs"
    
    # Saving configs
    with open(path+'config_name_map.json', 'r') as f:
        # Reading
        content = json.load(f)
        assert isinstance(content, dict), "Problème avec le format du json, il faut un dictionnaire à la racine"
    
    # Checking if this saveName is not already used
    if content.get(saveName, None) != None:
        raise ValueError("Save Name already used")

    # Rewrite
    with open(path+'config_name_map.json', 'w') as f:
        # Append
        content[saveName] = config
        # Rewrite
        json.dump(content, f, indent=4)

    # Saving Result
    try:
        df = pd.read_csv(path+"stats.csv")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    for result, name in zip(results, algo_names):
        tmp = pd.DataFrame(result).melt()
        tmp['Algo'] = [name for _ in range(len(tmp))]
        tmp['configName'] = saveName
        df = pd.concat([df,tmp])
    df.to_csv(path+"stats.csv")