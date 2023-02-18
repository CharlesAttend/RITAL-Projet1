from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer


def print_score(y_test, pred, name):
    reports = f"""
    Macro F1-score : {f1_score(y_test, pred, average='macro')}
    Micro F1-score : {f1_score(y_test, pred, average='micro')}
    Macro ROC-AUC: {roc_auc_score(y_test, pred)}
    Weighted ROC-AUC: {roc_auc_score(y_test, pred, average="weighted")}
    Classification report :
    {classification_report(y_test, pred)}
    """
    print(name, ':')
    print(reports)

def fit_eval(X_train, y_train, X_test, y_test, balanced=None):
    #Naïve Bayes
    if balanced==None:
        nb_clf = MultinomialNB()
    else:
        balanced = "balanced"
        nb_clf = MultinomialNB(fit_prior=True)
    nb_clf.fit(X_train, y_train)

    #Logistic Regression
    lr_clf = LogisticRegression(random_state=0, solver='lbfgs',n_jobs=-1, max_iter=300, class_weight=balanced)
    lr_clf.fit(X_train, y_train)

    #Linear SVM
    svm_clf = LinearSVC(random_state=0, tol=1e-5, max_iter=2000, class_weight=balanced)
    svm_clf.fit(X_train, y_train)

    pred_nb = nb_clf.predict(X_test)
    pred_lr = lr_clf.predict(X_test)
    pred_svm = svm_clf.predict(X_test)

    # Ridge Classifier ?

    print_score(y_test, pred_nb, 'Naïve Bayes')
    print_score(y_test, pred_lr, 'Logistic Regression')
    print_score(y_test, pred_svm, 'SVM')

def eval_from_config(config_name, vectorizer=CountVectorizer):
    vectorizer()
