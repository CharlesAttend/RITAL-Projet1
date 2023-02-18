import json 
from sklearn.metrics import classification_report

def write_classif_report(y_true, y_pred, name, **kwds):
    """
    NE FONCTIONNE PAS YET
    * Eval it with basic metrics
    * Write the metrics in json to keep a trace
    """
    classif_report_dict = classification_report(y_true, y_pred, output_dict=True, **kwds)
    with open("../reports/model_eval/dict.json", "r+") as f:
        eval_dict = json.load(f)
        eval_dict[name] = classif_report_dict
        json.dump(classif_report_dict, f, indent=2)

    return classif_report_dict