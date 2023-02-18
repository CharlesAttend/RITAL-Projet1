[La template du projet](https://neptune.ai/blog/how-to-structure-and-manage-nlp-projects-templates)

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```


Note : 
- Odd ratios method can improve word cloud
- Rapport :
    - Impact du cleaning
    - Variante de BoW
    - Comment les traitements varie en fonction des deux tache
- PoS facultatif

TODO : 
[] Faire de la cross val 
[] Faire de la viz (mais tu l'as pas mal fait avec les nuages)
[] Faire une fonction pour écrire les résultats des expériences quelque part (classifquement le classif report)
[] Implémenter filtre des nombres dans data.py/Custom_analyzer()
[] Implémenter filtre de la ponctuation dans data.py/Custom_analyzer()
[] Transferer le framework de test pour Chirac/Mitterand