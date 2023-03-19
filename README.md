# RITAL - Projet


```
.
├── README.md                                               <- The top-level README for developers using this project.
├── configs                                                 <- Configs used for a deprecated "training framework"
├── final_report                                            <- Final project report
├── n-sentences_bloc.ipynb                                  <- Notebook for testing n-sentence training blocs
├── param_analysis.ipynb                                    <- Visualization for the most importante features (words)
├── part1_speaker_recognition                               <- Main directory for the first part of the project
│   ├── data                                                <- raw datasets and prediction for validation dataset
│   │   ├── raw                                             
│   │   └── validation                                      
│   ├── gridsearch                                          <- Python Pickles for GridSearch results
│   │   └── results                                         
│   ├── notebooks                                           
│   │   ├── 1.0-Aymeric-wordclouds.ipynb                    <- Wordcloud viz
│   │   ├── 1.0-Charles-initial-data-exploration.ipynb      <- First notebook with Oversampler, Undersampler, RUS, ROS
│   │   ├── 2.0-Charles-spacy.ipynb                         <- A try for spacy lematizer, too slow xd
│   │   └── Estimations-et-Analyses.ipynb                   <- ROC Curve, Learning Curve, Complexity analysis
│   ├── reports                                             <- Saving figs for report
│   │   ├── figures                                         
│   │   └── model_eval                                      
│   │       └── dict.json                                   <- The deprecated framework saving his results
│   └── stats                                               <- Same 
│       ├── config_name_map.json                            
│       └── stats.csv                                       
├── part2_review                                            <- Main directory for the second part of the project
│   ├── data                                                <- Same structure for everything
│   ├── gridsearch                                          
│   ├── notebooks                                           
│   ├── reports                                             
│   └── stats                                               
├── postprocessing.ipynb                                    <- Postprocessing notebook around removing noise
├── project_ressources                                      <- Project instructions
│   ├── 1_0-Tutorial-Document-Classification.ipynb          
│   └── 1_1-BoW-Project.ipynb                               
├── src                                                     <- Utility functions
│   ├── data.py                                             
│   ├── eval.py                                             
│   ├── gridsearch.py                                       
│   ├── postprocessing.py                                   
│   └── utils.py                                            
├── test_embeding_vect.ipynb                                <- Test for word embedding as input for models
└── validation.ipynb                                        <- Notebook used to export prediction for validation dataset

```

