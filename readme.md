# RecDiv : analysis of the diversity of musical algorithmic recommendations

## Structure
The entry point of the code is the `main.py` function at the root of the project. Calling this 
function will recreate all the figures shown in the report.

* The core logic (machine learning models, graph objects and methods) is located in `./recodiv/`. 
* The data folder (not hosted on Github) must be `./recodiv/data/{data_set_name}/`
* The automation logic (users-songs-tags graphs instanciation, ML model instanciation, figure 
computation... ) is located in `./automation/`. 

This project uses spotify's automation framework [luigi](https://github.com/spotify/luigi/)

### Choosing a recommenders systems library
Even if choosing a library is not usually of particular interest in research article it is important for reproductibility purposes (see [Are We Really Making Much Progress?](http://arxiv.org/abs/1907.06902)). We explore in this section the existing libraries related to recommender systems.

Many applied mathematics domains have their own established libraries/stacks:
- [numpy](https://numpy.org/)/[scipy](https://scipy.org/) for general optimization, linear algebra ...
- [scikit-learn](https://scikit-learn.org) for general machine learning tasks (regression, classification)
- [pyTorch](https://pytorch.org/), [Tensorflow](https://www.tensorflow.org/) for deep neural newtworks 

This allows for a complete ecosystem of models, training techniques and evaluation.

Recommender systems deal with very large amount of data of different nature (explicit: user ratings, comments .., implicit: clicks, listenings, reads ...), therefore they require sometinmes very different algorithms. Moreover, basic metrics (such as Root Mean Squared Error) fail to capture models performance (see [How good your recommender system is?](https://doi.org/10.1007/s13042-017-0762-9)). Unfortunately,  there is no such thing as a library implementing major model and evaluations metrics. Here as some libraries exploring different aspects of recommender systems:

| Name                                                                              | Models                                                                                                      | Metrics                   | Comments                                                           |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| [Tensorflow recommenders](https://www.tensorflow.org/recommenders)                | NeuMF                                                                                                       | FactorizedTopK            | Benefits from the Tensorflow ecosystem, provides tools for scaling |
| [Apache Spark](https://spark.apache.org/docs/1.2.2/api/python/pyspark.mllib.html) | Collaborative filtering for implicit datasets                                                               | None                      | Created for algorithms deployment on clusters                      |
| [Implicit](https://github.com/benfred/implicit)                                   | Collaborative filtering for implicit datasets, Bayesian Personalised Ranking, Logistic Matrix Factorization | Precision, MAP, NDCG, AUC | Buggy metrics implementation from my experience                    |
| [Surprise](https://github.com/NicolasHug/Surprise)                                | Matrix Factorization (SVD, SVD++), NeuMF, random ...                                                        | RMSE, MSE, MAE, FCP       | Do not provide implicit dataset support out of the box             |

> In an effort to list libraries and the news associated to recommender systems, Prof. [Joeran Beel](https://isg.beel.org/people/joeran-beel/) created the website [RS_c](https://recommender-systems.com). 

The choosen library should have the following properties:
- Common models (Collaborative filtering, implicit or not ...) are implemented or are easy to implement
- Common metrics (Recall, AUC, )


## Licences
* The code in `recodiv/triversity/` is adapted from [triversity](https://github.com/Nobody35/triversity)
(mainly comments and formating modifications)

## Useful material
* [Tutorial](https://jessesw.com/Rec-System/) on collaborative filtering recsys
* [Implicit](https://implicit.readthedocs.io/en/latest/) collaborative filtering for implicit 
dataset library