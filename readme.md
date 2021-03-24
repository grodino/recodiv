# RecDiv : analysis of the diversity of musical algorithmic recommendations

## Structure
The entry point of the code is the `main.py` function at the root of the project. Calling this 
function will recreate all the figures shown in the report.

* The core logic (machine learning models, graph objects and methods) is located in `./recodiv/`. 
* The data folder (not hosted on Github) must be `./data/{data_set_name}/`
* The automation logic (users-songs-tags graphs instanciation, ML model instanciation, figure 
computation... ) is located in `./automation/`. 

This project uses spotify's automation framework [luigi](https://github.com/spotify/luigi/)

> Lauch luigi central scheduler : `luigid --background --pidfile .luigi/pidfile --logdir .luigi`

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

### A note on diversity metrics

Take a look at [spotify research](https://dl.acm.org/doi/10.1145/3366423.3380281) for diversity calculation with song embedding.
Here, diversity is clearely stated as diversity of audience for tags and diversity of categories listenend for users. 
An index is computed overs an emprirical distribution; this is something biologist do a lot : [Rényi Entropy](https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy), [a bio article](http://www.cambridge.org/core/journals/journal-of-tropical-ecology/article/new-local-estimator-of-regional-species-diversity-in-terms-of-shadow-species-with-a-case-study-from-sumatra/3862C02AFFBD2954004A9BB0A827A7E5)
 

### TODO
- Plot model perfomance indicators along diversity
- Diversity vs model parameters
- Diversity vs model performance
- Diversity vs recommendation depth (# of recommendations)
- In the selection phase, select items that would increase the number of tags reached and items that
  would balance all the categories reached

### List of figures
Here are all the figures that can be generated (listed as luigi tasks)

Dataset analysis :
- `PlotUserVolumeHistogram` : Histogram of the user volume (number of items listened per user). The whole dataset is taken into account
- `PlotUsersDiversitiesHistogram` : Histogram of the user diversity (individual herfindal diversity from the user layer to the tag layer; also called *user attention diversity* or *organic diversity*)
- `PlotTagsDiversitiesHistogram` : Histogram of the tag diversity (individual herfindal diversity from the tag layer to the user layer; also called *tag audience diversity*)

Model training/evaluation :
- `PlotTrainTestUsersDiversitiesHistogram` : Histogram of the *user attention diversity* computed separately on the train and test datasets
- `PlotTrainLoss` : Value of the training loss for an increasing number of epochs
- `PlotModelTuning` : Value of a given evaluation metric with respect to the number of latent factors and the regularization value, displayed as a 2D image.

Recommendation analysis
- `PlotRecommendationsUsersDiversitiesHistogram` : Histogram of the *user recommendation diversity* (the individua herfindal diversity of a user with the user-items edges representing user-recommendation links)
- `PlotDiversitiesIncreaseHistogram` : Histogram of the *user diversity increase* (the individual herfindal diversity of a user after recommendations are added to its user-items-tags graph minus its *organic diversity*)
- `PlotRecommendationDiversityVsUserDiversity` : Each *user recommendation diversity* with respect to its *organic diversity*
- `PlotUserDiversityIncreaseVsUserDiversity` : Each *user diversity increase* with respect to its *organic diversity*

Hyperparameters analysis
- `PlotDiversityVsLatentFactors` : The average *user recommendation diversity* for different number of latent factors in the trained model
- `PlotDiversityIncreaseVsLatentFactors` : The average *user diversity increase*  for different number of latent factors in the trained model
- `PlotDiversityVsRegularization` : The average *user recommendation diversity* for different number of regularization factor in the trained model
- `PlotDiversityIncreaseVsRegularization` : The average *user diversity increase* for different number of regularization factor in the trained model
- `PlotDiversityVsRecommendationVolume` : The average *user recommendation diversity* for different number of recommended items per user.


## Licences
* The code in `recodiv/triversity/` is adapted from [triversity](https://github.com/Nobody35/triversity)
(mainly comments and formating modifications)

## Useful material
* [Tutorial](https://jessesw.com/Rec-System/) on collaborative filtering recsys
* [Implicit](https://implicit.readthedocs.io/en/latest/) collaborative filtering for implicit 
dataset library