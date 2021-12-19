# RecDiv : analysis of the diversity of musical algorithmic recommendations


## Installing
1. Clone the repo on your machine 
```git clone git@github.com:grodino/recodiv.git```
2. Create the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment
```conda env create -f environment.yml```
3. Check that everything works ```python main.py --help```


## Structure
The entry point of the code is the `main.py` function at the root of the project. Calling this 
function will recreate all the figures shown in the report.


### Files
* The core logic (machine learning models, graph objects and methods) is located in `./recodiv/`. 
* The data folder (not hosted on Github) must be `./data/{data_set_name}/`
* The automation logic (users-songs-tags graphs instanciation, ML model instanciation, figure 
computation... ) is located in `./automation/`. 

This project uses spotify's automation framework [luigi](https://github.com/spotify/luigi/)

> Lauch luigi central scheduler : `luigid --background --pidfile .luigi/pidfile --logdir .luigi`


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
* [Lenskit](https://github.com/lenskit/lkpy): academinc recommender system library
* [RS_c](https://recommender-systems.com): academic recommender system website