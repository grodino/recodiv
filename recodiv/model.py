import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as pl

from lenskit import topn
from lenskit import util
from lenskit import batch
from lenskit import crossfold as xf
from lenskit.algorithms import als
from lenskit.metrics.predict import rmse
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import TopN, Memorized


from recodiv.utils import print_song_info
from recodiv.utils import get_msd_song_info


# util.log_to_stderr()
METRICS = {
    'ndcg': topn.ndcg,
    'recall': topn.recall,
    'precision': topn.precision,
    'recip_rank': topn.recip_rank,
}


def split_dataset(ratings, user_fraction=.1):
    """Split a dataset in train/test data"""

    n_users = len(ratings['user'].unique())

    # There are many ways to separate a dataset in (train, test) data, here are two:
    #   - Row separation: the test set will contain users that the model knows.
    #     The performance of the model will be its ability to predict "new" 
    #     tastes for a known user
    #   - User separation: the test set will contain users that the model has
    #     never encountered. The performance of the model will be its abiliy to
    #     predict new users behaviours considering the behaviour of other
    #     known users.
    # see [lkpy documentation](https://lkpy.readthedocs.io/en/stable/crossfold.html)
    # Here the sampling is as follow:
    #   - Sample test_fraction * n_total users
    #   - Randomly select half of their listenings for the test set
    result = list(xf.sample_users(
        ratings[['user', 'item', 'rating']], 
        partitions=1,
        size=int(n_users * user_fraction),
        method=xf.SampleFrac(.5)
    ))[0]

    print(f'n test users: {len(result.test["user"].unique())}')

    return result.train, result.test


def train_model(
        train, 
        test, 
        n_factors=30, 
        n_iterations=20, 
        regularization=.1, 
        evaluate_iterations=False,
        iteration_metrics=None,
        n_recommendations=50,
        confidence_factor=40):
    """Train (and evaluate iterations if requested) model
    
    :returns: (model, iterations_metrics). If evaluate_iterations == False,
        metrics is an empty pd.DataFrame
    """

    # Encapsulate the model into a TopN recommender
    model = Recommender.adapt(als.ImplicitMF(
        n_factors, 
        iterations=n_iterations, 
        weight=confidence_factor, 
        progress=tqdm, 
        method='cg'
    ))

    # Compute the confidence values for user-item pairs
    train['rating'] = 1 + confidence_factor * train['rating']
    
    metrics = pd.DataFrame()

    if evaluate_iterations:
        # Prepare metrics calculation
        analysis = topn.RecListAnalysis()
        users = test.user.unique()

        for metric_name in iteration_metrics:
            analysis.add_metric(METRICS[metric_name])

        for iteration, intermediate_model in enumerate(model.fit_iters(train)):
            # Create recommendations
            recommendations = batch.recommend(intermediate_model, users, n_recommendations)

            # Compute and save metrics
            results = analysis.compute(recommendations, test)
            results['iteration'] = iteration
            metrics = pd.concat([metrics, results], ignore_index=True)
        
        metrics = metrics.groupby('iteration')[list(iteration_metrics)].mean()
            
        # metrics.plot(
        #     logy=True, title='metrics with respect to iteration count'
        # )
        # pl.show()

    else:
        model.fit(train)
    
    return model, metrics
    

def generate_predictions(model, user_item):
    """Generate the rating predictions for each user->item pair
    
    :returns: pd.DataFrame. A dataframe with at least the columns 'user', 
        'item', 'prediction' (the predicted scores)
    """

    return batch.predict(model, user_item)
     

def generate_recommendations(model, test_ratings, n_recommendations=50):
    """Generate recommendations for a given model
    """

    users = test_ratings.user.unique()

    return batch.recommend(model, users, n_recommendations)


def evaluate_model_recommendations(recommendations, test, metrics):
    """Evaluates a model via its recommendations
    
    :param recommendations: pd.DataFrame with at least the following columns :  
        'user', 'item', 'score', 'rank'
    :param test: pd.DataFrame. The testing data
    :param metrics: list. A list of metrics' names (see recodiv.model.METRICS)
    """

    analysis = topn.RecListAnalysis()
    users = test.user.unique()
    rec_users = recommendations['user'].unique()

    for metric_name in metrics:
        analysis.add_metric(METRICS[metric_name])

    return analysis.compute(recommendations, test)


def evaluate_model_loss(model, predictions):

    # do not consider the user-item pairs where no prediction could be generated
    # (ie the items not in train set)
    predictions = predictions[predictions['prediction'].notna()]

    confidence = 1 + model.predictor.weight * predictions['rating'].to_numpy()
    prediction = predictions['prediction'].to_numpy()

    reg = model.predictor.reg * (
        np.linalg.norm(model.predictor.user_features_, 'fro') \
        + np.linalg.norm(model.predictor.item_features_, 'fro')
    )

    return confidence @ (1 - prediction)**2 + reg
