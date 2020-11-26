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
from lenskit.algorithms import Recommender


from recodiv.utils import print_song_info
from recodiv.utils import get_msd_song_info


# util.log_to_stderr()
METRICS = {
    'ndcg': topn.ndcg,
    'recall': topn.recall,
    'precision': topn.precision,
    'recip_rank': topn.recip_rank
}


def import_and_split(folder):
    """Import a dataset and split it in train/test data"""
    dataset_folder = Path(folder)

    ratings = pd.read_csv(
        dataset_folder.joinpath('msd_users.txt'),
        sep=' ',
        names=['node1_level', 'user', 'node2_level', 'item', 'rating'],
        dtype={
            'node1_level': np.int8,
            'node2_level': np.int8,
            'user': np.str,
            'item': np.str,
            'rating': np.int32
        },
        nrows=1_000_000,
        engine='c'
    )[['user', 'item', 'rating']]

    tags = pd.read_csv(
        dataset_folder.joinpath('msd_tags.txt'),
        sep=' ',
        names=['node1_level', 'item', 'node2_level', 'tag', 'tag_weight'],
        dtype={
            'node1_level': np.int8,
            'node2_level': np.int8,
            'item': np.str,
            'tag': np.str,
            'weight': np.int32
        },
        engine='c'
    )[['item', 'tag', 'tag_weight']]

    # There are many ways to separate a dataset in (train, test) data, here are two:
    #   - Row separation: the test set will contain users that the model knows.
    #     The performance of the model will be its ability to predict "new" 
    #     tastes for a known user
    #   - User separation: the test set will contain users that the model has
    #     never encountered. The performance of the model will be its abiliy to
    #     predict new users behaviours considering the behaviour of other
    #     known users.
    # see [lkpy documentation](https://lkpy.readthedocs.io/en/stable/crossfold.html)
    train, test = xf.sample_rows(
        ratings[['user', 'item', 'rating']], 
        None,
        1_000
    )

    return train, test


def train_model(
        train, 
        test, 
        n_factors=30, 
        n_iterations=20, 
        regularization=.1, 
        evaluate_iterations=False,
        iteration_metrics=None,
        n_recommendations=50):
    """Train (and evaluate iterations if requested) model
    
    :returns: (model, iterations_metrics). If evaluate_iterations == False,
        metrics is an empty pd.DataFrame
    """

    model = Recommender.adapt(
        als.ImplicitMF(n_factors, iterations=n_iterations, progress=tqdm)
    )
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
    

def generate_recommendations(model, train, test, n_recommendations=50, mode='all'):
    """Generate recommendations for a given model
    
    :param mode: 'all' or 'test'. Generate recommendations for all users
        in dataset or just for users in test data
    """

    if mode == 'all':
        users = test.user.unique()
    elif mode == 'test':
        users = pd.concat([train, test]).user.unique()
    else:
        raise NotImplementedError(f'Recommendation mode {mode} is not implemented')

    return batch.recommend(model, users, n_recommendations)

