import os
import pickle
from pathlib import Path
from time import perf_counter

import numba as nb
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
        n_factors=30,
        n_iterations=20,
        regularization=.1,
        save_training_loss=False,
        confidence_factor=40):
    """Train (and evaluate iterations if requested) model"""

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

    if save_training_loss:
        loss = np.zeros(n_iterations)

        for i, intermediate_model in enumerate(model.fit_iters(train)):
            predictions = generate_predictions(intermediate_model, train)
            loss[i] = evaluate_model_loss(intermediate_model, predictions)

    else:
        model.fit(train)
        loss = None

    return model, loss


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


def evaluate_model_recommendations(recommendations, test, metrics) -> pd.DataFrame:
    """Evaluates a model via its recommendations

    :param recommendations: pd.DataFrame with at least the following columns :  
        'user', 'item', 'score', 'rank'
    :param test: pd.DataFrame. The testing data
    :param metrics: list. A list of metrics' names (see recodiv.model.METRICS)
    """

    analysis = topn.RecListAnalysis(n_jobs=20)
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
        np.linalg.norm(model.predictor.user_features_, 'fro')
        + np.linalg.norm(model.predictor.item_features_, 'fro')
    )

    return confidence @ (1 - prediction)**2 + reg


def rank_to_weight(user_item, recommendations):
    """Compute the weight associated to each recommendation for each user in 
       recommendations

    :param user_item: pd.DataFrame(columns=['user', 'item', 'rating']). All the
        known user-item listenings counts
    :param recommendations: pd.DataFrame(columns=['user', 'item', 'rank']). All
        the recommendations made to each user in recommendations.

    :returns: the recommendations DataFrame with the column ['weight']
    """

    n_r = recommendations['rank'].max()  # n_recommendations
    users = recommendations.user.unique()
    n_users = users.shape[0]

    # get the volume of the users in the recommendations DataFrame
    user_volume = user_item.groupby('user')['rating'].sum()

    def user_weights(x):
        # x.name is the id of the user
        return (2 * user_volume[x.name] / (n_r * (n_r - 1))) * (n_r - x)

    recommendations['weight'] = recommendations.groupby(
        'user')['rank'].transform(user_weights)

    return recommendations


@nb.njit
def wasserstein_1d(distribution: np.ndarray, other: np.ndarray) -> float:
    """Compute the Optimal Transport Distance between histograms

    We assume that distribution a sorted in increasing index and have the same
    total weight
    """

    work = w_sum = u_sum = r = 0

    i = j = 0

    while i < distribution.shape[0] and j < other.shape[0]:
        if i <= j:
            work += np.abs(w_sum - u_sum) * (i - r)
            w_sum += distribution[i]

            r = i
            i += 1

        else:
            work += np.abs(w_sum - u_sum) * (j - r)
            u_sum += other[j]

            r = j
            j += 1

    return work / u_sum


def tags_distance(distribution: pd.Series, other: pd.Series, tags: pd.Index, p=1):
    """Compute the Optimal Transport Distance between histograms (see
    https://arxiv.org/pdf/1803.00567.pdf p.30-33)
    """

    if p < 1:
        raise ValueError('p must be greater or equal that 1')
    if p != 1:
        raise NotImplementedError('Only wasserstein 1 is currently supported')

    # Make the tag distributions have the same support
    distrib = distribution.reindex(index=tags, fill_value=0)
    other = other.reindex(index=tags, fill_value=0)

    # Sort by tag (in the lexicographic order) and normalize the distributions
    # This is important because in the distance we implicitly associate a tag to
    # a point in N.
    distrib = distrib.sort_index()
    distrib = distrib / distrib.sum()

    other = other.sort_index()
    other = other / other.sum()

    # print(distrib, other, sep='\n')

    return wasserstein_1d(distrib.to_numpy(), other.to_numpy())


if __name__ == '__main__':
    tags = pd.Index(['rock', 'pop', 'jazz', 'metal', 'classical'])
    distribution = pd.Series({'rock': 3, 'pop': 10, 'jazz': 1})
    other = pd.Series({'rock': 10, 'pop': 1, 'jazz': 5})

    other_unbalanced = pd.Series(
        {'rock': 10, 'pop': 1, 'jazz': 5, 'metal': 10})

    print(tags_distance(distribution, other, tags, p=2))
    print(tags_distance(distribution, other_unbalanced, tags, p=2))
