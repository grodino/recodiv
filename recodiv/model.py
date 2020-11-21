import os
import pickle
from pathlib import Path

import implicit
from implicit.evaluation import train_test_split
from implicit.evaluation import ranking_metrics_at_k
import progressbar
import numpy as np
from matplotlib import pyplot as pl
from scipy.sparse import csr_matrix

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from recodiv.utils import print_song_info
from recodiv.utils import get_msd_song_info
from recodiv.evaluation import mean_percentile_rank


def confidence_matrix(graph, n_users, n_songs):
    """Returns the sparse confidence matrix of user tastes associated to graph
    """

    # confidence = 1 + CONFIDENCE_FACTOR*n_listenings
    CONFIDENCE_FACTOR = 40

    users = []
    listened_songs = []
    confidences = []

    for user, songs in graph.graphs[0][1].items():
        for song, occurrences in songs.items():
            users.append(user)
            listened_songs.append(song)
            confidences.append(1 + CONFIDENCE_FACTOR * occurrences)

    # Lines = users, columns = songs
    return csr_matrix(
        (confidences, (users, listened_songs)),
        shape=(n_users, n_songs)
    )


def train_msd_collaborative_filtering(graph, n_users, n_songs):
    """Train model and return it"""

    N_LATENT_FACTORS = 100

    song_info = get_msd_song_info()
    confidence = confidence_matrix(graph, n_users, n_songs)

    # Optimization recommended by implicit
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # model = implicit.als.AlternatingLeastSquares(
    #     regularization=0.015,
    #     iterations=1,
    #     factors=N_LATENT_FACTORS,
    #     use_cg=True,
    #     calculate_training_loss=True
    # )
    model = implicit.lmf.LogisticMatrixFactorization(
        factors=N_LATENT_FACTORS,
        learning_rate=1.00,
        regularization=0.6,
        dtype=np.float32,
        iterations=30,
        neg_prop=30
    )

    train_confidence, test_confidence = train_test_split(
        confidence,
        train_percentage=0.8
    )

    model.fit(train_confidence.T.tocsr())
    

    # # Best regularization = 0.015 in [0.008, 0.009, 0.01, 0.015, 0.02]
    # for epoch in range(50):
    #     # model = implicit.lmf.LogisticMatrixFactorization(
    #     #     factors=N_LATENT_FACTORS,
    #     #     learning_rate=1.00,
    #     #     regularization=0.6,
    #     #     dtype=np.float32,
    #     #     iterations=30,
    #     #     neg_prop=30
    #     # )

    #     model.fit(train_confidence.T.tocsr())

    #     # metrics.append(ranking_metrics_at_k(
    #     #     model,
    #     #     train_confidence.T.tocsr(),
    #     #     test_confidence.T.tocsr(),
    #     #     K=10,
    #     #     show_progress=True
    #     # ))

    #     recommendations = model.recommend_all(
    #         train_confidence,
    #         N=10,
    #         filter_already_liked_items=False,
    #         show_progress=True
    #     )
    #     MPR = mean_percentile_rank(test_confidence, recommendations)
    #     print(MPR)


    return model, train_confidence, test_confidence


def recommendations_graph(graph, model, n_users, n_songs, n_recommendations):
    """Inserts the recommendations layer to the existing user-song-category
    graph"""

    confidence = confidence_matrix(graph, n_users, n_songs)
    recommendations = model.recommend_all(
        confidence, N=n_recommendations, filter_already_liked_items=False
    )

    # TODO : test model.recommend_all() to see if their is an improvement
    for user_id, user_recommendations in progressbar.progressbar(enumerate(recommendations)):
        for rank, song_id in enumerate(user_recommendations):
            # create user -> recommendations link
            graph.add_link(0, user_id, 3, song_id, weight=1/(rank + 1), index_node=False)

            # Create recommendation -> tags links
            # try:
            #     tags = graph.graphs[1][2][song_id]
            #     for tag_id, weight in tags.items():
            #         graph.add_link(3, song_id, 2, tag_id, weight, index_node=False)
            # except KeyError:
            #     pass
            tags = graph.graphs[1][2][song_id]
            for tag_id, weight in tags.items():
                graph.add_link(3, song_id, 2, tag_id, weight, index_node=False)

    return graph
