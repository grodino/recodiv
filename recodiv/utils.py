import os
import pickle
from pathlib import Path

import implicit
import progressbar
import numpy as np
from matplotlib import pyplot as pl
from scipy.sparse import csr_matrix

from recodiv.triversity.graph import IndividualHerfindahlDiversities


def get_msd_song_info():
    """Reads and returns the artist and title of each songs identified by its
    hash in the MSD"""

    songs_info = {}
    songs_file = Path('recodiv/data/million_songs_dataset/extra/unique_tracks.txt')

    with open(songs_file, 'r', encoding='utf8') as file:
        for line in file.readlines():
            song_hash, _, artist, song_title = line.rstrip('\n').split(
                sep='<SEP>')
            songs_info[song_hash] = (artist, song_title)

    return songs_info


def print_song_info(songs_ids, graph, songs_info):
    """Print song title artist and tags"""

    songs_hashes = graph.id_to_hash(songs_ids, 1)

    print('Songs :')
    for song_id, song_hash in zip(songs_ids, songs_hashes):
        artist, title = songs_info[song_hash]
        print(f'\t[{artist}] {title}')

        print('\t\t', end='')
        for category_id in graph.graphs[1][2][song_id].keys():
            print(graph.id_to_hash(category_id, 2), end=' ')
        print()


def confidence_matrix(graph, n_users, n_songs):
    """Returns the sparse confidence matrix of user tastes associated to graph
    """

    # confidence = 1 + CONFIDENCE_FACTOR*n_listenings
    CONFDENCE_FACTOR = 40

    users = []
    listened_songs = []
    confidences = []

    for user, songs in graph.graphs[0][1].items():
        for song, occurences in songs.items():
            users.append(user)
            listened_songs.append(song)
            confidences.append(1 + CONFDENCE_FACTOR * occurences)

    # Lines = users, columns = songs
    return csr_matrix(
        (confidences, (users, listened_songs)),
        shape=(n_users, n_songs)
    )


def create_msd_graph(recall_file=None):
    """Creates and returns the tri-partite graph associated to the Million Songs
    Dataset along some metadata

    :param recall_file: the file path to from which to recall the graph

    :returns: graph, (n_users, n_songs, n_categories)
    """

    # EXPERIMENT CONSTANTS #####################################################
    DATASET_FOLDER = 'recodiv/data/million_songs_dataset/'
    N_ENTRIES = 1_000_000  # Number of data entries to read
    # Number n of nodes sets (or "layers") of the n-partite graph: users, songs,
    # categories and recommendations
    N_LAYERS = 4

    # GRAPH BUILDING ###########################################################
    if not(recall_file):
        graph = IndividualHerfindahlDiversities.from_folder(
            DATASET_FOLDER,
            N_LAYERS,
            n_entries=[0, N_ENTRIES]
        )

    else:
        graph = IndividualHerfindahlDiversities.recall(recall_file)


    n_users = len(graph.ids[0])
    n_songs = len(graph.ids[1])
    n_categories = len(graph.ids[2])

    print(f'{n_users} users')
    print(f'{n_songs} songs')
    print(f'{n_categories} categories')


    n_links = [[sum([len(items[1]) for items in destination.items()]) for destination in origin] for origin in graph.graphs]
    print(f'{n_links[0][1]} user -> song links')
    print(f'{n_links[0][3]} user -> recommendation links')
    print(f'{n_links[1][2]} song -> category links')
    print(f'{n_links[3][2]} recommendation -> category links')

    user_volume = [
        (user, len(songs.keys())) for user, songs in graph.graphs[0][1].items()
    ]
    _, max_volume = max(user_volume, key=lambda x: x[1])
    _, min_volume = min(user_volume, key=lambda x: x[1])
    mean_volume = sum(volume for _, volume in user_volume) / n_users
    print(f'Minimum number of songs listened by a user : {min_volume}')
    print(f'Maximum number of songs listened by a user : {max_volume}')
    print(f'Average number of songs listened by a user : {mean_volume}')

    tag_volume = [
        (song, len(tags.keys())) for song, tags in graph.graphs[1][2].items()
    ]
    _, max_volume = max(tag_volume, key=lambda x: x[1])
    _, min_volume = min(tag_volume, key=lambda x: x[1])
    mean_volume = sum(volume for _, volume in tag_volume) / n_songs
    print(f'Minimum number of tags for a song : {min_volume}')
    print(f'Maximum number of tags for a song : {max_volume}')
    print(f'Average number of tags for a song : {mean_volume}')

    return graph, (n_users, n_songs, n_categories)


def train_msd_collaborative_filtering(graph, n_users, n_songs):
    """Train model and return it"""

    N_LATENT_FACTORS = 20

    song_info = get_msd_song_info()
    confidence = confidence_matrix(graph, n_users, n_songs)

    # Optimization recommended by implicit
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    model = implicit.als.AlternatingLeastSquares(
        factors=N_LATENT_FACTORS,
        use_cg=True,
        iterations=5,
        calculate_training_loss=True
    )
    model.fit(confidence.T.tocsr())

    pl.plot(model.loss)
    pl.show()

    return model


def recommendations_graph(graph, model, n_users, n_songs, n_recommendations):
    """Inserts the recommendations layer to the existing user-song-category
    graph"""

    confidence = confidence_matrix(graph, n_users, n_songs)
    recommendations = model.recommend_all(
        confidence, N=n_recommendations
    )

    print(recommendations)

    # TODO : test model.recommend_all() to see if their is an improvement
    for user_id, user_recommendations in progressbar.progressbar(enumerate(recommendations)):
        for song_id in user_recommendations:
            # create user -> recommendations link
            graph.add_link(0, user_id, 3, song_id, weight=1, index_node=False)

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
