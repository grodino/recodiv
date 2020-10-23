import os
import pickle
from pathlib import Path

import implicit
import numpy as np
from scipy.sparse import csr_matrix

from recodiv.triversity.graph import IndividualHerfindahlDiversities


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
    print(f'{n_links[1][2]} song -> category links')

    return graph, (n_users, n_songs, n_categories)


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
    songs_hashes = graph.id_to_hash(songs_ids, 1)

    print('Songs :')
    for song_id, song_hash in zip(songs_ids, songs_hashes):
        artist, title = songs_info[song_hash]
        print(f'\t[{artist}] {title}')

        print('\t\t', end='')
        for category_id in graph.graphs[1][2][song_id].keys():
            print(graph.id_to_hash(category_id, 2), end=' ')
        print()


def train_msd_collaborative_filtering(graph, n_users, n_songs):
    """Train model and return it"""
    song_info = get_msd_song_info()

    users = []
    listened_songs = []
    listenings = []

    for user, songs in graph.graphs[0][1].items():
        for song, occurences in songs.items():
            users.append(user)
            listened_songs.append(song)
            listenings.append(occurences)

    users = np.array(users, dtype=np.float32)
    listened_songs = np.array(listened_songs, dtype=np.float32)
    listenings = np.array(users, dtype=np.float32)

    # Lines = users, columns = songs
    user_songs = csr_matrix(
        (listenings, (users, listened_songs)),
        shape=(n_users, n_songs)
    )

    # Optimization recommended by implicit
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    model = implicit.als.AlternatingLeastSquares(factors=20)
    model.fit(user_songs)

    user_listenings = [(user, len(songs.keys())) for user, songs in
                       graph.graphs[0][1].items()]
    user_id, max_listening = max(user_listenings, key=lambda x: x[1])
    print((user_id, max_listening))

    recommendations = model.recommend(user_id, user_songs, N=10)
    print(recommendations)

    listened_songs_ids = [node_id for node_id in
                          graph.graphs[0][1][user_id].keys()]
    print_song_info(listened_songs_ids, graph, song_info)

    recommended_songs_ids = [node_id for node_id, _ in recommendations]
    print_song_info(recommended_songs_ids, graph, song_info)

    return model