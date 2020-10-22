import os
import pickle
from pathlib import Path

import implicit
import argparse
import numpy as np
from scipy.sparse import csr_matrix

from utils import print_song_info
from utils import create_msd_graph
from utils import get_msd_song_info


def main(base_path):
    graph, (n_users, n_songs, n_categories) = create_msd_graph(
        persist_file=base_path.joinpath('graph.pickle')
    )
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

    # Recall or compute and save the model
    model_path = base_path.joinpath('model.pickle')

    if model_path.is_file():
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = implicit.als.AlternatingLeastSquares(factors=20)
        model.fit(user_songs)

        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

    user_listenings = [(user, len(songs.keys())) for user, songs in graph.graphs[0][1].items()]
    user_id, max_listening = max(user_listenings, key=lambda x: x[1])
    print((user_id, max_listening))

    recommendations = model.recommend(user_id, user_songs, N=10)
    print(recommendations)

    listened_songs_ids = [node_id for node_id in graph.graphs[0][1][user_id].keys()]
    print_song_info(listened_songs_ids, graph, song_info)

    recommended_songs_ids = [node_id for node_id, _ in recommendations]
    print_song_info(recommended_songs_ids, graph, song_info)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--output-dir', type=str, default='.',
        help='Where to save generated files'
    )
    opts = p.parse_args()
    main(Path(opts.output_dir))