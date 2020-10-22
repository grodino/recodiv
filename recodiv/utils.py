import pickle
from pathlib import Path

from triversity.graph import IndividualHerfindahlDiversities


def create_msd_graph(persist_file=None):
    """Creates and returns the tri-partite graph associated to the Million Songs
    Dataset along some metadata

    :param persist_folder: if given, check if the graph was persisted before

    :returns: created graph, (number of users, number of songs, number of
        categories)
    """

    # EXPERIMENT CONSTANTS #####################################################
    DATASET_FOLDER = 'recodiv/data/million_songs_dataset/'
    N_ENTRIES = 1_000_000  # Number of data entries to read
    # Number n of nodes sets (or "layers") of the n-partite graph: users, songs,
    # categories and recommendations
    N_LAYERS = 4

    # GRAPH BUILDING ###########################################################
    if not(persist_file):
        graph = IndividualHerfindahlDiversities.from_folder(
            DATASET_FOLDER,
            N_LAYERS,
            n_entries=[0, N_ENTRIES]
        )

    else:
        persist_file = Path(persist_file)

        if persist_file.is_file():
            graph = IndividualHerfindahlDiversities.recall(persist_file)
        else:
            graph = IndividualHerfindahlDiversities.from_folder(
                DATASET_FOLDER,
                N_LAYERS,
                n_entries=[0, N_ENTRIES]
            )
            graph.persist(persist_file)


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
