from pathlib import Path

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


def dataset_info(graph):
    """Returns information on the dataset (number of users, links ...)"""

    n_users = len(graph.ids[0])
    n_songs = len(graph.ids[1])
    n_tags = len(graph.ids[2])

    n_user_song_links = graph.n_links(0, 1)
    n_song_tag_links = graph.n_links(1, 2)

    user_song_volume = [
        (user, len(songs.keys())) for user, songs in graph.graphs[0][1].items()
    ]
    _, max_user_song_volume = max(user_song_volume, key=lambda x: x[1])
    _, min_user_song_volume = min(user_song_volume, key=lambda x: x[1])
    mean_user_song_volume = sum(volume for _, volume in user_song_volume) / n_users

    song_tag_volume = [
        (song, len(tags.keys())) for song, tags in graph.graphs[1][2].items()
    ]
    _, max_song_tag_volume = max(song_tag_volume, key=lambda x: x[1])
    _, min_song_tag_volume = min(song_tag_volume, key=lambda x: x[1])
    mean_song_tag_volume = sum(volume for _, volume in song_tag_volume) / n_songs

    return {
        'n_users': n_users,
        'n_songs': n_songs,
        'n_tags': n_tags,
        'n_users_song_links': n_user_song_links,
        'n_song_tag_links': n_song_tag_links,
        'max_user_song_volume': max_user_song_volume,
        'max_song_tag_volume': max_song_tag_volume,
        'min_user_song_volume': min_user_song_volume,
        'min_song_tag_volume': min_song_tag_volume,
        'mean_user_song_volume': mean_user_song_volume,
        'mean_song_tag_volume': mean_song_tag_volume,
    }
