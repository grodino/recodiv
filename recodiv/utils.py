from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as pl
from sklearn.linear_model import LinearRegression

from recodiv.model import rank_to_weight
from recodiv.triversity.graph import IndividualHerfindahlDiversities


def get_msd_song_info():
    """Reads and returns the artist and title of each songs identified by its
    hash in the MSD"""

    songs_info = {}
    songs_file = Path('data/million_songs_dataset/extra/unique_tracks.txt')

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


def generate_graph(user_item: pd.DataFrame, item_tag: Optional[pd.DataFrame]=None, graph=None) -> IndividualHerfindahlDiversities:
    """Generate a graph from user -> item and item -> tag links"""

    if graph is None:
        graph = IndividualHerfindahlDiversities(3)

    for link in tqdm(user_item.itertuples(), desc="user->item", total=len(user_item)):
        graph.add_link(0, link.user, 1, link.item, link.rating)

    if item_tag is not None:
        for link in tqdm(item_tag.itertuples(), desc="item->tag", total=len(item_tag)):
            graph.add_link(1, link.item, 2, link.tag, link.weight)

    return graph


def generate_recommendations_graph(recommendations: pd.DataFrame, item_tag: Optional[pd.DataFrame]=None) -> IndividualHerfindahlDiversities:
    """Generate the user-recommendations-tags graph"""

    n_recommendations = int(recommendations['rank'].max())

    reco_user_item = recommendations.copy()
    reco_user_item['rating'] = n_recommendations - recommendations['rank'] + 1

    return generate_graph(reco_user_item[['user', 'item', 'rating']], item_tag)


def linear_regression(data: pd.DataFrame, x: str, y: str) -> Tuple[float, float]:
    """Return the coefficients of a 2D linear regression on data

    Find a, b such that y = a x + b
    
    :param data: dataframe containing the data to fit
    :param x: the name of the column to use for the x variable
    :param y: the name of the column to use for the y variable

    :returns: the coefficients (a, b)
    """

    reg = LinearRegression().fit(data[x].to_numpy().reshape(-1, 1), data[y].to_numpy())
    
    return reg.coef_[0], reg.intercept_


def build_recommendations_listenings_graph(
    listenings_graph: IndividualHerfindahlDiversities, 
    user_item: pd.DataFrame, 
    recommendations: pd.DataFrame) -> IndividualHerfindahlDiversities:
    """Build the recommendations + listenings graph

    user_item is needed to access the user listening volume
    """
    
    # Normalise the recommendations by the volume the user had prior to the
    # recommendations
    reco_user_item = rank_to_weight(user_item, recommendations)[['user', 'item', 'weight']] \
        .rename(columns={'weight': 'rating'})

    # No need to give the item_tag info because it is already in the graph
    return generate_graph(reco_user_item, item_tag=None, graph=listenings_graph)


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


def plot_histogram(values, min_quantile=.1, max_quantile=.9, n_bins=100, ax=None, log=False):
    """Plot the histogram of values with information on mean

    :param values: np.ndarray (N,) containing the values from which to compute
        the histogram.
    :param min_quantile: float. Ignore the values that are smaller than 
        (1 - min_quantile)*100 percent of the values.
    :param max_quantile: float. Ignore the values that are higher than 
        max_quantiles*100 percent of the values.
    :param n_bins: the number of bins of the histogram. See 
        matplotlib.pyplot.hist
    :param ax: The matplolib axe to plot on. If None, the Figure and Axes are 
        created and returned
    :param log: Set the y axis (the counts) in logscale

    :returns: (matplotlib.figure.Figure or None, matplotlib.axes.Axes). If ax is
        not given, return Figure and Axes objects, else return None and Axes 
        object
    """

    min_value, max_value = np.quantile(values, [min_quantile, max_quantile])
    mean = np.mean(values)

    if ax == None:
        fig, ax = pl.subplots()
    else:
        fig = None

    ax.hist(values[(min_value < values) & (values < max_value)], bins=n_bins, log=log)
    ax.axvline(mean, ls='--', color='pink')
    ax.text(mean + 1, 2, f'mean: {mean:.02f}', color='pink')

    return fig, ax