import os

import implicit
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as pl

from triversity import IndividualHerfindahlDiversities


# EXPERIMENT CONSTANTS #########################################################
DATASET_FOLDER = './data/million_songs_dataset/'
N_ENTRIES = 1_000_000 # Number of data entries to read 
N_LAYERS = 3 # Number n of nodes sets (or "layers") of the n-partite graph


# GRAPH BUILDING ###############################################################
graph = IndividualHerfindahlDiversities.from_folder(
    DATASET_FOLDER,
    N_LAYERS,
    n_entries=[N_ENTRIES, 0]
)
graph.normalise_all()

n_users = len(graph.ids[0])
n_songs = len(graph.ids[1])
n_categories = len(graph.ids[2])

print(f'{n_users} users')
print(f'{n_songs} songs')
print(f'{n_categories} categories')

n_links = [[sum([len(items[1]) for items in destination.items()]) for destination in origin] for origin in graph.graphs]
print(f'{n_links[0][1]} user -> song links')
print(f'{n_links[1][2]} song -> category links')

# COMPUTE DIVERSITY ############################################################
diversities = graph.spread_and_divs((0, 1, 2), save=True)
print(diversities[0], diversities[1_000])
print(graph.graphs[0][1][1])
# print(graph.saved, graph.res)


# PLOT RESULTS #################################################################
# test = list(diversities.values())
# test.remove(max(test))
# test.remove(max(test))
# test.remove(max(test))
# test.remove(max(test))
# test.remove(max(test))
# test.remove(max(test))

# pl.hist(test, bins=100)
# pl.show()


# RECOMMENDER TEST #############################################################
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

listenings_matrix = csr_matrix(
    (listenings, (users, listened_songs)),
    shape=(n_users, n_songs)
)

# Optimization recommended by implicit
os.environ['OPEN_BLAS_NUM_THREADS'] = '1'
model = implicit.als.AlternatingLeastSquares(factors=20)
model.fit(listenings_matrix)