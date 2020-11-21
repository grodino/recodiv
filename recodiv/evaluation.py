import itertools

import numpy as np
from tqdm import tqdm
from recmetrics.metrics import mark
from recmetrics.metrics import novelty
from recmetrics.metrics import prediction_coverage


def mean_percentile_rank(test_set, recommended_items):
    """Compute the mean percentile rank
    
    :param test_set: scipy.sparse.coo_matrix. Dataset to use for evaluation.
        test_set[user[i], item[j]] = number of interactions of user i with item j

    :param recommended_items: 2D array with recommended_items[i, j] = j'th best item for user i

    :returns: the mean percentile rank in percentage (value between 0 and 100)
    """

    # Convert to Compressed Sparse Row for efficient row slicing/indexing
    test_set = test_set.tocsr()
    n_users, n_items = test_set.shape
    _, n_recommend = recommended_items.shape


    # MPR = \frac{\sum_{u, i} r^t_{u,i} rank_{u,i}}{\sum_{u, i} r^t_{u,i}}
    # weighted_rank = \sum_{u, i} r^t_{u,i} rank_{u,i}
    # total_rates = \sum_{u, i} r^t_{u,i}
    weighted_rank = 0
    total_ratings = 0

    rng = np.random.default_rng()

    for user_id, user_recommended_items in tqdm(enumerate(recommended_items), total=n_users):
        test_items = test_set[user_id, :]
        user_recommended_items.sort()

        # For the items that are not recommended, the rank is distributed with steps 1/n_items
        not_recommended_rank = rng.integers(n_recommend, n_users, size=n_items - n_recommend)
        id_not_recommended = 0

        for item_id, rating in zip(test_items.indices, test_items.data):
            found_ids = np.where(user_recommended_items == item_id)[0]
            
            if len(found_ids) > 0:
                item_rank = found_ids[0]
            else:
                # item_rank = not_recommended_rank[id_not_recommended]
                # id_not_recommended += 1
                item_rank = 0
            
            weighted_rank += rating * item_rank / test_items.shape[0]
            total_ratings += rating

    return (weighted_rank * 100) / total_ratings


def random_model(n_users, n_items, n_recommendations):
    """Recommend n_recommandations to each user by sampling random items
    
    :param n_users: int, number of users
    :param n_items: int, number of items
    :param n_recommandations: int, number of items to recommend to each user

    :returns: 2D matrix of shape (n_users, n_recommendations) of items recommended
        to each users
    """

    rng = np.random.default_rng()
    return rng.integers(0, n_items, size=(n_users, n_recommendations))


def evaluate_model(n_users, n_items, ratings, recommendations):
    """Compute several metrics on the output of a model
    
    :param n_users: int, number of users
    :param n_items: int, number of items
    :param ratings: 2D csr_matrix of shape (n_users, n_items) reprensenting the 
        interactions of users with items
    :param recommended_items: 2D array with recommended_items[i, j] = j'th best 
        item for user i

    :returns: list of metrics
    """

    n_recommendations = recommendations.shape[1]
    random_recomendations = random_model(n_users, n_items, n_recommendations)
    ratings = ratings.tocoo()
    ratings.eliminate_zeros()

    rated_by_user = [list() for _ in range(n_users)]
    for user_id, item_id in zip(ratings.row, ratings.col):
        rated_by_user[user_id].append(item_id)

    ratings_counts = np.bincount(ratings.data, minlength=n_items)
    ratings_counts = dict(enumerate(ratings_counts))
    
    for key, value in ratings_counts.items():
        if value == 0:
            ratings_counts[key] = n_users

    print('Computing metrics ...', end=' ', flush=True)
    metrics =  [
        (
            mark(rated_by_user, recommendations),
            mark(rated_by_user, random_recomendations)
        ),
        (
            prediction_coverage(recommendations, range(n_items)),
            prediction_coverage(random_recomendations, range(n_items))
        ),
        (
            novelty(recommendations, ratings_counts, n_users, n_recommendations)[0],
            novelty(random_recomendations, ratings_counts, n_users, n_recommendations)[0]
        )
    ]
    print('Done')

    return metrics



if __name__ == '__main__':
    from scipy.sparse import coo_matrix

    N_RATINGS = 100_000
    N_USERS = 1_000
    N_ITEMS = 10_000
    N_RECOMMEND = 10
    MAX_RATING = 100

    rng = np.random.default_rng()
    ratings = np.array(rng.power(0.5, size=N_RATINGS) * MAX_RATING, dtype=int)

    users_active = rng.integers(0, N_USERS, size=N_RATINGS)
    items_rated = rng.integers(0, N_ITEMS, size=N_RATINGS)

    test_set = coo_matrix(
        (ratings, (users_active, items_rated)), shape=(N_USERS, N_ITEMS)
    )

    recommended_items = np.zeros((N_USERS, N_RECOMMEND), dtype=int)
    for user_id in range(N_USERS):
        recommended_items[user_id, :] = rng.integers(0, N_ITEMS, size=N_RECOMMEND, dtype=int)

    print(mean_percentile_rank(test_set, recommended_items))