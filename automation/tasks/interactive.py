import json
from pathlib import Path

import luigi
import numpy as np
import pandas as pd

from recodiv.utils import get_msd_song_info
from recodiv.triversity.graph import IndividualHerfindahlDiversities
from automation.tasks.dataset import Dataset, ImportDataset
from automation.tasks.model import GenerateRecommendations
from automation.tasks.traintest import BuildTrainTestGraphs, ComputeTrainTestUserDiversity, GenerateTrainTest
from automation.tasks.recommendations import BuildRecommendationGraph, BuildRecommendationsWithListeningsGraph, ComputeRecommendationDiversities, ComputeRecommendationWithListeningsUsersDiversityIncrease


################################################################################
# INTERACTIVE PLOTTING                                                         #
################################################################################
class ComputeRecommendationDiversityVsUserDiversityVsLatentFactors(luigi.Task):
    """Compute the diversity of the recommendations associated to each user with
    respect to the user diversity before recommendations for different number of
    factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors_values = luigi.parameter.ListParameter(
        description='Values of number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'user_diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction
            ),
        }

        for n_factors in self.model_n_factors_values:
            req[f'{n_factors}-recommendation_diversity'] = ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return req

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.n_recommendations}reco-{self.model_n_factors_values}factors-users_diversities{self.alpha}.csv')
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input()['user_diversity']['train'].path)

        # compute user volume
        user_item = pd.read_csv(self.input()['train_test']['train'].path)
        volume = np.log10(user_item.groupby('user')['rating'].sum()) \
            .rename('volume')

        # Get the diversity values for the different number of factors
        reco_diversities = []

        for n_factors in self.model_n_factors_values:
            divs = pd.read_csv(
                self.input()[f'{n_factors}-recommendation_diversity'].path
            ).rename(columns={'diversity': 'reco_diversity'})

            divs['n_factors'] = n_factors
            reco_diversities.append(divs)

        reco_diversities: pd.DataFrame = pd.concat(
            reco_diversities, ignore_index=True)

        # inner join, only keep users for whom we calculated a recommendation diversity value
        merged = reco_diversities.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')
        merged.to_csv(self.output().path)

        return merged


class ComputeDiversityIncreaseVsUserDiversityVsLatentFactors(luigi.Task):
    """Compute the diversity of the recommendations associated to each user with
    respect to the user diversity before recommendations for different number of
    factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors_values = luigi.parameter.ListParameter(
        description='Values of number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'user_diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction
            ),
        }

        for n_factors in self.model_n_factors_values:
            req[f'{n_factors}-diversity_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return req

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.n_recommendations}reco-{self.model_n_factors_values}factors-diversity{self.alpha}_increase.csv')
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input()['user_diversity']['train'].path)

        # compute user volume
        user_item = pd.read_csv(self.input()['train_test']['train'].path)
        volume = np.log10(user_item.groupby('user')['rating'].sum()) \
            .rename('volume')

        # Get the diversity values for the different number of factors
        deltas = []

        for n_factors in self.model_n_factors_values:
            divs = pd.read_csv(
                self.input()[f'{n_factors}-diversity_increase'].path
            ).rename(columns={'diversity': 'diversity_increase'})

            divs = divs[divs['diversity_increase'] != 0]

            divs['n_factors'] = n_factors
            deltas.append(divs)

        deltas: pd.DataFrame = pd.concat(deltas, ignore_index=True)

        # inner join, only keep users for whom we calculated a recommendation diversity value
        merged = deltas.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')
        merged.to_csv(self.output().path)

        return merged


class ComputeRecommendationDiversityVsUserDiversityVsRecoVolume(luigi.Task):
    """Compute the diversity of the recommendations associated to each user with
    respect to the user diversity before recommendations for different number of
    recommendations per user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate per user'
    )

    def requires(self):
        req = {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'user_diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction
            ),
        }

        for n_recommendations in self.n_recommendations_values:
            req[f'{n_recommendations}-recommendation_diversity'] = ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=n_recommendations
            )

        return req

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.n_recommendations_values}reco-{self.model_n_factors}factors-users_diversities{self.alpha}.csv')
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input()['user_diversity']['train'].path)

        # compute user volume
        user_item = pd.read_csv(self.input()['train_test']['train'].path)
        volume = np.log10(user_item.groupby('user')['rating'].sum()) \
            .rename('volume')

        # Get the diversity values for the different number of factors
        reco_diversities = []

        for n_recommendations in self.n_recommendations_values:
            divs = pd.read_csv(
                self.input()[
                    f'{n_recommendations}-recommendation_diversity'].path
            ).rename(columns={'diversity': 'reco_diversity'})

            divs['n_recommendations'] = n_recommendations
            reco_diversities.append(divs)

        reco_diversities: pd.DataFrame = pd.concat(
            reco_diversities, ignore_index=True)

        # inner join, only keep users for whom we calculated a recommendation diversity value
        merged = reco_diversities.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')
        merged.to_csv(self.output().path)

        return merged


class ComputeDiversityIncreaseVsUserDiversityVsRecoVolume(luigi.Task):
    """Compute the diversity increase associated to each user with respect to
    the user diversity before recommendations for different number of
    recommendations per user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate per user'
    )

    def requires(self):
        req = {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'user_diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction
            ),
        }

        for n_recommendations in self.n_recommendations_values:
            req[f'{n_recommendations}-diversity_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=n_recommendations
            )

        return req

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.n_recommendations_values}reco-{self.model_n_factors}factors-diversity{self.alpha}_increase.csv')
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input()['user_diversity']['train'].path)

        # compute user volume
        user_item = pd.read_csv(self.input()['train_test']['train'].path)
        volume = np.log10(user_item.groupby('user')['rating'].sum()) \
            .rename('volume')

        # Get the diversity values for the different number of factors
        deltas = []

        for n_recommendations in self.n_recommendations_values:
            divs = pd.read_csv(
                self.input()[f'{n_recommendations}-diversity_increase'].path
            ).rename(columns={'diversity': 'diversity_increase'})

            divs = divs[divs['diversity_increase'] != 0]

            divs['n_recommendations'] = n_recommendations
            deltas.append(divs)

        deltas: pd.DataFrame = pd.concat(deltas, ignore_index=True)

        # inner join, only keep users for whom we calculated a recommendation diversity value
        merged = deltas.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')
        merged.to_csv(self.output().path)

        return merged


class AnalyseUser(luigi.Task):
    """ Look at the items listened by a user, its diversity before and after 
        recommendation etc... """

    user_id = luigi.parameter.Parameter(
        description='The id string of the user'
    )

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return {
            'dataset': ImportDataset(dataset=self.dataset),
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'train_test_graph': BuildTrainTestGraphs(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'recommendation_graph': BuildRecommendationGraph(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'recommendation_with_listen': BuildRecommendationsWithListeningsGraph(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent
        return luigi.LocalTarget(model.joinpath(f'user_{self.user_id}-info.json'))

    def run(self):
        test = pd.read_csv(self.input()['train_test']['test'].path)
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path)
        song_info = get_msd_song_info()

        # Compute the bipartite projection of the user graph on the tags layer
        test_graph = IndividualHerfindahlDiversities.recall(
            self.input()['train_test_graph']['test'].path
        )
        test_graph.normalise_all()
        distribution = test_graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        listened_tag_distribution = pd.Series(distribution) \
            .sort_values(ascending=False)

        dist = np.array(listened_tag_distribution)
        dist = dist / np.sum(dist)
        print('ORGANIC DIVERSITY 2', 1 / np.sum(dist**2))
        print('ORGANIC DIVERSITY 0', len(dist))

        # TEST
        reco_listen_graph = IndividualHerfindahlDiversities.recall(
            self.input()['recommendation_with_listen'].path
        )
        reco_listen_graph.normalise_all()
        distribution = reco_listen_graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        print('N_TAGS', len(distribution))
        print('DIV manual', sum(1 for x in distribution.values() if x > 0))
        print('DIVERSITY all', reco_listen_graph.diversities(
            (0, 1, 2), alpha=0)[self.user_id])
        print('SUM', sum(distribution.values()))
        print(distribution)

        # Compute the bipartite projection of the recommendation graph on the
        # tags layer
        recommendation_graph = IndividualHerfindahlDiversities.recall(
            self.input()['recommendation_graph'].path
        )
        recommendation_graph.normalise_all()
        distribution = recommendation_graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        recommended_tag_distribution = pd.Series(distribution) \
            .sort_values(ascending=False)

        def track_id_to_dict(track_ids):
            items = {}

            for track_id in track_ids:
                items[track_id] = {
                    'artist': song_info[track_id][0],
                    'title': song_info[track_id][1],
                }

            return items

        info = {
            'user_id': self.user_id,
            'n_iterations': self.n_iterations,
            'model_n_factors': self.model_n_factors,
            'model_regularization': self.model_regularization,
            'model_user_fraction': self.model_user_fraction,
        }

        # Listened items
        listened_items = test[test['user'] == self.user_id]
        info['listened_items'] = track_id_to_dict(listened_items['item'])
        info['n_listened'] = len(info['listened_items'])

        # Listened tags
        tags = listened_items.merge(item_tag, how='left', on='item')
        info['listened_tags'] = list(tags.tag.unique())
        info['n_listened_tags'] = len(info['listened_tags'])

        # Recommended items
        recommended_items = recommendations[recommendations['user']
                                            == self.user_id]
        info['recommended_items'] = track_id_to_dict(recommended_items['item'])
        info['n_recommended_items'] = len(info['recommended_items'])

        # Recommended tags
        recommended_tags = recommended_items.merge(
            item_tag, how='left', on='item')
        info['recommended_tags'] = list(recommended_tags.tag.unique())
        info['n_recommended_tags'] = len(info['recommended_tags'])

        # Intersection of recommended tags and listened tags
        info['common_tags'] = list(np.intersect1d(
            recommended_tags.tag.unique(), tags.tag.unique()))
        info['n_common_tags'] = len(info['common_tags'])

        with self.output().open('w') as file:
            json.dump(info, file, indent=4)

        return info, listened_tag_distribution, recommended_tag_distribution
