import json
from pathlib import Path

import luigi
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from recodiv.utils import get_msd_song_info
from recodiv.triversity.graph import IndividualHerfindahlDiversities
from automation.tasks.dataset import Dataset, ImportDataset
from automation.tasks.model import GenerateRecommendations
from automation.tasks.traintest import BuildTrainTestGraphs, ComputeTrainTestUserDiversity, GenerateTrainTest
from automation.tasks.recommendations import BuildRecommendationGraph, BuildRecommendationsWithListeningsGraph, ComputeRecommendationDiversities, ComputeRecommendationWithListeningsUsersDiversityIncrease


class AnalyseUser(luigi.Task):
    """ Look at the items listened by a user, its diversity before and after
        recommendation etc... """

    user_id = luigi.parameter.Parameter(
        description='The id string of the user'
    )

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    alpha_values = luigi.parameter.ListParameter(
        description='Values of the true diversity order to use'
    )
    fold_id = luigi.parameter.IntParameter(
        description='Which fold to use for the data'
    )
    n_recommendation_values = luigi.parameter.ListParameter(
        description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = {}

        for alpha in self.alpha_values:
            req[alpha] = {}
            req[alpha]['diversity'] = ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                split=self.split,
                alpha=alpha,
            )

            for n_recommendations in self.n_recommendation_values:
                req[alpha][n_recommendations] = {}
                req[alpha][n_recommendations]['reco_diversity'] = ComputeRecommendationDiversities(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    alpha=alpha,
                    n_recommendations=n_recommendations,
                )
                req[alpha][n_recommendations]['diversity_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    alpha=alpha,
                    n_recommendations=n_recommendations,
                )

        return req

    def output(self):
        model = Path(
            self.input()
            [self.alpha_values[0]]
            [self.n_recommendation_values[0]]
            ['reco_diversity']
            [self.fold_id]
            .path
        ).parent

        return luigi.LocalTarget(model.joinpath(
            f'users/user_{self.user_id}'
            f'-n_recommendations_{self.n_recommendation_values}'
            f'-alpha_values_{self.alpha_values}'
            f'-info.json'
        ))

    def run(self):
        self.output().makedirs()
        info = {}

        for alpha in self.alpha_values:
            # Organic diversity
            organic_diversities = pd.read_csv(
                self.input()[alpha]['diversity'][self.fold_id]['test'].path
            ).set_index('user')
            info[f'alpha {alpha}'] = {
                'organic diversity': float(organic_diversities.loc[self.user_id]['diversity']),
                'recommendation diversity': {},
                'diversity increase': {}
            }

            for n_recommendations in self.n_recommendation_values:
                # Recommendation diversity
                reco_diversities = pd.read_csv(
                    self.input()
                    [alpha]
                    [n_recommendations]
                    ['reco_diversity']
                    [self.fold_id]
                    .path
                ).set_index('user')
                info[f'alpha {alpha}']['recommendation diversity'][n_recommendations] = \
                    float(reco_diversities.loc[self.user_id]['diversity'])

                # Diversity increase
                diversity_increase = pd.read_csv(
                    self.input()
                    [alpha]
                    [n_recommendations]
                    ['diversity_increase']
                    [self.fold_id]
                    .path
                ).set_index('user')
                info[f'alpha {alpha}']['diversity increase'][n_recommendations] = \
                    float(diversity_increase.loc[self.user_id]['diversity'])

        with self.output().open('w') as file:
            json.dump(info, file, indent=4)


class ComputeUserRecommendationsTagsDistribution(luigi.Task):
    """Compute the distributions of tags of the items recommended to a given user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user_id = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    def requires(self):
        return BuildRecommendationGraph(
            dataset=self.dataset,
            model=self.model,
            split=self.split,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input()[self.fold_id].path).parent
        folder = model.joinpath('users')

        return luigi.LocalTarget(
            folder.joinpath(
                f'{self.n_recommendations}reco-user{self.user_id}-tags-distribution.csv')
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(
            self.input()[self.fold_id].path)

        # Compute the bipartite projection of the user graph on the tags layer
        graph.normalise_all()
        distribution = graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        distribution = pd.Series(distribution, name='weight') \
            .sort_values(ascending=False)

        distribution.to_csv(self.output().path)


class ComputeUserListenedRecommendedTagsDistribution(luigi.Task):
    """Compute the tag distribution reached by a user thtough their listened and recommended items"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user_id = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    def requires(self):
        return BuildRecommendationsWithListeningsGraph(
            dataset=self.dataset,
            model=self.model,
            split=self.split,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input()[self.fold_id].path).parent
        folder = model.joinpath('users')

        return luigi.LocalTarget(
            folder.joinpath(
                f'listening-{self.n_recommendations}reco-user{self.user_id}-tags-distribution.csv')
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(
            self.input()[self.fold_id].path)

        # Compute the bipartite projection of the user graph on the tags layer
        graph.normalise_all()
        distribution = graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        distribution = pd.Series(distribution, name='weight') \
            .sort_values(ascending=False)

        distribution.to_csv(self.output().path)


class ComputeTrainTestUserTagsDistribution(luigi.Task):
    """Compute the tag distibution of items listened by given user. Sorts by
    decreasing normalized weight"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    user_id = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    def requires(self):
        return BuildTrainTestGraphs(
            dataset=self.dataset,
            split=self.split
        )

    def output(self):
        folder = self.dataset.data_folder.joinpath(
            self.split['name'], f'fold-{self.fold_id}', 'users'
        )

        return {
            'train': luigi.LocalTarget(
                folder.joinpath(
                    f'train-user{self.user_id}-tags-distribution.csv')
            ),
            'test': luigi.LocalTarget(
                folder.joinpath(
                    f'test-user{self.user_id}-tags-distribution.csv')
            ),
        }

    def run(self):
        self.output()['train'].makedirs()

        train_graph = IndividualHerfindahlDiversities.recall(
            self.input()[self.fold_id]['train'].path
        )
        test_graph = IndividualHerfindahlDiversities.recall(
            self.input()[self.fold_id]['test'].path
        )

        # Compute the bipartite projection of the user graph on the tags layer
        test_graph.normalise_all()
        test_distribution = test_graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        test_distribution = pd.Series(test_distribution, name='weight') \
            .sort_values(ascending=False)

        # Compute the bipartite projection of the user graph on the tags layer
        train_graph.normalise_all()
        train_distribution = train_graph.spread_node(
            self.user_id, (0, 1, 2)
        )
        train_distribution = pd.Series(train_distribution, name='weight') \
            .sort_values(ascending=False)

        test_distribution.to_csv(self.output()['test'].path)
        train_distribution.to_csv(self.output()['train'].path)


class PlotUserTagHistograms(luigi.Task):
    """Plot the listened, recommended and listened+recommended tags histograms
    for a user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    user_id = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    n_tags = luigi.parameter.IntParameter(
        default=30, description="The number of most represented tags showed in the histogram"
    )

    def requires(self):
        return {
            'recommended_tags': ComputeUserRecommendationsTagsDistribution(
                dataset=self.dataset,
                user_id=self.user_id,
                model=self.model,
                split=self.split,
                n_recommendations=self.n_recommendations
            ),
            'listened_tags': ComputeTrainTestUserTagsDistribution(
                dataset=self.dataset,
                split=self.split,
                user_id=self.user_id,
                fold_id=self.fold_id,
            ),
            'after_reco_tags': ComputeUserListenedRecommendedTagsDistribution(
                dataset=self.dataset,
                user_id=self.user_id,
                model=self.model,
                split=self.split,
                n_recommendations=self.n_recommendations
            ),
            'increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                model=self.model,
                split=self.split,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        folder = Path(
            self.input()['recommended_tags'].path).parent.joinpath('figures')

        return {
            'png': luigi.LocalTarget(
                folder.joinpath(
                    f'increase{self.alpha}-{self.n_recommendations}reco-user{self.user_id}-tag-histograms.png')
            ),
            'eps': luigi.LocalTarget(
                folder.joinpath(
                    f'increase{self.alpha}-{self.n_recommendations}reco-user{self.user_id}-tag-histograms.eps')
            )
        }

    def run(self):
        self.output()['png'].makedirs()

        reco_distribution: pd.DataFrame = pd.read_csv(
            self.input()['recommended_tags'].path, index_col=0
        ).reset_index().rename(columns={
            'index': 'tag', 'weight': 'recommended'
        })
        listened_distribution: pd.DataFrame = pd.read_csv(
            self.input()['listened_tags']['train'].path, index_col=0
        ).reset_index().rename(columns={
            'index': 'tag', 'weight': 'listened'
        })
        after_reco_distribution: pd.DataFrame = pd.read_csv(
            self.input()['after_reco_tags'].path, index_col=0
        ).reset_index().rename(columns={
            'index': 'tag', 'weight': 'afterreco'
        })

        # Build the index
        best_listen_tags = listened_distribution['tag'][:10]
        best_reco_tags = reco_distribution['tag'][:20]
        best_tags = best_listen_tags.append(best_reco_tags).unique()

        heaviest_tags = pd.merge(
            listened_distribution,
            reco_distribution,
            on='tag',
            how='outer'
        )
        heaviest_tags = heaviest_tags[heaviest_tags['tag'].isin(best_tags)]

        heaviest_tags_after_reco = pd.merge(
            heaviest_tags,
            after_reco_distribution,
            on='tag',
            how='left'
        )
        heaviest_tags_after_reco = after_reco_distribution[:len(heaviest_tags)]
        heaviest_tags_after_reco['afterreco'] = - \
            heaviest_tags_after_reco['afterreco']

        increase: pd.DataFrame = pd.read_csv(
            self.input()['increase'][self.fold_id].path)

        if float(increase[increase['user'] == self.user_id]['diversity']) > 0:
            color = 'green'
        else:
            color = 'red'

        width = .8
        fig, axes = pl.subplots(1, 1)

        # Plot the listened and recommended tags distributions
        ax = heaviest_tags.plot.bar(
            x='tag',
            # logy=True,
            # ax=axes[0],
            ax=axes,
            title='',
            xlabel='',
            width=width,
        )

        # Plot the recommended+listened tags distribution
        ax = heaviest_tags_after_reco.plot.bar(
            x='tag',
            y='afterreco',
            ax=ax,
            color=color,
            title='',
            xlabel='',
            width=width,
            bottom=0,
        )

        # Remove legend
        # ax.get_legend().remove()
        pl.legend(loc=4)

        # Inset plot to show the whole recommended+listened tags distribution
        axins = inset_axes(ax, width=3, height=1, loc=1)
        axins.tick_params(labelleft=False, labelbottom=False,
                          bottom=False, left=False)

        after_reco_distribution['afterreco'] = after_reco_distribution['afterreco'].abs(
        )
        # after_reco_distribution[:50].plot.bar(
        after_reco_distribution[:40].plot.bar(
            ax=axins,
            x='tag',
            y='afterreco',
            color=color,
            xlabel='',
            ylabel='',
        )
        axins.get_legend().remove()

        # Nice tags display
        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)
        pl.setp(ax.get_xticklabels(), rotation=90,
                rotation_mode="anchor", ha="right")

        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{abs(y_tick):.02f}' for y_tick in y_ticks])

        x_ticks = ax.get_xticks()
        ax.plot([x_ticks[0] - width/2, x_ticks[-1] + width/2],
                [0, 0], '--', c='0.01')

        fig.subplots_adjust(bottom=0.2)
        pl.savefig(self.output()['png'].path, format='png', dpi=300)
        pl.savefig(self.output()['eps'].path)

        pl.clf()
