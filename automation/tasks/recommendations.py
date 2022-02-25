from pathlib import Path
from typing import List

import luigi
from luigi.format import Nop
import tikzplotlib
import pandas as pd
from matplotlib import colors, cm, figure
from matplotlib import pyplot as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from recodiv.utils import plot_histogram
from recodiv.utils import linear_regression
from recodiv.utils import generate_recommendations_graph
from recodiv.utils import build_recommendations_listenings_graph
from recodiv.triversity.graph import IndividualHerfindahlDiversities
from automation.tasks.model import GenerateRecommendations
from automation.tasks.dataset import ComputeUsersDiversities, Dataset, ImportDataset
from automation.tasks.traintest import BuildTrainTestGraphs, ComputeTrainTestUserDiversity, ComputeTrainTestUserTagsDistribution, GenerateTrainTest


################################################################################
# RECOMMENDATIONS ANALYSIS                                                     #
################################################################################
class BuildRecommendationGraph(luigi.Task):
    """Build the user-song-tag graph for the recommendations.

    We assume that the number of listenings of each recommendation is given by
    (n_recommendations - rank)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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

    def requires(self):
        req = {
            'dataset': ImportDataset(self.dataset),
            'recommendations': [],
        }

        for fold_id in range(self.split['n_fold']):
            req['recommendations'].append(GenerateRecommendations(
                dataset=self.dataset,
                model=self.model,
                split=self.split,
                fold_id=fold_id,
                n_recommendations=self.n_recommendations
            ))

        return req

    def output(self):
        out = []

        for fold_id in range(self.split['n_fold']):
            model = Path(self.input()['recommendations'][fold_id].path).parent

            out.append(luigi.LocalTarget(
                model.joinpath(
                    f'recommendations-{self.n_recommendations}-graph.pk'),
                format=Nop
            ))

        return out

    def run(self):
        for out in self.output():
            out.makedirs()

        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)

        for fold_id, recommendations_file in enumerate(self.input()['recommendations']):
            recommendations: pd.DataFrame = pd.read_csv(
                recommendations_file.path)

            graph = generate_recommendations_graph(recommendations, item_tag)
            graph.persist(self.output()[fold_id].path)

            del graph


class ComputeRecommendationDiversities(luigi.Task):
    """Compute the diversity of the songs recommended to users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )
    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return BuildRecommendationGraph(
            dataset=self.dataset,
            model=self.model,
            split=self.split,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        out = []

        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        for fold_id in range(self.split['n_fold']):
            model = Path(self.input()[fold_id].path).parent

            out.append(luigi.LocalTarget(
                model.joinpath(
                    f'recommendations-{self.n_recommendations}-users_diversities{alpha}.csv')
            ))

        return out

    def run(self):
        for out in self.output():
            out.makedirs()

        for fold_id, graph_file in enumerate(self.input()):
            graph = IndividualHerfindahlDiversities.recall(graph_file.path)

            graph.normalise_all()
            diversities = graph.diversities((0, 1, 2), alpha=self.alpha)

            pd.DataFrame({
                'user': list(diversities.keys()),
                'diversity': list(diversities.values())
            }).to_csv(self.output()[fold_id].path, index=False)

            del graph, diversities


class PlotRecommendationsUsersDiversitiesHistogram(luigi.Task):
    """Plot the histogram of recommendations diversity for each user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )
    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return ComputeRecommendationDiversities(
            dataset=self.dataset,
            alpha=self.alpha,
            model=self.model,
            split=self.split,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        figures = Path(
            self.input()[self.fold_id].path
        ).parent.joinpath('figures')

        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}-recommendation_user_diversity{self.alpha}_histogram.png'
        ))

    def run(self):
        self.output().makedirs()

        diversities = pd.read_csv(self.input()[self.fold_id].path)
        fig, ax = plot_histogram(
            diversities['diversity'].to_numpy(),
            min_quantile=0,
            max_quantile=1
        )

        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of recommendations diversity index')
        pl.savefig(self.output().path, format='png', dpi=300)

        del fig, ax, diversities


class BuildRecommendationsWithListeningsGraph(luigi.Task):
    """Add the recommendations to the train user-item-tag graph

    This describes the situation where a user listens only to recommendations
    after the model has been trained"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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

    def requires(self):
        req = {
            'graph': BuildTrainTestGraphs(
                dataset=self.dataset,
                split=self.split
            ),
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split,
            ),
            'recommendations': [],
        }

        for i in range(self.split['n_fold']):
            req['recommendations'].append(GenerateRecommendations(
                dataset=self.dataset,
                model=self.model,
                split=self.split,
                fold_id=i,
                n_recommendations=self.n_recommendations
            ))

        return req

    def output(self):
        out = []

        for i in range(self.split['n_fold']):
            model = Path(self.input()['recommendations'][i].path).parent

            out.append(luigi.LocalTarget(
                model.joinpath(
                    f'listenings-{self.n_recommendations}recommendations'
                    '-graph.pk'
                ),
                format=Nop
            ))

        return out

    def run(self):
        self.output()[0].makedirs()

        for i, fold_graph in enumerate(self.input()['graph']):
            graph = IndividualHerfindahlDiversities.recall(
                fold_graph['test'].path
            )

            # Used to compute the volume a user would have listened to if listening its music
            user_item = pd.read_csv(self.input()['train_test'][i]['test'].path)
            recommendations = pd.read_csv(
                self.input()['recommendations'][i].path)

            graph = build_recommendations_listenings_graph(
                graph,
                user_item,
                recommendations
            )
            graph.persist(self.output()[i].path)

            del graph, user_item, recommendations


class ComputeRecommendationWithListeningsUsersDiversities(luigi.Task):
    """Compute the diversity of the users who were recommended, assuming they
       listened to all recommendations"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def requires(self):
        return BuildRecommendationsWithListeningsGraph(
            dataset=self.dataset,
            model=self.model,
            split=self.split,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        out = []

        for i in range(self.split['n_fold']):
            model = Path(self.input()[i].path).parent

            out.append(luigi.LocalTarget(
                model.joinpath(
                    f'listenings-recommendations-{self.n_recommendations}-users_diversities{alpha}.csv'
                ),
                format=Nop
            ))

        return out

    def run(self):
        for i in range(self.split['n_fold']):
            graph = IndividualHerfindahlDiversities.recall(
                self.input()[i].path
            )

            graph.normalise_all()
            diversities = graph.diversities((0, 1, 2), alpha=self.alpha)
            diversities = pd.DataFrame({
                'user': list(diversities.keys()),
                'diversity': list(diversities.values())
            })

            diversities.to_csv(self.output()[i].path, index=False)

            del graph, diversities


class ComputeRecommendationWithListeningsUsersDiversityIncrease(luigi.Task):
    """Compare the diversity of a user if they start listenings only to
    recommendations or if they continue to listen their music"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def requires(self):
        return {
            'with_recommendations': ComputeRecommendationWithListeningsUsersDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                model=self.model,
                split=self.split,
                n_recommendations=self.n_recommendations
            ),
            'original': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                split=self.split,
                alpha=self.alpha,
            )
        }

    def output(self):
        out = []

        for i in range(self.split['n_fold']):
            model = Path(self.input()['with_recommendations'][i].path).parent

            # Avoid issues where 0.0 and 0 lead to different file titles
            alpha = float(self.alpha)
            alpha = int(alpha) if alpha.is_integer() else alpha

            out.append(luigi.LocalTarget(
                model.joinpath(
                    f'listenings-recommendations-{self.n_recommendations}-users_diversities{alpha}_increase.csv')
            ))

        return out

    def run(self):
        for i in range(self.split['n_fold']):
            with_recommendations = pd.read_csv(self.input()['with_recommendations'][i].path) \
                .set_index('user')
            original = pd.read_csv(self.input()['original'][i]['test'].path) \
                .set_index('user')

            deltas = (with_recommendations['diversity'] - original['diversity']) \
                .reset_index()

            deltas.to_csv(self.output()[i].path, index=False)

            del original, deltas


class PlotUserDiversityIncreaseVsUserDiversity(luigi.Task):
    """Plot the user diversity increase with respect to the user diversity
       before recommendations"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate per user'
    )

    alpha_values = luigi.parameter.ListParameter(
        description="A list of true diversity orders"
    )

    fold_id = luigi.parameter.IntParameter(
        description="The id of the fold to choose the test data from"
    )

    bounds = luigi.parameter.ListParameter(
        default=None,
        description=(
            "The bounding box of the graph supplied as (x_min, x_max, y_min,"
            "y_max). If a value is None, leaves matplotlib default"
        )
    )

    users = luigi.ListParameter(
        default=[], description='Specific users to pinpoint in the graph'
    )

    show_colorbar = luigi.BoolParameter(
        default=True, description='Whether to display the colobar or not'
    )

    def requires(self):
        req = {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split
            ),
            'user_diversity': [],
            'diversity_increase': [],
        }

        for alpha in self.alpha_values:
            for n_recommendations in self.n_recommendations_values:
                req['user_diversity'].append(ComputeTrainTestUserDiversity(
                    dataset=self.dataset,
                    alpha=alpha,
                    split=self.split
                ))
                req['diversity_increase'].append(ComputeRecommendationWithListeningsUsersDiversityIncrease(
                    dataset=self.dataset,
                    alpha=alpha,
                    model=self.model,
                    split=self.split,
                    n_recommendations=n_recommendations
                ))

        return req

    def output(self):
        figures = Path(self.input()['diversity_increase'][0][self.fold_id].path) \
            .parent.joinpath('figures')

        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}-recommendations_diversity{self.alpha_values}_increase_vs_original_diversity.png'
            )),
            'pdf': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}-recommendations_diversity{self.alpha_values}_increase_vs_original_diversity.eps'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()

        n_recs = len(self.n_recommendations_values)
        n_alphas = len(self.alpha_values)

        fig: figure.Figure
        fig, axes = pl.subplots(
            n_alphas, n_recs,
            constrained_layout=True,
            figsize=(.8*6.4, .8*(4.8 + 1.5)),
            dpi=600
        )
        flat_axes: List[pl.Axes] = axes.flatten()

        # Compute the min and max volume
        min_volume, max_volume = float('inf'), 0

        for i, ax in enumerate(flat_axes):
            diversities = pd.read_csv(
                self.input()['user_diversity'][i][self.fold_id]['test'].path)
            increase = pd.read_csv(self.input()['diversity_increase'][i][self.fold_id].path).rename(
                columns={'diversity': 'increase'})

            # compute user volume
            user_item = pd.read_csv(
                self.input()['train_test'][self.fold_id]['train'].path)
            volume = user_item.groupby('user')['rating'].sum() \
                .rename('volume')

            # Compute the min, max volume over all the plots
            min_volume = min(min_volume, volume.min())
            max_volume = max(max_volume, volume.max())

            # inner join, only keep users for whom we calculated a diversity increase value
            merged = increase.merge(diversities, on='user')
            merged = merged.merge(volume, on='user')

            if self.bounds == None:
                self.bounds = [None, None, None, None]

            # Plot the user points
            ax.scatter(
                x=merged['diversity'],
                y=merged['increase'],
                marker='o',
                c=merged['volume'],
                cmap='viridis',
                s=10,
                norm=colors.LogNorm(vmin=min_volume, vmax=max_volume),
                rasterized=True
            )
            ax.set_box_aspect(1/1.3333333)
            ax.set_xlim(self.bounds[:2])
            ax.set_ylim(self.bounds[2:])

            # Plot the linear regression
            a, b = linear_regression(merged, 'diversity', 'increase')
            x = merged[(self.bounds[0] < merged['diversity']) & (merged['diversity'] < self.bounds[1])]['diversity'] \
                .sort_values().to_numpy()
            y = a * x + b
            ax.plot(x, y, '--', c='purple')

            markers = ['^', '*', 'o', 's']
            for i, user in enumerate(self.users):
                merged[merged['user'] == user].plot(
                    ax=ax,
                    x='diversity',
                    y='increase',
                    marker=markers[i],
                    markeredgewidth=1,
                    # markeredgecolor='yellow',
                    c='red',
                )

            # Normalize the display of alpha
            alpha = float(self.alpha_values[i//n_alphas])
            if alpha.is_integer():
                alpha = f'{int(alpha)}'
            elif alpha == float('inf'):
                alpha = f'+\infty'

            ax.set_title(
                f'({chr(97 + i)}) '
                f'$k = {self.n_recommendations_values[i%n_alphas]}$, '
                f'$\\alpha = {alpha}$'
            )

        # Create the colorbar
        norm = colors.LogNorm(vmin=min_volume, vmax=max_volume)
        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap='viridis'),
            ax=axes[0, :],
            location='top',
            label='user listening volume',
            shrink=.7,
            pad=.1
        )

        fig.set_constrained_layout_pads(w_pad=0.01, wspace=0.03)
        fig.supxlabel('organic diversity')
        fig.supylabel('diversity increase')

        fig.savefig(self.output()['png'].path, format='png', dpi=300)
        fig.savefig(self.output()['pdf'].path)

        pl.clf()

        del diversities, increase, merged


# WIP
class PlotRecommendationDiversityVsUserDiversity(luigi.Task):
    """Plot the diversity of the recommendations associated to each user with
    respect to the user diversity before recommendations (ie train set
    diversity)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    bounds = luigi.parameter.ListParameter(
        default=None,
        description=(
            "The bounding box of the graph supplied as (x_min, x_max, y_min,"
            "y_max). If a value is None, leaves matplotlib default"
        )
    )

    def requires(self):
        return {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split,
            ),
            'diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                split=self.split,
                alpha=self.alpha,
            ),
            'recommendation_diversity': ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        figures = Path(
            self.input()['recommendation_diversity'].path).parent.joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_vs_original_diversity.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_vs_original_diversity.tex'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()
        diversities = pd.read_csv(self.input()['diversity']['train'].path)
        reco_diversities = pd.read_csv(
            self.input()['recommendation_diversity'].path
        ).rename(columns={'diversity': 'reco_diversity'})

        # compute user volume
        user_item = pd.read_csv(self.input()['train_test']['train'].path)
        volume = user_item.groupby('user')['rating'].sum() \
            .rename('volume')

        # inner join, only keep users for whom we calculated a recommendation diversity value
        merged: pd.DataFrame = reco_diversities.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')

        up_bound = min(merged['diversity'].max(),
                       merged['reco_diversity'].max())
        low_bound = max(merged['diversity'].min(),
                        merged['reco_diversity'].min())

        _, ax = pl.subplots()
        ax.plot([low_bound, up_bound], [low_bound, up_bound], '--', c='purple')

        if self.bounds == None:
            self.bounds = [None, None, None, None]

        merged.plot.scatter(
            ax=ax,
            x='diversity',
            y='reco_diversity',
            marker='+',
            c='volume',
            colormap='viridis',
            norm=colors.LogNorm(vmin=volume.min(), vmax=volume.max()),
            xlim=self.bounds[:2],
            ylim=self.bounds[2:],
        )
        pl.xlabel('organic diversity')
        pl.ylabel('recommendation diversity')

        pl.savefig(self.output()['png'].path, format='png', dpi=300)

        with open(self.output()['latex'].path, 'w') as file:
            code: str = tikzplotlib.get_tikz_code(
                extra_axis_parameters=[
                    'clip mode=individual', 'clip marker paths=true']
            )

            code = code.replace('ytick={1,10,100,1000}', 'ytick={0,1,2,3}')
            code = code.replace(
                'meta=colordata', 'meta expr=lg10(\\thisrow{colordata})')
            code = code.replace('point meta', '% point meta')
            code = code.replace('semithick', 'very thick')

            file.write(code)

        pl.clf()

        del diversities, reco_diversities, merged


# Rest is now deprecated
class PlotDiversitiesIncreaseHistogram(luigi.Task):
    """Plot the histogram of recommendations diversity for each user"""

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

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return ComputeRecommendationWithListeningsUsersDiversityIncrease(
            dataset=self.dataset,
            alpha=self.alpha,
            n_iterations=self.n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        figures = Path(self.input().path).parent.joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_increase_histogram.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_increase_histogram.tex'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()
        deltas: pd.DataFrame = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(deltas['diversity'].to_numpy(
        ), min_quantile=0, max_quantile=1, log=True)

        ax.set_xlabel('diversity increase')
        ax.set_ylabel('user count')

        pl.savefig(self.output()['png'].path, format='png', dpi=300)
        # pl.savefig(self.output()['pdf'].path, backend='pgf')

        tikzplotlib.save(self.output()['latex'].path)

        del ax, deltas


class ComputeUserRecommendationsTagsDistribution(luigi.Task):
    """Compute the distributions of tags of the items recommended to a given user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return BuildRecommendationGraph(
            dataset=self.dataset,
            n_iterations=self.n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input().path).parent
        folder = model.joinpath('user-info')

        return luigi.LocalTarget(
            folder.joinpath(
                f'{self.n_recommendations}reco-user{self.user}-tags-distribution.csv')
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(self.input().path)

        # Compute the bipartite projection of the user graph on the tags layer
        graph.normalise_all()
        distribution = graph.spread_node(
            self.user, (0, 1, 2)
        )
        distribution = pd.Series(distribution, name='weight') \
            .sort_values(ascending=False)

        distribution.to_csv(self.output().path)


class ComputeUserListenedRecommendedTagsDistribution(luigi.Task):
    """Compute the tag distribution reached by a user thtough their listened and recommended items"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return BuildRecommendationsWithListeningsGraph(
            dataset=self.dataset,
            n_iterations=self.n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input().path).parent
        folder = model.joinpath('user-info')

        return luigi.LocalTarget(
            folder.joinpath(
                f'listening-{self.n_recommendations}reco-user{self.user}-tags-distribution.csv')
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(self.input().path)

        # Compute the bipartite projection of the user graph on the tags layer
        graph.normalise_all()
        distribution = graph.spread_node(
            self.user, (0, 1, 2)
        )
        distribution = pd.Series(distribution, name='weight') \
            .sort_values(ascending=False)

        distribution.to_csv(self.output().path)


class PlotUserRecommendationsTagsDistribution(luigi.Task):
    """Compute the tag distibution of items listened by given user. Sorts by
    decreasing normalized weight"""

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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    n_tags = luigi.parameter.IntParameter(
        default=30, description="The number of most represented tags showed in the histogram"
    )

    def requires(self):
        return ComputeUserRecommendationsTagsDistribution(
            dataset=self.dataset,
            user=self.user,
            n_iterations=self.n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        folder = Path(self.input().path).parent.joinpath('figures')

        return luigi.LocalTarget(
            folder.joinpath(
                f'{self.n_recommendations}reco-user{self.user}-tags-distribution.png')
        )

    def run(self):
        self.output().makedirs()

        distribution: pd.Series = pd.read_csv(self.input().path, index_col=0)
        distribution[:self.n_tags].plot.bar(
            rot=50
        )
        pl.savefig(self.output().path, format='png', dpi=300)


class PlotUserListeningRecommendationsTagsDistributions(luigi.Task):
    """Plot the most listened and recommended tags for a user"""

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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    n_tags = luigi.parameter.IntParameter(
        default=30, description="The number of most represented tags showed in the histogram"
    )

    def requires(self):
        return {
            'recommended_tags': ComputeUserRecommendationsTagsDistribution(
                dataset=self.dataset,
                user=self.user,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'listened_tags': ComputeTrainTestUserTagsDistribution(
                dataset=self.dataset,
                user=self.user,
                user_fraction=self.model_user_fraction,
            ),
        }

    def output(self):
        folder = Path(
            self.input()['recommended_tags'].path).parent.joinpath('figures')

        return luigi.LocalTarget(
            folder.joinpath(
                f'{self.n_recommendations}reco-listened-user{self.user}-tags-distribution.png')
        )

    def run(self):
        self.output().makedirs()

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

        heaviest_tags = pd.merge(
            listened_distribution[:self.n_tags],
            reco_distribution[:self.n_tags],
            on='tag',
            how='outer'
        )

        ax = heaviest_tags.plot.bar(x='tag', logy=True)
        pl.setp(ax.get_xticklabels(), rotation=-40,
                rotation_mode="anchor", ha="left")
        pl.title(
            f'{self.n_recommendations} reco, {self.model_n_factors} factors, {self.model_regularization} regularization')

        pl.savefig(self.output().path, format='png', dpi=300)


class ComputeHeaviestTagRank(luigi.Task):
    """For each user, compute the most listened tag, the find the rank of this tag in the recommendations"""
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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return {
            'recommendation_graph': BuildRecommendationGraph(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'train_test_graph': BuildTrainTestGraphs(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction,
            )
        }

    def output(self):
        model = Path(self.input()['recommendation_graph'].path).parent

        return luigi.LocalTarget(
            model.joinpath(
                f'{self.n_recommendations}reco-user-heaviest-tag-reco-rank.csv')
        )

    def run(self):
        self.output().makedirs()

        listened_graph = IndividualHerfindahlDiversities.recall(
            self.input()['train_test_graph']['train'].path)
        recommended_graph = IndividualHerfindahlDiversities.recall(
            self.input()['recommendation_graph'].path)

        # Compute the bipartite projection of the user graph on the tags layer
        listened_graph.normalise_all()
        recommended_graph.normalise_all()

        listened_distributions = listened_graph.spread((0, 1, 2))
        recommended_distributions = recommended_graph.spread((0, 1, 2))

        # Compute the most recommended tag for each user
        heaviest_reco_tag = {}

        for user, distrib in recommended_distributions.items():
            heaviest_reco_tag[user] = max(
                distrib.items(), key=lambda item: item[1])[0]

        # Copute the rank of this tag in the user's listenings
        heaviest_tag_rank = {}

        for user in recommended_distributions.keys():
            tag_distribution = sorted(
                listened_distributions[user].items(),
                key=lambda item: item[1],
                reverse=True
            )

            # look for the heaviest recommended tag
            for i, (tag, _) in enumerate(tag_distribution):
                if tag == heaviest_reco_tag[user]:
                    heaviest_tag_rank[user] = i
                    break

        pd.DataFrame({
            'user': heaviest_tag_rank.keys(),
            'rank': heaviest_tag_rank.values(),
            'tag': [heaviest_reco_tag[user] for user in heaviest_tag_rank.keys()]
        }).to_csv(self.output().path, index=False)


class PlotHeaviestTagRankVsPercentageIncreased(luigi.Task):
    """Plot and compute the mean heaviest tag rank in two sets : the set of users who's
    diversity is increased by the recommendations and the set of users who's
    diversity is decreased by the recommendations."""

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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def requires(self):
        return {
            'tag_rank': ComputeHeaviestTagRank(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations,
                alpha=self.alpha
            ),
        }

    def output(self):
        model = Path(self.input()['tag_rank'].path).parent
        figures = model.joinpath('figures')

        return luigi.LocalTarget(
            figures.joinpath(
                f'{self.n_recommendations}reco-heaviest-tag-reco-rank-vs-increase{self.alpha}.png')
        )

    def run(self):
        tag_rank = pd.read_csv(self.input()['tag_rank'].path)
        increase: pd.DataFrame = pd.read_csv(self.input()['increase'].path)

        merged = pd.merge(tag_rank, increase, on='user')
        increased = merged[merged['diversity'] > 0]
        decreased = merged[merged['diversity'] < 0]

        increased_mean_rank = increased['rank'].mean()
        decreased_mean_rank = decreased['rank'].mean()

        ax = merged.plot.scatter(
            x='diversity',
            y='rank',
            marker='+',
            logy=True,
            ylim=(9e-1, 1e3),
        )

        ax.text(
            0, 1,
            f'increased mean rank : {increased_mean_rank}\ndecreased mean rank : {decreased_mean_rank}',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
        )

        pl.xlabel('Diversity increase')
        pl.ylabel('Best recommended tag rank')

        pl.savefig(self.output().path, format='png', dpi=300)
        pl.clf()


class PlotUserTagHistograms(luigi.Task):
    """Plot the listened, recommended and listened+recommended tags histograms
    for a user"""

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
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    n_tags = luigi.parameter.IntParameter(
        default=30, description="The number of most represented tags showed in the histogram"
    )

    def requires(self):
        return {
            'recommended_tags': ComputeUserRecommendationsTagsDistribution(
                dataset=self.dataset,
                user=self.user,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'listened_tags': ComputeTrainTestUserTagsDistribution(
                dataset=self.dataset,
                user=self.user,
                user_fraction=self.model_user_fraction,
            ),
            'after_reco_tags': ComputeUserListenedRecommendedTagsDistribution(
                dataset=self.dataset,
                user=self.user,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        folder = Path(
            self.input()['recommended_tags'].path).parent.joinpath('figures')

        return {
            'png': luigi.LocalTarget(
                folder.joinpath(
                    f'increase{self.alpha}-{self.n_recommendations}reco-user{self.user}-tag-histograms.png')
            ),
            'latex': luigi.LocalTarget(
                folder.joinpath(
                    f'increase{self.alpha}-{self.n_recommendations}reco-user{self.user}-tag-histograms.tex')
            ),
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

        increase: pd.DataFrame = pd.read_csv(self.input()['increase'].path)

        if float(increase[increase['user'] == self.user]['diversity']) > 0:
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

        inset = 'at={(insetPosition)},anchor={outer north east},\nwidth=2in,height=1.25in,'
        inset_coord = '\coordinate (insetPosition) at (rel axis cs:0.95,0.95);'

        with open(self.output()['latex'].path, 'w') as file:
            code = tikzplotlib.get_tikz_code(
                extra_axis_parameters=[
                    'scaled ticks=false', 'tick label style={/pgf/number format/fixed}']
            )
            code = code.replace('xmajorticks=false,',
                                'xmajorticks=false,\n' + inset)
            code = code.replace(
                '\\addlegendentry{listened}', '\\addlegendentry{listened}\n' + inset_coord)

            file.write(code)


class MetricsSummary(luigi.Task):
    """Create a single file that summarizes metrics computed for each test user, for
    manual analysis"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha_values = luigi.parameter.ListParameter(
        description="The true diversity orders used"
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
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        req = {'train_test': GenerateTrainTest(
            dataset=self.dataset,
            user_fraction=self.model_user_fraction
        )}

        for alpha in self.alpha_values:
            req[f'train_test_div{alpha}'] = ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction,
                alpha=alpha
            )
            req[f'dataset_div{alpha}'] = ComputeUsersDiversities(
                dataset=self.dataset,
                alpha=alpha
            )

            for n_recommendations in self.n_recommendations_values:
                req[f'reco{n_recommendations}_div{alpha}'] = ComputeRecommendationDiversities(
                    dataset=self.dataset,
                    alpha=alpha,
                    n_iterations=self.n_iterations,
                    model_n_factors=self.model_n_factors,
                    model_regularization=self.model_regularization,
                    model_user_fraction=self.model_user_fraction,
                    n_recommendations=n_recommendations
                )
                req[f'reco{n_recommendations}_div{alpha}_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                    dataset=self.dataset,
                    alpha=alpha,
                    n_iterations=self.n_iterations,
                    model_n_factors=self.model_n_factors,
                    model_regularization=self.model_regularization,
                    model_user_fraction=self.model_user_fraction,
                    n_recommendations=n_recommendations
                )

        return req

    def output(self):
        folder = self.dataset.base_folder.joinpath('aggregated')

        return luigi.LocalTarget(
            folder.joinpath(
                f'summary-{self.n_recommendations_values}reco-{self.alpha_values}div-{self.model_regularization}reg-{self.model_n_factors}factors.csv')
        )

    def run(self):
        self.output().makedirs()

        # Compute the volume for each user in test
        user_item: pd.DataFrame = pd.read_csv(
            self.input()['train_test']['test'].path)
        volume = user_item[['user', 'rating']].groupby('user') \
            .sum() \
            .rename(columns={'rating': 'volume'})['volume']

        # Compute the organic (train) diversity of each test user
        organic_diversities = []

        for alpha in self.alpha_values:
            organic_diversity: pd.DataFrame = pd.read_csv(
                self.input()[f'train_test_div{alpha}']['train'].path)
            organic_diversities.append(
                organic_diversity.set_index('user')['diversity'].rename(
                    f'train_diversity{alpha}')
            )

        # Compute the recommendation diversity for each user
        reco_diversities = []

        for alpha in self.alpha_values:
            for n_recommendations in self.n_recommendations_values:
                reco_diversity: pd.DataFrame = pd.read_csv(
                    self.input()[f'reco{n_recommendations}_div{alpha}'].path)
                reco_diversities.append(
                    reco_diversity.set_index('user')['diversity'].rename(
                        f'reco{n_recommendations}_div{alpha}')
                )

        # Compute the diversity increase for each user
        increases = []

        for alpha in self.alpha_values:
            for n_recommendations in self.n_recommendations_values:
                increase: pd.DataFrame = pd.read_csv(
                    self.input()[f'reco{n_recommendations}_div{alpha}_increase'].path)
                increases.append(
                    increase.set_index('user')['diversity'].rename(
                        f'reco{n_recommendations}_div{alpha}_increase')
                )

        columns = [volume] + organic_diversities + reco_diversities + increases
        summary = pd.DataFrame(index=volume.index)

        for col in columns:
            summary[col.name] = col

        summary.reset_index().to_csv(self.output().path, index=False)
