from pathlib import Path
from typing import List

import luigi
from matplotlib.axes import Axes
import tikzplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import ticker

from recodiv.utils import generate_graph
from recodiv.utils import generate_recommendations_graph
from recodiv.model import rank_to_weight
from recodiv.triversity.graph import IndividualHerfindahlDiversities
from automation.tasks.dataset import Dataset, ImportDataset
from automation.tasks.model import EvaluateModel, GenerateRecommendations
from automation.tasks.traintest import BuildTrainTestGraphs
from automation.tasks.recommendations import ComputeRecommendationWithListeningsUsersDiversityIncrease


################################################################################
# HYPERPARAMETERS ANALYSIS                                                     #
################################################################################
class ComputeRecommendationDiversityVsHyperparameter(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
    number of latent factors

    The models in the given list must be implicit-MF models and differ only by
    the studied hyper-parameter.

    TODO: compute this for all the folds, create an Avg version
    """

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    hyperparameter = luigi.parameter.Parameter(
        description='Name of the hyper-parameter being studied'
    )

    models = luigi.parameter.ListParameter(
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
    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        tasks['dataset'] = ImportDataset(dataset=self.dataset)

        for model in self.models:
            tasks[model[self.hyperparameter]] = GenerateRecommendations(
                dataset=self.dataset,
                model=model,
                split=self.split,
                fold_id=self.fold_id,
                n_recommendations=max(self.n_recommendations_values)
            )

        return tasks

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')

        n_factors_values = [model['n_factors']
                            for model in self.models]
        regularization_values = [model['regularization']
                                 for model in self.models]

        # Convert array to tuple to avoid "[" and "]" in paths which could be
        # interpreted as wildcards
        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.n_recommendations_values}recommendations'
                f'_diversity{self.alpha}'
                f'_vs_{tuple(n_factors_values)}n_factors'
                f'_{tuple(regularization_values)}reg.csv'
            ),
            format=luigi.format.Nop
        )

    def run(self):
        self.output().makedirs()
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)

        hyperparam_values = [model[self.hyperparameter]
                             for model in self.models]

        data = pd.DataFrame(
            index=hyperparam_values,
            columns=[
                f'{n_recommendations} recommendations' for n_recommendations in self.n_recommendations_values]
        )

        for hyperparam in hyperparam_values:
            recommendations: pd.DataFrame = pd.read_csv(
                self.input()[hyperparam].path)

            for n_recommendations in self.n_recommendations_values:
                graph = generate_recommendations_graph(
                    recommendations[recommendations['rank']
                                    <= n_recommendations],
                    item_tag
                )

                graph.normalise_all()
                diversities = graph.diversities((0, 1, 2), alpha=self.alpha)
                data.loc[hyperparam, f'{n_recommendations} recommendations'] = \
                    sum(diversities.values()) / len(diversities)

        data = data.reset_index().rename(
            columns={'index': self.hyperparameter})
        data.to_csv(self.output().path)


class PlotRecommendationDiversityVsHyperparameter(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    hyperparameter = luigi.parameter.Parameter(
        description='Name of the hyper-parameter being studied'
    )

    models = luigi.parameter.ListParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    alpha_values = luigi.parameter.ListParameter(
        description="List of diversity order"
    )
    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user'
    )
    n_recommendations_ndcg = luigi.parameter.IntParameter(
        default=10, description='The number of recommendations to use for the NDCG evaluation'
    )

    def requires(self):
        req = {
            'diversity': []
        }

        # Get the model evaluation for all the hyperparameter values
        for model in self.models:
            req[model[self.hyperparameter]] = EvaluateModel(
                dataset=self.dataset,
                model=model,
                split=self.split,
                n_recommendations=self.n_recommendations_ndcg
            )

        # Get the diversity values for the different values of alpha
        for alpha in self.alpha_values:
            req['diversity'].append(ComputeRecommendationDiversityVsHyperparameter(
                dataset=self.dataset,
                hyperparameter=self.hyperparameter,
                alpha=alpha,
                models=self.models,
                split=self.split,
                fold_id=self.fold_id,
                n_recommendations_values=self.n_recommendations_values,
            ))

        return req

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')

        n_factors_values = [model['n_factors']
                            for model in self.models]
        regularization_values = [model['regularization']
                                 for model in self.models]

        return {
            'png': luigi.LocalTarget(
                figures.joinpath(
                    f'{"".join(map(str, self.n_recommendations_values))}recommendations'
                    f'_ndcg{self.n_recommendations_ndcg}'
                    f'_diversity{"".join(map(str, self.alpha_values))}'
                    f'_vs_{"".join(map(str, n_factors_values))}n_factors'
                    f'_{"".join(map(str, regularization_values))}reg.png'
                ),
                format=luigi.format.Nop
            ),
            'eps': luigi.LocalTarget(
                figures.joinpath(
                    f'{"".join(map(str, self.n_recommendations_values))}recommendations'
                    f'_ndcg{self.n_recommendations_ndcg}'
                    f'_diversity{"".join(map(str, self.alpha_values))}'
                    f'_vs_{"".join(map(str, n_factors_values))}n_factors'
                    f'_{"".join(map(str, regularization_values))}reg.eps'
                ),
                format=luigi.format.Nop
            ),
        }

    def run(self):
        self.output()['png'].makedirs()

        # Assuming the n_factors parameters all differ in the models
        hyperparam_values = [model[self.hyperparameter]
                             for model in self.models]
        letters = ['a)', 'b)', 'c)', 'd)']
        n_alphas = len(self.alpha_values)
        n_recos = len(self.n_recommendations_values)

        fig, axes = pl.subplots(
            1, n_alphas,
            # constrained_layout=True,
            figsize=((1.5 * 6.4), 4.8),
            dpi=600
        )
        axes_flat: List[Axes] = axes.flatten()

        for i_alpha, (alpha, ax) in enumerate(zip(self.alpha_values, axes_flat)):
            data: pd.DataFrame = pd.read_csv(
                self.input()['diversity'][i_alpha].path,
                index_col=0
            )

            data = data.set_index(self.hyperparameter)
            data['ndcg'] = 0

            for hyperparam in hyperparam_values:
                metric = pd.read_json(
                    self.input()[hyperparam].path,
                    orient='index'
                )
                print(metric)
                data.loc[hyperparam, 'ndcg'] = metric['ndcg'].mean()

            data = data.reset_index()

            # Fix the display value of alpha and set the title
            alpha = float(alpha)
            if alpha.is_integer():
                alpha = f'{int(alpha)}'
            elif alpha == float('inf'):
                alpha = '+\infty'

            ax.set_box_aspect(1/1.3333333)
            ax.set_title(f'{letters[i_alpha]} $\\alpha = {alpha}$')
            # pl.ticklabel_format(axis='x', style='plain')

            for n_reco in self.n_recommendations_values:
                # data[f'{n_reco} recommendations'] = data[f'{n_reco} recommendations'].subtract(
                #     data[f'{n_reco} recommendations'].min()
                # )
                data[f'{n_reco} recommendations'] = data[f'{n_reco} recommendations'].divide(
                    data[f'{n_reco} recommendations'].max()
                )

                ax.plot(
                    np.log10(data[self.hyperparameter]),
                    data[f'{n_reco} recommendations'],
                    '-+',
                    linewidth='1',
                    label=f'{n_reco} recommendations'
                )

            ax.set_xticks(np.log10(data[self.hyperparameter]))
            ax.set_xticklabels([
                str(round(10**tick, 4)) for tick in np.log10(data[self.hyperparameter])
            ])

            ndcg_ax = ax.twinx()
            ndcg_ax.plot(
                np.log10(data[self.hyperparameter]),
                data['ndcg'],
                '--',
                linewidth=2,
                color='black',
                label=f'NDCG @ {self.n_recommendations_ndcg}'
            )

            ndcg_ax.set_yticklabels([])

        # Re-add the tick labels for the last figure
        ndcg_ax.set_yticklabels([
            round(tick, 4) for tick in ndcg_ax.get_yticks()
        ])

        # Add the NDCG y label to the last figure
        ndcg_ax.set_ylabel('NDCG')

        # Add the diversity y label to the fist figure
        axes[0].set_ylabel('diversity (scaled)')

        # Add the x label
        fig.supxlabel(self.hyperparameter)

        fig.legend(
            handles=ax.get_lines() + ndcg_ax.get_lines(),
            ncol=n_recos + 1,
            loc='upper center'
        )

        fig.savefig(self.output()['png'].path, format='png', dpi=600)
        fig.savefig(self.output()['eps'].path)


# Deprecated
class PlotDiversityIncreaseVsLatentFactors(luigi.Task):
    """Plot the mean user diversity increase after recommendation as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    n_factors_values = luigi.parameter.ListParameter(
        description='List of number of user/item latent facors'
    )
    model_regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        for n_factors in self.n_factors_values:
            tasks[(n_factors, 'deltas')] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )
            tasks[(n_factors, 'metrics')] = EvaluateModel(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}recommendations_diversity{self.alpha}_increase_vs_{self.n_factors_values}factors_{self.model_regularization}reg.png'
        ))

    def run(self):
        self.output().makedirs()

        mean_deltas = []
        metrics = pd.DataFrame()
        factors = []

        for key, value in self.input().items():
            if key[1] == 'deltas':
                deltas = pd.read_csv(value.path)

                factors.append(key[0])
                mean_deltas.append(deltas['diversity'].mean())

            elif key[1] == 'metrics':
                metric = pd.read_json(value.path, orient='index').transpose()
                metric['n_factors'] = key[0]
                metrics = pd.concat((metrics, metric))

        metrics.set_index('n_factors', inplace=True)
        # metrics = metrics / metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()

        # Add plots
        div_line = ax1.semilogx(factors, mean_deltas,
                                color='green', label='diversity')
        ax1.set_xlabel('number of factors')
        ax1.set_ylabel('mean diversity increase')

        ax2 = ax1.twinx()
        ax2.set_ylabel('metrics')
        metrics_lines = metrics['ndcg'].plot(
            ax=ax2, legend=False, logy=True).get_lines()

        # Obscure trick to have only one legend
        lines = [*div_line, ]

        for line in metrics_lines:
            lines.append(line)

        labels = ['diversity', ] + list(metrics.columns)
        ax1.legend(lines, labels, loc='center right')

        pl.title('User diversity increase after recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.clf()


class ComputeDiversityVsRecommendationVolume(luigi.Task):
    """Compute the user diversity of recommendations against the number of
       recommendations made"""

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
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        return {
            'dataset': ImportDataset(dataset=self.dataset),
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=max(self.n_recommendations_values)
            ),
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent
        return luigi.LocalTarget(model.joinpath(
            f'recommendations_diversity{self.alpha}_vs_{self.n_recommendations_values}reco.csv'
        ))

    def run(self):
        self.output().makedirs()
        recommendations = pd.read_csv(self.input()['recommendations'].path) \
            .rename(columns={'score': 'rating'})

        # To compute the user volume in the rank to weight conversion
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        mean_diversities = []

        for n_recommendations in self.n_recommendations_values:
            graph = generate_recommendations_graph(
                recommendations[recommendations['rank'] <= n_recommendations],
                item_tag
            )

            graph.normalise_all()
            diversities = graph.diversities((0, 1, 2), alpha=self.alpha)

            mean_diversities.append(
                sum(diversities.values()) / len(diversities)
            )

        pd.DataFrame({
            'n_recommendations': self.n_recommendations_values,
            'diversity': mean_diversities
        }).to_csv(self.output().path, index=False)

        del recommendations


class PlotDiversityVsRecommendationVolume(luigi.Task):
    """Plot the user diversity of recommendations against the number of
       recommendations made"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    n_factors_values = luigi.parameter.ListParameter(
        description='List of number of user/item latent factors'
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
        tasks = {}

        for n_factors in self.n_factors_values:
            tasks[n_factors] = ComputeDiversityVsRecommendationVolume(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations_values=self.n_recommendations_values
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'recommendations_diversity{self.alpha}_vs_{self.n_recommendations_values}reco-{self.n_factors_values}factors.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'recommendations_diversity{self.alpha}_vs_{self.n_recommendations_values}reco-{self.n_factors_values}factors.tex'
            ))
        }

    def run(self):
        self.output()['png'].makedirs()

        for n_factors, file in self.input().items():
            data = pd.read_csv(file.path)

            pl.semilogx(data['n_recommendations'],
                        data['diversity'], label=f'{n_factors} factors')
            # pl.plot(data['n_recommendations'], data['diversity'], label=f'{n_factors} factors')

        pl.xlabel('Number of recommendations per user')
        pl.ylabel('diversity')
        pl.legend()

        pl.savefig(self.output()['png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['latex'].path)
        pl.clf()

        del data


# Deprecated
class ComputeDiversityIncreaseVsRecommendationVolume(luigi.Task):
    """Compute the user diversity of recommendations against the number of
       recommendations made DEPRECATED"""

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
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        return {
            'dataset': ImportDataset(dataset=self.dataset),
            'graph': BuildTrainTestGraphs(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                n_iterations=self.n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=max(self.n_recommendations_values)
            ),
        }

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(aggregated.joinpath(
            f'diversity{self.alpha}-increase_vs_{self.n_recommendations_values}reco-{self.model_n_factors}factors.csv'
        ))

    def run(self):
        self.output().makedirs()

        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path) \
            .rename(columns={'score': 'rating'})

        graph = IndividualHerfindahlDiversities.recall(
            self.input()['graph']['train'].path
        )

        # To compute the user volume in the rank to weight conversion
        user_item = pd.read_csv(self.input()['train_test']['test'].path)
        mean_diversities = []

        for n_recommendations in self.n_recommendations_values:
            recommendations = recommendations[recommendations['rank']
                                              <= n_recommendations]

            # Normalise the recommendations by the volume the user had prior to the
            # recommendations
            user_reco = rank_to_weight(user_item, recommendations)[['user', 'item', 'weight']] \
                .rename(columns={'weight': 'rating'})

            graph_with_reco = generate_graph(user_reco, graph=graph)
            graph_with_reco.normalise_all()
            diversities = graph.diversities((0, 1, 2), alpha=self.alpha)

            mean_diversities.append(
                sum(diversities.values()) / len(diversities)
            )

            del diversities

        pd.DataFrame({
            'n_recommendations': self.n_recommendations_values,
            'increase': mean_diversities
        }).to_csv(self.output().path, index=False)

        del item_tag, recommendations


# Deprecated
class PlotDiversityIncreaseVsRecommendationVolume(luigi.Task):
    """Plot the user diversity of recommendations against the number of
       recommendations made"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    n_factors_values = luigi.parameter.ListParameter(
        description='List of number of user/item latent factors'
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
        tasks = {}

        for n_factors in self.n_factors_values:
            tasks[n_factors] = ComputeDiversityIncreaseVsRecommendationVolume(
                dataset=self.dataset,
                alpha=self.alpha,
                n_iterations=self.n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations_values=self.n_recommendations_values
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'recommendations_diversity{self.alpha}-increase_vs_{self.n_recommendations_values}reco-{self.n_factors_values}factors.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'recommendations_diversity{self.alpha}-increase_vs_{self.n_recommendations_values}reco-{self.n_factors_values}factors.tex'
            ))
        }

    def run(self):
        self.output()['png'].makedirs()

        for n_factors, file in self.input().items():
            data = pd.read_csv(file.path)

            pl.semilogx(data['n_recommendations'],
                        data['increase'], label=f'{n_factors} factors')
            # pl.plot(data['n_recommendations'], data['diversity'], label=f'{n_factors} factors')

        pl.xlabel('Number of recommendations per user')
        pl.ylabel('Diversity increase')
        pl.legend()

        pl.savefig(self.output()['png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['latex'].path)
        pl.clf()

        del data
