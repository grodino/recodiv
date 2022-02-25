import json
from pathlib import Path

import luigi
from luigi.format import Nop
import tikzplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

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

        hyperparam_values = [model[self.hyperparameter]
                             for model in self.models]
        regularization = self.models[0]['regularization']

        return luigi.LocalTarget(aggregated.joinpath(
            f'{self.n_recommendations_values}recommendations'
            f'_diversity{self.alpha}'
            f'_vs_{hyperparam_values}{self.hyperparameter}'
            f'_{regularization}reg.csv'
        ))

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

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )
    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user'
    )

    def requires(self):
        req = {}

        for model in self.models:
            # TODO: remove the hard-coded n_recommendations value by the
            # n_recommendations_values values. Ex: ndcg is not the same for
            # different values of n_recommendations.
            req[model[self.hyperparameter]] = EvaluateModel(
                dataset=self.dataset,
                model=model,
                split=self.split,
                n_recommendations=10
            )

        req['diversity'] = ComputeRecommendationDiversityVsHyperparameter(
            dataset=self.dataset,
            hyperparameter=self.hyperparameter,
            alpha=self.alpha,
            models=self.models,
            split=self.split,
            fold_id=self.fold_id,
            n_recommendations_values=self.n_recommendations_values,
        )

        return req

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')

        hyperparam_values = [model[self.hyperparameter]
                             for model in self.models]
        regularization = self.models[0]['regularization']

        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}recommendations'
                f'_diversity{self.alpha}'
                f'_vs_{hyperparam_values}{self.hyperparameter}'
                f'_{regularization}reg.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}recommendations'
                f'_diversity{self.alpha}'
                f'_vs_{hyperparam_values}{self.hyperparameter}'
                f'_{regularization}reg.tex'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()
        data: pd.DataFrame = pd.read_csv(
            self.input()['diversity'].path, index_col=0)

        data = data[1:].set_index(self.hyperparameter)
        data['ndcg'] = 0

        # Assuming the n_factors paramets all differ in the models
        hyperparam_values = [model[self.hyperparameter]
                             for model in self.models]

        for hyperparam in hyperparam_values[1:]:
            metric = pd.read_json(
                self.input()[hyperparam].path,
                orient='index'
            )
            data.loc[hyperparam, 'ndcg'] = metric['ndcg'][0]

        # data = data.subtract(data.min())
        # data = data.divide(data.max())
        data = data.reset_index()

        _, ax = pl.subplots(1, 1)

        for n_reco in self.n_recommendations_values:
            data.plot(
                ax=ax,
                x=self.hyperparameter,
                y=f'{n_reco} recommendations',
                xlabel=self.hyperparameter,
                ylabel='diversity',
                logx=True,
                style='-',
                linewidth='1',
            )

        ndcg_ax = ax.twinx()
        data.plot(
            ax=ndcg_ax,
            x=self.hyperparameter,
            y='ndcg',
            xlabel=self.hyperparameter,
            ylabel='NDCG',
            logx=True,
            style='--',
            linewidth='2',
            color='black',
            legend='NDCG @ 10'
        )

        ax.legend(loc=4)

        pl.savefig(self.output()['png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['latex'].path)


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
