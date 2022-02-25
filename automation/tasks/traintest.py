import json

import luigi
import tikzplotlib
import numpy as np
import pandas as pd
from luigi.format import Nop
from matplotlib import pyplot as pl

from recodiv.utils import plot_histogram
from recodiv.utils import generate_graph
from recodiv.model import split_dataset
from automation.tasks.dataset import Dataset, ImportDataset
from recodiv.triversity.graph import IndividualHerfindahlDiversities


################################################################################
# TRAIN/TEST DATASETS GENERATION AND ANALYSIS                                  #
################################################################################
class GenerateTrainTest(luigi.Task):
    """Import a dataset (with adequate format) and generate train/test data"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    def requires(self) -> luigi.Task:
        return ImportDataset(dataset=self.dataset)

    def output(self) -> list[dict[str, luigi.LocalTarget]]:
        out = []

        for i in range(self.split['n_fold']):
            out.append({
                'train': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/train.csv'),
                    format=Nop
                ),
                'test': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/test.csv'),
                    format=Nop
                )
            })

        return out

    def run(self):
        for out in self.output():
            out['train'].makedirs()
            out['test'].makedirs()

        user_item = pd.read_csv(self.input()['user_item'].path)

        if self.split['name'] == 'leave-one-out':
            for i, (train, test) in enumerate(split_dataset(user_item, self.split['row_fraction'])):
                # Find out the items and users that are in test set but not in
                # train set set (they would cause cold start issues)
                isolated_users = np.setdiff1d(
                    test['user'].unique(),
                    train['user'].unique()
                )
                isolated_items = np.setdiff1d(
                    test['item'].unique(),
                    train['item'].unique()
                )

                n_test_users = test['user'].unique().shape[0]
                n_train_users = train['user'].unique().shape[0]
                n_isolated_users = isolated_users.shape[0]
                # Thanks to the splitting procedure, we should not have isolated
                # users
                assert(n_isolated_users == 0)
                print(
                    f'#isolated users in fold {i}: {n_isolated_users} '
                    f'| #test users {n_test_users} ({100 * n_isolated_users/n_test_users:0.2f}%) '
                    f'| #train users {n_train_users} ({100 * n_isolated_users/n_train_users:0.2f}%) '
                )
                n_test_items = test['item'].unique().shape[0]
                n_train_items = train['item'].unique().shape[0]
                n_isolated_items = isolated_items.shape[0]
                print(
                    f'#isolated items in fold {i}: {n_isolated_items} '
                    f'| #test items {n_test_items} ({100 * n_isolated_items/n_test_items:0.2f}%) '
                    f'| #train items {n_train_items} ({100 * n_isolated_items/n_train_items:0.2f}%) '
                )

                # Remove these users and items and the corresponding
                # interactions from the test set
                test = test.set_index('item') \
                    .drop(isolated_items) \
                    .reset_index()

                train.to_csv(self.output()[i]['train'].path, index=False)
                test.to_csv(self.output()[i]['test'].path, index=False)

        else:
            raise NotImplementedError(
                'The requested train-test split method is not implemented')

        del user_item


class TrainTestInfo(luigi.Task):
    """Compute information about the training and testings datasets (n_users ...)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    def requires(self):
        return GenerateTrainTest(
            dataset=self.dataset,
            split=self.split,
        )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath(
                f'{self.split["name"]}/train_test_info.json')
        )

    def run(self):
        info = {}

        for i, fold in enumerate(self.input()):
            train = pd.read_csv(fold['train'].path)
            test = pd.read_csv(fold['test'].path)

            info[f'fold-{i}'] = {}
            info[f'fold-{i}']['train'] = {
                'n_users': len(train['user'].unique()),
                'n_items': len(train['item'].unique()),
                'n_user_item_links': len(train)
            }
            info[f'fold-{i}']['test'] = {
                'n_users': len(test['user'].unique()),
                'n_items': len(test['item'].unique()),
                'n_user_item_links': len(test)
            }

        with self.output().open('w') as file:
            json.dump(info, file, indent=4)


class BuildTrainTestGraphs(luigi.Task):
    """Build users-songs-tags graph for the train and test sets"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    def requires(self):
        return {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split
            ),
            'dataset': ImportDataset(
                dataset=self.dataset
            )
        }

    def output(self):
        out = []

        for i in range(self.split['n_fold']):
            out.append({
                'train': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/train-graph.pk'),
                    format=Nop
                ),
                'test': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/test-graph.pk'),
                    format=Nop
                )
            })

        return out

    def run(self):
        self.output()[0]['train'].makedirs()

        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)

        for i, fold in enumerate(self.input()['train_test']):
            train_user_item = pd.read_csv(
                fold['train'].path)
            test_user_item = pd.read_csv(
                fold['test'].path)

            train_graph = generate_graph(train_user_item, item_tag)
            test_graph = generate_graph(test_user_item, item_tag)

            train_graph.persist(self.output()[i]['train'].path)
            test_graph.persist(self.output()[i]['test'].path)

            del train_graph, test_graph, train_user_item, test_user_item

        del item_tag


class ComputeTrainTestUserDiversity(luigi.Task):
    """Compute the user diversity of the users in the train and test sets"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def requires(self):
        return BuildTrainTestGraphs(
            dataset=self.dataset,
            split=self.split
        )

    def output(self):
        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        out = []

        for i in range(self.split['n_fold']):
            out.append({
                'train': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/train_users_diversities{alpha}.csv'),
                    format=Nop
                ),
                'test': luigi.LocalTarget(
                    self.dataset.data_folder.joinpath(
                        f'{self.split["name"]}/fold-{i}/test_users_diversities{alpha}.csv'),
                    format=Nop
                )
            })

        return out

    def run(self):
        for i in range(self.split['n_fold']):
            train_graph = IndividualHerfindahlDiversities.recall(
                self.input()[i]['train'].path
            )
            test_graph = IndividualHerfindahlDiversities.recall(
                self.input()[i]['test'].path
            )

            train_graph.normalise_all()
            train_diversities = train_graph.diversities(
                (0, 1, 2), alpha=self.alpha)

            test_graph.normalise_all()
            test_diversities = test_graph.diversities(
                (0, 1, 2), alpha=self.alpha)

            pd.DataFrame({
                'user': list(train_diversities.keys()),
                'diversity': list(train_diversities.values())
            }).to_csv(self.output()[i]['train'].path, index=False)

            pd.DataFrame({
                'user': list(test_diversities.keys()),
                'diversity': list(test_diversities.values())
            }).to_csv(self.output()[i]['test'].path, index=False)

            del train_graph, test_graph, train_diversities, test_diversities


# Deprectated
class PlotTrainTestUsersDiversitiesHistogram(luigi.Task):
    """Plot the histogram of user diversity"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    def output(self):
        figures = self.dataset.data_folder.joinpath(f'figures')
        return {
            'train_png': luigi.LocalTarget(figures.joinpath(f'trainset_user_diversity{self.alpha}_histogram.png')),
            'train_tex': luigi.LocalTarget(figures.joinpath(f'trainset_user_diversity{self.alpha}_histogram.tex')),
            'test_png': luigi.LocalTarget(figures.joinpath(f'testset_user_diversity{self.alpha}_histogram.png')),
            'test_tex': luigi.LocalTarget(figures.joinpath(f'testset_user_diversity{self.alpha}_histogram.tex')),
        }

    def requires(self):
        return ComputeTrainTestUserDiversity(
            dataset=self.dataset,
            user_fraction=self.user_fraction,
            alpha=self.alpha
        )

    def run(self):
        self.output()['train_tex'].makedirs()

        train_diversities = pd.read_csv(self.input()['train'].path)
        test_diversities = pd.read_csv(self.input()['test'].path)

        fig, ax = plot_histogram(
            train_diversities['diversity'].to_numpy(), min_quantile=0, max_quantile=1)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        fig.savefig(self.output()['train_png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['train_tex'].path)

        fig, ax = plot_histogram(
            test_diversities['diversity'].to_numpy(), min_quantile=0, max_quantile=1)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        fig.savefig(self.output()['test_png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['test_tex'].path)

        del fig, ax, train_diversities, test_diversities


class ComputeTrainTestUserTagsDistribution(luigi.Task):
    """Compute the tag distibution of items listened by given user. Sorts by
    decreasing normalized weight"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    def requires(self):
        return BuildTrainTestGraphs(
            dataset=self.dataset,
            user_fraction=self.user_fraction
        )

    def output(self):
        folder = self.dataset.data_folder.joinpath('user-info')

        return {
            'train': luigi.LocalTarget(
                folder.joinpath(f'train-user{self.user}-tags-distribution.csv')
            ),
            'test': luigi.LocalTarget(
                folder.joinpath(f'test-user{self.user}-tags-distribution.csv')
            ),
        }

    def run(self):
        self.output()['train'].makedirs()

        train_graph = IndividualHerfindahlDiversities.recall(self.input()[
                                                             'train'].path)
        test_graph = IndividualHerfindahlDiversities.recall(self.input()[
                                                            'test'].path)

        # Compute the bipartite projection of the user graph on the tags layer
        test_graph.normalise_all()
        test_distribution = test_graph.spread_node(
            self.user, (0, 1, 2)
        )
        test_distribution = pd.Series(test_distribution, name='weight') \
            .sort_values(ascending=False)

        # Compute the bipartite projection of the user graph on the tags layer
        train_graph.normalise_all()
        train_distribution = train_graph.spread_node(
            self.user, (0, 1, 2)
        )
        train_distribution = pd.Series(train_distribution, name='weight') \
            .sort_values(ascending=False)

        test_distribution.to_csv(self.output()['test'].path)
        train_distribution.to_csv(self.output()['train'].path)


class PlotTrainTestUserTagsDistribution(luigi.Task):
    """Compute the tag distibution of items listened by given user. Sorts by
    decreasing normalized weight"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    n_tags = luigi.parameter.IntParameter(
        default=30, description="The number of most represented tags showed in the histogram"
    )

    def requires(self):
        return ComputeTrainTestUserTagsDistribution(
            dataset=self.dataset,
            user_fraction=self.user_fraction,
            user=self.user
        )

    def output(self):
        folder = self.dataset.data_folder.joinpath('user-info/figures')

        return {
            'train': luigi.LocalTarget(
                folder.joinpath(
                    f'train-user{self.user}-{self.n_tags}tags-distribution.png')
            ),
            'test': luigi.LocalTarget(
                folder.joinpath(
                    f'test-user{self.user}-{self.n_tags}tags-distribution.png')
            ),
        }

    def run(self):
        self.output()['train'].makedirs()

        test_distribution: pd.Series = pd.read_csv(
            self.input()['test'].path, index_col=0)
        test_distribution[:self.n_tags].plot.bar(
            rot=50
        )
        pl.savefig(self.output()['test'].path, format='png', dpi=300)

        pl.clf()

        train_distribution: pd.Series = pd.read_csv(
            self.input()['train'].path, index_col=0)
        train_distribution[:self.n_tags].plot.bar(
            rot=50
        )
        pl.savefig(self.output()['train'].path, format='png', dpi=300)
