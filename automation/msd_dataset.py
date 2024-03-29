import json
import time
import shutil
import logging
from pathlib import Path

import luigi
from luigi.format import Nop
import binpickle
import tikzplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import colors
from matplotlib import pyplot as pl
from matplotlib import patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from recodiv.utils import dataset_info
from recodiv.utils import plot_histogram
from recodiv.utils import generate_graph
from recodiv.utils import get_msd_song_info
from recodiv.utils import linear_regression
from recodiv.utils import generate_recommendations_graph
from recodiv.utils import build_recommendations_listenings_graph
from recodiv.model import train_model
from recodiv.model import split_dataset
from recodiv.model import rank_to_weight
from recodiv.model import evaluate_model_loss
from recodiv.model import generate_predictions
from recodiv.model import generate_recommendations
from recodiv.model import evaluate_model_recommendations
from recodiv.triversity.graph import IndividualHerfindahlDiversities


# Path to generated folder
GENERATED = Path('generated/')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

################################################################################
# DATASETS DECLARATION                                                         #
################################################################################


class Dataset():
    """Representation of a dataset

    All dataset classes must have the following properties: 
        - user_item : The users -> items links as a pd.DataFrame(columns=['user', 'item', 'rating']).to_csv()
        - item_tag : The items -> tags links as a pd.DataFrame(columns=['item', 'tag', 'weight']).to_csv()
        - IMPORT_FOLDER : where to find the dataset folder
        - NAME : the dataset name (must be unique among all the datasets)
        - base_folder : the root folder of all the following experiments
        - data_folder : the folder where the data will be imported
    """

    IMPORT_FOLDER: str
    NAME: str
    base_folder: Path
    data_folder: Path
    user_item: pd.DataFrame
    item_tag: pd.DataFrame


class MsdDataset(Dataset):
    """The Million Songs Dataset class"""

    IMPORT_FOLDER = 'data/million_songs_dataset/'
    NAME = 'MSD-confidence-corrected'

    def __init__(self, name, *args, n_users=0,  **kwargs):
        super().__init__(*args, **kwargs)

        self.NAME = name
        self.base_folder = GENERATED.joinpath(f'dataset-{self.NAME}')
        self.data_folder = self.base_folder.joinpath('data/')
        self.n_users = int(n_users)

        self.user_item = None
        self.item_tag = None

    def import_data(self):
        """Randomly select self.n_users (all if n_users == 0) and import all
           their listenings

        Only listened songs are imported, only tags that are related to listened
        songs are imported
        """

        logger.info('Importing dataset')
        t = time.perf_counter()

        logger.debug('Reading user->item links file')
        user_item = pd.read_csv(
            Path(self.IMPORT_FOLDER).joinpath('msd_users.txt'),
            sep=' ',
            names=['node1_level', 'user', 'node2_level', 'item', 'rating'],
            dtype={
                'node1_level': np.int8,
                'node2_level': np.int8,
                'user': np.str,
                'item': np.str,
                'rating': np.float32
            },
            # nrows=1_000_000,
            engine='c'
        )[['user', 'item', 'rating']]

        logger.debug('Reading item->tag links file')
        item_tag = pd.read_csv(
            Path(self.IMPORT_FOLDER).joinpath('msd_tags.txt'),
            sep=' ',
            names=['node1_level', 'item', 'node2_level', 'tag', 'weight'],
            dtype={
                'node1_level': np.int8,
                'node2_level': np.int8,
                'item': np.str,
                'tag': np.str,
                'weight': np.int32
            },
            engine='c'
        )[['item', 'tag', 'weight']]

        # Select a portion of the dataset
        if self.n_users > 0:
            logger.debug(f'Sampling {self.n_users} users')
            rng = np.random.default_rng()

            users = user_item['user'].unique()
            selected_users = rng.choice(users, self.n_users, replace=False)

            user_item.set_index('user', inplace=True)
            user_item = user_item.loc[selected_users]
            user_item.reset_index(inplace=True)

            # Only keep songs that are listened to
            # Too slow when importing the whole dataset
            logger.debug(f'Removing songs not listened to')
            item_tag.set_index('item', inplace=True)
            item_tag = item_tag.loc[user_item['item']
                                    ].reset_index().drop_duplicates()

        logger.debug(
            f'Finished importing dataset in {time.perf_counter() - t}')

        self.user_item = user_item
        self.item_tag = item_tag

    def __str__(self):
        return self.NAME

################################################################################
# DATASET ANALYSIS                                                             #
################################################################################


class ImportDataset(luigi.Task):
    """Import a dataset from via its class (ex: MsdDataset)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        return {
            'user_item': luigi.local_target.LocalTarget(
                self.dataset.data_folder.joinpath('user_item.csv')
            ),
            'item_tag': luigi.local_target.LocalTarget(
                self.dataset.data_folder.joinpath('item_tag.csv')
            )
        }

    def run(self):
        for out in self.output().values():
            out.makedirs()

        self.dataset.import_data()

        self.dataset.user_item.to_csv(
            self.output()['user_item'].path, index=False)
        self.dataset.item_tag.to_csv(
            self.output()['item_tag'].path, index=False)


class BuildDatasetGraph(luigi.Task):
    """Build users-songs-tags graph"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def requires(self):
        return ImportDataset(dataset=self.dataset)

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('graph.pk'),
            format=luigi.format.Nop
        )

    def run(self):
        self.output().makedirs()

        user_item = pd.read_csv(self.input()['user_item'].path)
        item_tag = pd.read_csv(self.input()['item_tag'].path)

        graph = generate_graph(user_item, item_tag)
        graph.persist(self.output().path)

        del graph, user_item, item_tag


class DatasetInfo(luigi.Task):
    """Save information on dataset (number of links, users ...)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def requires(self):
        return BuildDatasetGraph(
            dataset=self.dataset
        )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('info.json')
        )

    def run(self):
        graph_file = self.input()
        graph = IndividualHerfindahlDiversities.recall(graph_file.path)

        with self.output().open('w') as file:
            json.dump(dataset_info(graph), file, indent=4)

        del graph


class ComputeUserVolume(luigi.Task):
    """Compute the number of songs listened by each user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('users_volume.csv')
        )

    def requires(self):
        return ImportDataset(
            dataset=self.dataset
        )

    def run(self):
        user_item = pd.read_csv(self.input()['user_item'].path)
        user_item[['user', 'item']] \
            .groupby('user') \
            .count() \
            .reset_index() \
            .rename(columns={'item': 'n_items'}) \
            .to_csv(self.output().path, index=False)


class PlotUserVolumeHistogram(luigi.Task):
    """Plot the user volume histogram"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        figures = self.dataset.data_folder.joinpath(f'figures')
        return luigi.LocalTarget(figures.joinpath('user_volume_histogram.png'))

    def requires(self):
        return ComputeUserVolume(
            dataset=self.dataset
        )

    def run(self):
        self.output().makedirs()
        user_volume = pd.read_csv(self.input().path)

        figure, ax = plot_histogram(
            user_volume['n_items'].to_numpy(),
            min_quantile=0
        )
        ax.set_xlabel('User volume')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of user volume')

        figure.savefig(self.output().path, format='png', dpi=300)

        del figure, ax, user_volume


class ComputeUsersDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def output(self):
        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath(f'users_diversities{alpha}.csv')
        )

    def requires(self):
        return BuildDatasetGraph(
            dataset=self.dataset
        )

    def run(self):
        graph_file = self.input()
        graph: IndividualHerfindahlDiversities = IndividualHerfindahlDiversities.recall(
            graph_file.path)

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2), alpha=self.alpha)

        pd.DataFrame({
            'user': list(diversities.keys()),
            'diversity': list(diversities.values())
        }).to_csv(self.output().path, index=False)

        del graph


class PlotUsersDiversitiesHistogram(luigi.Task):
    """Plot the histogram of user diversity"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def output(self):
        figures = self.dataset.data_folder.joinpath(f'figures')
        return luigi.LocalTarget(figures.joinpath(f'user_diversity{self.alpha}_histogram.png'))

    def requires(self):
        return ComputeUsersDiversities(
            dataset=self.dataset,
            alpha=self.alpha
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(
            diversities['diversity'].to_numpy(), min_quantile=0)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of user diversity index')
        fig.savefig(self.output().path, format='png', dpi=300)

        del fig, ax, diversities


class ComputeTagsDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def output(self):
        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath(f'tags_diversities{alpha}.csv')
        )

    def requires(self):
        return BuildDatasetGraph(
            dataset=self.dataset
        )

    def run(self):
        graph_file = self.input()
        graph = IndividualHerfindahlDiversities.recall(graph_file.path)

        graph.normalise_all()
        diversities = graph.diversities((2, 1, 0), alpha=self.alpha)

        pd.DataFrame({
            'tag': list(diversities.keys()),
            'diversity': list(diversities.values())
        }).to_csv(self.output().path, index=False)

        del graph, diversities


class PlotTagsDiversitiesHistogram(luigi.Task):
    """Plot the histogram of user diversity"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    def output(self):
        figures = self.dataset.data_folder.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(f'tag_diversity{self.alpha}_histogram.png'))

    def requires(self):
        return ComputeTagsDiversities(
            dataset=self.dataset,
            alpha=self.alpha
        )

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(
            diversities['diversity'].to_numpy(), min_quantile=0)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('Tag count')
        ax.set_title('Histogram of tag diversity index')
        fig.savefig(self.output().path, format='png', dpi=300)

        del fig, ax, diversities


################################################################################
# TRAIN/TEST DATASETS GENERATION AND ANALYSIS                                  #
################################################################################
class GenerateTrainTest(luigi.Task):
    """Import a dataset (with adequate format) and generate train/test data"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    def requires(self):
        return ImportDataset(dataset=self.dataset)

    def output(self):
        return {
            'train': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(
                    f'train-{int((1 - self.user_fraction) * 100)}.csv'),
                format=Nop
            ),
            'test': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(
                    f'test-{int(self.user_fraction * 100)}.csv'),
                format=Nop
            )
        }

    def run(self):
        for out in self.output().values():
            out.makedirs()

        user_item = pd.read_csv(self.input()['user_item'].path)
        train, test = split_dataset(user_item, self.user_fraction)

        train.to_csv(self.output()['train'].path, index=False)
        test.to_csv(self.output()['test'].path, index=False)

        del user_item, train, test


class TrainTestInfo(luigi.Task):
    """Compute information about the training and testings datasets (n_users ...)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    def requires(self):
        return GenerateTrainTest(
            dataset=self.dataset,
            user_fraction=self.user_fraction
        )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('train_test_info.json')
        )

    def run(self):
        train = pd.read_csv(self.input()['train'].path)
        test = pd.read_csv(self.input()['test'].path)
        info = {}

        info['train'] = {
            'n_users': len(train['user'].unique()),
            'n_items': len(train['item'].unique()),
            'n_user_item_links': len(train)
        }
        info['test'] = {
            'n_users': len(test['user'].unique()),
            'n_items': len(test['item'].unique()),
            'n_user_item_links': len(test)
        }

        with self.output().open('w') as file:
            json.dump(info, file, indent=4)

        del train, test


class BuildTrainTestGraphs(luigi.Task):
    """Build users-songs-tags graph for the train and test sets"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    def requires(self):
        return {
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.user_fraction
            ),
            'dataset': ImportDataset(
                dataset=self.dataset
            )
        }

    def output(self):
        return {
            'train': luigi.LocalTarget(
                self.dataset.data_folder.joinpath('trainset-graph.pk'),
                format=luigi.format.Nop
            ),
            'test': luigi.LocalTarget(
                self.dataset.data_folder.joinpath('testset-graph.pk'),
                format=luigi.format.Nop
            ),
        }

    def run(self):
        self.output()['train'].makedirs()

        train_user_item = pd.read_csv(self.input()['train_test']['train'].path)
        test_user_item = pd.read_csv(self.input()['train_test']['test'].path)
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)

        train_graph = generate_graph(train_user_item, item_tag)
        test_graph = generate_graph(test_user_item, item_tag)

        train_graph.persist(self.output()['train'].path)
        test_graph.persist(self.output()['test'].path)

        del train_graph, test_graph, train_user_item, test_user_item, item_tag


class ComputeTrainTestUserDiversity(luigi.Task):
    """Compute the user diversity of the users in the train and test sets"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    def requires(self):
        return BuildTrainTestGraphs(
            dataset=self.dataset,
            user_fraction=self.user_fraction
        )

    def output(self):
        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return {
            'train': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(
                    f'trainset_users_diversities{alpha}.csv')
            ),
            'test': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(
                    f'testset_users_diversities{alpha}.csv')
            ),
        }

    def run(self):
        train_graph = IndividualHerfindahlDiversities.recall(
            self.input()['train'].path
        )
        test_graph = IndividualHerfindahlDiversities.recall(
            self.input()['test'].path
        )

        train_graph.normalise_all()
        train_diversities = train_graph.diversities(
            (0, 1, 2), alpha=self.alpha)

        test_graph.normalise_all()
        test_diversities = test_graph.diversities((0, 1, 2), alpha=self.alpha)

        pd.DataFrame({
            'user': list(train_diversities.keys()),
            'diversity': list(train_diversities.values())
        }).to_csv(self.output()['train'].path, index=False)

        pd.DataFrame({
            'user': list(test_diversities.keys()),
            'diversity': list(test_diversities.values())
        }).to_csv(self.output()['test'].path, index=False)

        del train_graph, test_graph, train_diversities, test_diversities


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


################################################################################
# MODEL TRAINING/EVALUATION, RECOMMENDATION GENERATION                         #
################################################################################
class TrainModel(luigi.Task):
    """Train a given model and save it"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    regularization = luigi.parameter.FloatParameter(
        default=.1, description='Regularization factor for the norm of user/item factors'
    )
    confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    save_training_loss = luigi.BoolParameter(
        default=False, description='Save the value of the training loss at each iteration'
    )

    def requires(self):
        return GenerateTrainTest(
            dataset=self.dataset,
            user_fraction=self.user_fraction
        )

    def output(self):
        model = self.dataset.base_folder.joinpath(
            f'model-{self.n_iterations}it-{self.n_factors}f-{str(self.regularization).replace(".", "_")}reg-{int(self.confidence_factor)}c/'
        )

        out = {'model': luigi.LocalTarget(model.joinpath('model.bpk'))}

        if self.save_training_loss == True:
            out['train_loss'] = luigi.LocalTarget(
                model.joinpath(f'train_loss.csv')
            )

        return out

    def run(self):
        for out in self.output().values():
            out.makedirs()

        # TODO : verify that this syntax works well with the now used dict input
        train_file, test_file = self.input()

        train = pd.read_csv(self.input()['train'].path)

        model, loss = train_model(
            train,
            n_factors=self.n_factors,
            n_iterations=self.n_iterations,
            confidence_factor=self.confidence_factor,
            regularization=self.regularization,
            save_training_loss=self.save_training_loss
        )

        binpickle.dump(model, self.output()['model'].path)

        if self.save_training_loss == True:
            pd.DataFrame({'iteration': np.arange(loss.shape[0]), 'train_loss': loss}) \
                .to_csv(self.output()['train_loss'].path, index=False)

        del train, model, loss


class TrainBaselines(luigi.Task):
    """ Train baseline recommender models """
    pass


class PlotTrainLoss(luigi.Task):
    """Plot the loss of a model for each iteration step"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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

    def requires(self):
        return TrainModel(
            dataset=self.dataset,
            n_iterations=self.model_n_iterations,
            n_factors=self.model_n_factors,
            regularization=self.model_regularization,
            confidence_factor=self.model_confidence_factor,
            user_fraction=self.model_user_fraction,
            save_training_loss=True
        )

    def output(self):
        model = Path(self.input()['model'].path).parent

        return {
            'png': luigi.LocalTarget(model.joinpath('training-loss.png'), format=Nop),
            'latex': luigi.LocalTarget(model.joinpath('training-loss.tex'), format=Nop)
        }

    def run(self):
        loss = pd.read_csv(self.input()['train_loss'].path)

        iterations = loss['iteration'].to_numpy()
        loss = loss['train_loss'].to_numpy()

        fig, ax = pl.subplots()
        ax.semilogy(iterations, loss)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        fig.savefig(self.output()['png'].path, format='png', dpi=300)

        tikzplotlib.save(self.output()['latex'].path)


class GenerateRecommendations(luigi.Task):
    """Generate recommendations for users in test dataset with a given model"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            'data': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'model': TrainModel(
                dataset=self.dataset,
                n_iterations=self.model_n_iterations,
                n_factors=self.model_n_factors,
                regularization=self.model_regularization,
                user_fraction=self.model_user_fraction,
                save_training_loss=False,
                confidence_factor=self.model_confidence_factor
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent

        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}.csv')
        )

    def run(self):
        self.output().makedirs()

        model = binpickle.load(self.input()['model']['model'].path)
        ratings = pd.read_csv(self.input()['data']['test'].path)

        generate_recommendations(
            model,
            ratings,
            n_recommendations=self.n_recommendations
        ).to_csv(self.output().path, index=False)

        del ratings


class GeneratePredictions(luigi.Task):
    """Compute the predicted rating values for the user-item pairs in the test set"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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

    train_predictions = luigi.parameter.BoolParameter(
        default=False, description='Whether or not to compute predictions for the train set'
    )

    def requires(self):
        return {
            'data': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'model': TrainModel(
                dataset=self.dataset,
                n_iterations=self.model_n_iterations,
                n_factors=self.model_n_factors,
                regularization=self.model_regularization,
                user_fraction=self.model_user_fraction,
                save_training_loss=False,
                confidence_factor=self.model_confidence_factor
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent

        out = {}
        out['test'] = luigi.LocalTarget(
            model.joinpath(f'test-predictions.csv'))

        if self.train_predictions:
            out['train'] = luigi.LocalTarget(
                model.joinpath(f'train-predictions.csv')
            )

        return out

    def run(self):
        self.output()['test'].makedirs()

        test_user_item = pd.read_csv(self.input()['data']['test'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_predictions(
            model,
            test_user_item,
        ).to_csv(self.output()['test'].path, index=False)

        if self.train_predictions:
            train_user_item = pd.read_csv(self.input()['data']['train'].path)
            generate_predictions(
                model,
                train_user_item,
            ).to_csv(self.output()['train'].path, index=False)

            del train_user_item

        del test_user_item, model


class EvaluateUserRecommendations(luigi.Task):
    """Compute evaluations metrics on a trained model"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations,
                model_confidence_factor=self.model_confidence_factor
            ),
            'dataset': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            )
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent

        return luigi.LocalTarget(
            model.joinpath(
                f'users_eval_{self.n_recommendations}-recommendations.csv'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        recommendations = pd.read_csv(self.input()['recommendations'].path)
        test = pd.read_csv(self.input()['dataset']['test'].path)

        # # NOTE : Issue appears when the two following condition is met :
        # #   - a user had only one listening and it is put in the test set
        # # NOTE : seems not to be necessary anymore thanks to the better train/test sampling
        # #
        # # Then, the model will not know the user, thus will not be able to
        # # recommend songs to this user. Therefore, to avoid issues, we simply
        # # discard users with no recommendations from the test set
        # recommendations.set_index('user', inplace=True)
        # test.set_index('user', inplace=True)

        # missing = test.index.difference(recommendations.index)
        # common = test.index.intersection(recommendations.index).unique()
        # print(f'In test set but not recommended : {missing}')

        # recommendations = recommendations.loc[common].reset_index()
        # test.reset_index(inplace=True)

        metrics_names = ['ndcg', 'precision', 'recip_rank', 'recall']
        metrics = evaluate_model_recommendations(
            recommendations,
            test,
            metrics_names
        )[metrics_names].reset_index()

        metrics.to_csv(self.output().path, index=False)

        del recommendations, test, metrics


class PlotUserEvaluationHistogram(luigi.Task):
    """Plot the histogram of an evaluation metric among all users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
        return EvaluateUserRecommendations(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            model_confidence_factor=self.model_confidence_factor,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input().path).parent.joinpath('figures/')

        return luigi.LocalTarget(
            model.joinpath(
                f'users_eval_{self.n_recommendations}-recommendations_histogram.png'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        metrics = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(
            metrics['ndcg'].to_numpy(), min_quantile=0, max_quantile=0.99)
        ax.set_xlabel('NDCG value')
        ax.set_ylabel('User count')
        fig.savefig(self.output().path, format='png', dpi=300)


class EvaluateModel(luigi.Task):
    """Compute evaluations metrics on a trained model"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            'model': TrainModel(
                dataset=self.dataset,
                n_iterations=self.model_n_iterations,
                n_factors=self.model_n_factors,
                regularization=self.model_regularization,
                user_fraction=self.model_user_fraction,
                save_training_loss=False,
                confidence_factor=self.model_confidence_factor
            ),
            'predictions': GeneratePredictions(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                train_predictions=True,
                model_confidence_factor=self.model_confidence_factor
            ),
            'user_eval': EvaluateUserRecommendations(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                model_confidence_factor=self.model_confidence_factor,
                n_recommendations=self.n_recommendations
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent

        return luigi.LocalTarget(
            model.joinpath(
                f'{self.n_recommendations}-recommendations_model_eval.json'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        test_predictions = pd.read_csv(
            self.input()['predictions']['test'].path)
        train_predictions = pd.read_csv(
            self.input()['predictions']['train'].path)

        model = binpickle.load(self.input()['model']['model'].path)

        metrics: pd.DataFrame = pd.read_csv(self.input()['user_eval'].path)
        metrics = metrics.set_index('user')
        metrics = metrics.mean()

        metrics['test_loss'] = evaluate_model_loss(model, test_predictions)
        metrics['train_loss'] = evaluate_model_loss(model, train_predictions)

        metrics.to_json(
            self.output().path,
            orient='index',
            indent=4
        )

        del metrics


class TuneModelHyperparameters(luigi.Task):
    """Evaluate a model on a hyperparameter grid and get the best combination

    The best combination is defined as the one that yelds the best NDCG
    """

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors_values = luigi.parameter.ListParameter(
        description='List of numer of user/item latent factors'
    )
    model_regularization_values = luigi.parameter.FloatParameter(
        description='List of regularization factors for the norm of user/item factors'
    )
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        grid = np.meshgrid(self.model_n_factors_values,
                           self.model_regularization_values)
        # Transform np.meshgrid into list od tuples, with each tuple
        # representing a set of parameters to train the model against
        self.hyperparameters = list(zip(*map(lambda x: x.flatten(), grid)))

        required = {}

        for n_factors, regularization in self.hyperparameters:
            required[(n_factors, regularization)] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations,
                model_confidence_factor=self.model_confidence_factor
            )

        return required

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')

        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.model_n_factors_values}factors_{self.model_regularization_values}reg_{self.n_recommendations}reco_{self.model_confidence_factor}c_model_eval.json'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()
        metrics = pd.DataFrame()

        for (n_factors, regularization), metrics_file in self.input().items():
            metric = pd.read_json(
                metrics_file.path, orient='index').transpose()
            metric['n_factors'] = n_factors
            metric['regularization'] = regularization

            metrics = pd.concat((metrics, metric))

        metrics.set_index(['n_factors', 'regularization'], inplace=True)
        optimal = {}

        opt_n_factors, opt_regularization = metrics.index[metrics['ndcg'].argmax(
        )]
        optimal['n_factors'] = float(opt_n_factors)
        optimal['regularization'] = float(opt_regularization)

        with open(self.output().path, 'w') as file:
            json.dump(optimal, file, indent=4)

        del metrics


class PlotModelTuning(luigi.Task):
    """Plot the 2D matrix of the model performance (ndcg value) on a 
       hyperparameter grid"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors_values = luigi.parameter.ListParameter(
        description='List of numer of user/item latent factors'
    )
    model_regularization_values = luigi.parameter.FloatParameter(
        description='List of regularization factors for the norm of user/item factors'
    )
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )
    model_confidence_factor = luigi.parameter.FloatParameter(
        default=40.0, description='The multplicative factor used to extract confidence values from listenings counts'
    )

    tuning_metric = luigi.parameter.Parameter(
        default='test_loss', description='Wich metric to plot for each n_factors-regularization pair'
    )
    tuning_best = luigi.parameter.ChoiceParameter(
        default='min', choices=['min', 'max'], description='Best value for the tuning metric greater is better or lower is better'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        grid = np.meshgrid(self.model_n_factors_values,
                           self.model_regularization_values)
        # Transform np.meshgrid into list od tuples, with each tuple
        # representing a set of parameters to train the model against
        self.hyperparameters = list(zip(*map(lambda x: x.flatten(), grid)))

        required = {}

        for n_factors, regularization in self.hyperparameters:
            required[(n_factors, regularization)] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return required

    def output(self):
        aggregated = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')

        return {
            'png': luigi.LocalTarget(
                aggregated.joinpath(
                    f'{self.model_n_factors_values}'
                    + f'factors_{self.model_regularization_values}'
                    + f'reg_{self.n_recommendations}'
                    + f'reco_{self.model_confidence_factor}'
                    + f'c_model_eval_{self.tuning_metric}.png'),
                format=Nop
            ),
            'latex': luigi.LocalTarget(
                aggregated.joinpath(
                    f'{self.model_n_factors_values}'
                    + f'factors_{self.model_regularization_values}'
                    + f'reg_{self.n_recommendations}'
                    + f'reco_{self.model_confidence_factor}'
                    + f'c_model_eval_{self.tuning_metric}.tex'),
                format=Nop
            )
        }

    def run(self):
        self.output()['png'].makedirs()
        metrics = pd.DataFrame()

        for (n_factors, regularization), metrics_file in self.input().items():
            metric = pd.read_json(
                metrics_file.path, orient='index').transpose()
            metric['n_factors'] = n_factors
            metric['regularization'] = regularization

            metrics = pd.concat((metrics, metric))

        metrics_matrix = metrics.pivot(index='n_factors', columns='regularization')[
            self.tuning_metric]
        metrics_matrix_n = metrics_matrix.to_numpy()

        fig, ax = pl.subplots()

        # Display the matrix as a heatmap
        img = ax.imshow(metrics_matrix_n)

        # Create the color bar
        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel(self.tuning_metric.replace(
            "_", " "), rotation=-90, va="bottom")

        # Set the x and y axis values
        ax.set_xticks(list(range(len(metrics_matrix.columns))))
        ax.set_xticklabels(
            [f'{value:.0e}' for value in metrics_matrix.columns])

        ax.set_yticks(list(range(len(metrics_matrix.index))))
        ax.set_yticklabels(list(metrics_matrix.index))

        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        pl.setp(ax.get_xticklabels(), rotation=-40,
                rotation_mode="anchor", ha="right")

        # Annotate the best value
        if self.tuning_best == 'min':
            opt_n_factors, opt_regularization = np.unravel_index(
                metrics_matrix_n.flatten().argmin(),
                metrics_matrix_n.shape
            )
            opt_text = 'MIN'
            opt_color = 'white'
        else:
            opt_n_factors, opt_regularization = np.unravel_index(
                metrics_matrix_n.flatten().argmax(),
                metrics_matrix_n.shape
            )
            opt_text = 'MAX'
            opt_color = 'black'

        ax.text(
            opt_regularization,
            opt_n_factors,
            opt_text,
            ha="center",
            va="center",
            color=opt_color
        )

        ax.set_ylabel('Number of latent factors')
        ax.set_xlabel('Regularization coefficient')

        fig.savefig(self.output()['png'].path, format='png', dpi=300)
        tikzplotlib.save(self.output()['latex'].path)

        pl.clf()

        del fig, ax, metrics, metrics_matrix


class PlotModelEvaluationVsLatentFactors(luigi.Task):
    """Compute model evaluation metrics against the number of factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors_values = luigi.parameter.ListParameter(
        description='List of numer of user/item latent factors'
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
        req = {}

        for n_factors in self.model_n_factors_values:
            req[n_factors] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return req

    def output(self):
        aggregated = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')

        return luigi.LocalTarget(
            aggregated.joinpath(
                f'{self.model_n_factors_values}'
                + f'factors_{self.model_regularization}'
                + f'reg_{self.n_recommendations}'
                + f'reco_{self.model_confidence_factor}'
                + f'model_eval.png'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        data: pd.DataFrame = pd.DataFrame(
            index=self.model_n_factors_values,
            columns=['NDCG', 'precision']
        )

        for n_factors in self.model_n_factors_values:
            metric = pd.read_json(
                self.input()[n_factors].path,
                orient='index'
            ).transpose()

            data.loc[n_factors, 'NDCG'] = float(metric['ndcg'][0])
            data.loc[n_factors, 'precision'] = float(metric['precision'][0])
            data.loc[n_factors, 'recip_rank'] = float(metric['recip_rank'][0])
            data.loc[n_factors, 'test_loss'] = float(metric['test_loss'][0])

        data = data.subtract(data.min())
        data = data.divide(data.max())

        data.plot(xlabel="number of factors",
                  ylabel="metric (scaled and centered)", logx=True)
        pl.savefig(self.output().path, format='png', dpi=300)


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

    model_n_iterations = luigi.parameter.IntParameter(
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
            'dataset': ImportDataset(self.dataset),
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent

        return luigi.LocalTarget(
            model.joinpath(
                f'recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations: pd.DataFrame = pd.read_csv(
            self.input()['recommendations'].path)

        graph = generate_recommendations_graph(recommendations, item_tag)
        graph.persist(self.output().path)

        del graph


class ComputeRecommendationDiversities(luigi.Task):
    """Compute the diversity of the songs recommended to users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
        return BuildRecommendationGraph(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input().path).parent

        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return luigi.LocalTarget(
            model.joinpath(
                f'recommendations-{self.n_recommendations}-users_diversities{alpha}.csv')
        )

    def run(self):
        graph = IndividualHerfindahlDiversities.recall(self.input().path)

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2), alpha=self.alpha)

        pd.DataFrame({
            'user': list(diversities.keys()),
            'diversity': list(diversities.values())
        }).to_csv(self.output().path, index=False)

        del graph, diversities


class PlotRecommendationsUsersDiversitiesHistogram(luigi.Task):
    """Plot the histogram of recommendations diversity for each user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
        return ComputeRecommendationDiversities(
            dataset=self.dataset,
            alpha=self.alpha,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        figures = Path(self.input().path).parent.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}-recommendation_user_diversity{self.alpha}_histogram.png'
        ))

    def run(self):
        self.output().makedirs()

        diversities = pd.read_csv(self.input().path)
        fig, ax = plot_histogram(
            diversities['diversity'].to_numpy(), min_quantile=0, max_quantile=1)

        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of recommendations diversity index')
        pl.savefig(self.output().path, format='png', dpi=300)

        del fig, ax, diversities


class PlotRecommendationDiversityVsUserDiversity(luigi.Task):
    """Plot the diversity of the recommendations associated to each user with
    respect to the user diversity before recommendations (ie train set
    diversity)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
                user_fraction=self.model_user_fraction,
            ),
            'diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction,
            ),
            'recommendation_diversity': ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
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


class BuildRecommendationsWithListeningsGraph(luigi.Task):
    """Add the recommendations to the train user-item-tag graph

    This describes the situation where a user listens only to recommendations
    after the model has been trained"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            'graph': BuildTrainTestGraphs(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'train_test': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'recommendations': GenerateRecommendations(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent

        return luigi.LocalTarget(
            model.joinpath(
                f'listenings-recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(
            self.input()['graph']['test'].path
        )

        # Used to compute the volume a user would have listened to if listening its music
        user_item = pd.read_csv(self.input()['train_test']['test'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path)

        graph = build_recommendations_listenings_graph(
            graph,
            user_item,
            recommendations
        )
        graph.persist(self.output().path)

        del graph, user_item, recommendations


class ComputeRecommendationWithListeningsUsersDiversities(luigi.Task):
    """Compute the diversity of the users who were recommended, assuming they
       listened to all recommendations"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
        return BuildRecommendationsWithListeningsGraph(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input().path).parent

        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return luigi.LocalTarget(
            model.joinpath(
                f'listenings-recommendations-{self.n_recommendations}-users_diversities{alpha}.csv')
        )

    def run(self):
        graph = IndividualHerfindahlDiversities.recall(
            self.input().path
        )

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2), alpha=self.alpha)
        diversities = pd.DataFrame({
            'user': list(diversities.keys()),
            'diversity': list(diversities.values())
        })

        diversities.to_csv(self.output().path, index=False)

        del graph, diversities


class ComputeRecommendationWithListeningsUsersDiversityIncrease(luigi.Task):
    """Compare the diversity of a user if they start listenings only to
    recommendations or if they continue to listen their music"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            'with_recommendations': ComputeRecommendationWithListeningsUsersDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'original': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
            )
        }

    def output(self):
        model = Path(self.input()['with_recommendations'].path).parent

        # Avoid issues where 0.0 and 0 lead to different file titles
        alpha = float(self.alpha)
        alpha = int(alpha) if alpha.is_integer() else alpha

        return luigi.LocalTarget(
            model.joinpath(
                f'listenings-recommendations-{self.n_recommendations}-users_diversities{alpha}_increase.csv')
        )

    def run(self):
        with_recommendations = pd.read_csv(self.input()['with_recommendations'].path) \
            .set_index('user')
        original = pd.read_csv(self.input()['original']['test'].path) \
            .set_index('user')

        deltas = (with_recommendations['diversity'] - original['diversity']) \
            .dropna() \
            .reset_index()

        deltas.to_csv(self.output().path, index=False)

        del original, deltas


class PlotDiversitiesIncreaseHistogram(luigi.Task):
    """Plot the histogram of recommendations diversity for each user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            model_n_iterations=self.model_n_iterations,
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


class PlotUserDiversityIncreaseVsUserDiversity(luigi.Task):
    """Plot the user diversity increase with respect to the user diversity
       before recommendations"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
        return {
            'dataset': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            ),
            'user_diversity': ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=self.alpha,
                user_fraction=self.model_user_fraction
            ),
            'diversity_increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        figures = Path(
            self.input()['diversity_increase'].path).parent.joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_increase_vs_original_diversity.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations}-recommendations_diversity{self.alpha}_increase_vs_original_diversity.tex'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()
        diversities = pd.read_csv(self.input()['user_diversity']['train'].path)
        increase = pd.read_csv(self.input()['diversity_increase'].path).rename(
            columns={'diversity': 'increase'})

        # compute user volume
        user_item = pd.read_csv(self.input()['dataset']['train'].path)
        volume = user_item.groupby('user')['rating'].sum() \
            .rename('volume')

        # inner join, only keep users for whom we calculated a diversity increase value
        merged = increase.merge(diversities, on='user')
        merged = merged.merge(volume, on='user')
        merged = merged[merged['increase'] != 0]

        if self.bounds == None:
            self.bounds = [None, None, None, None]

        a, b = linear_regression(merged, 'diversity', 'increase')
        x = merged[(self.bounds[0] < merged['diversity']) & (merged['diversity'] < self.bounds[1])]['diversity'] \
            .sort_values().to_numpy()
        y = a * x + b

        ax: pl.Axes = merged.plot.scatter(
            x='diversity',
            y='increase',
            marker='+',
            c='volume',
            colormap='viridis',
            norm=colors.LogNorm(vmin=volume.min(), vmax=volume.max()),
            xlim=self.bounds[:2],
            ylim=self.bounds[2:],
            colorbar=self.show_colorbar
        )

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
                c='red'
            )

        pl.xlabel('organic diversity')
        pl.ylabel('diversity increase')
        ax.get_legend().remove()

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
            code = code.replace('colorbar,', 'colorbar horizontal,')
            code = code.replace('ytick={0,1,2,3}', 'xtick={0,1,2,3}')
            code = code.replace('yticklabels', 'xticklabels')
            code = code.replace(
                'ylabel={volume}', 'xlabel={volume},xticklabel pos=upper,at={(0,1.2)},anchor=north west')

            file.write(code)

        pl.clf()

        del diversities, increase, merged


class ComputeUserRecommendationsTagsDistribution(luigi.Task):
    """Compute the distributions of tags of the items recommended to a given user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    user = luigi.parameter.Parameter(
        description="The hash of the studied user"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
            model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
            model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
            model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                    model_n_iterations=self.model_n_iterations,
                    model_n_factors=self.model_n_factors,
                    model_regularization=self.model_regularization,
                    model_user_fraction=self.model_user_fraction,
                    n_recommendations=n_recommendations
                )
                req[f'reco{n_recommendations}_div{alpha}_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                    dataset=self.dataset,
                    alpha=alpha,
                    model_n_iterations=self.model_n_iterations,
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


################################################################################
# HYPERPARAMETERS ANALYSIS                                                     #
################################################################################
class ComputeRecommendationDiversityVsLatentFactors(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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

    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        tasks['dataset'] = ImportDataset(dataset=self.dataset)

        for n_factors in self.n_factors_values:
            tasks[n_factors] = GenerateRecommendations(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=max(self.n_recommendations_values)
            )

        return tasks

    def output(self):
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(aggregated.joinpath(
            f'{self.n_recommendations_values}recommendations_diversity{self.alpha}_vs_{self.n_factors_values}factors_{self.model_regularization}reg.csv'
        ))

    def run(self):
        self.output().makedirs()
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)

        data = pd.DataFrame(
            index=self.n_factors_values,
            columns=[
                f'{n_recommendations} recommendations' for n_recommendations in self.n_recommendations_values]
        )

        for n_factors in self.n_factors_values:
            recommendations: pd.DataFrame = pd.read_csv(
                self.input()[n_factors].path)

            for n_recommendations in self.n_recommendations_values:
                graph = generate_recommendations_graph(
                    recommendations[recommendations['rank']
                                    <= n_recommendations],
                    item_tag
                )

                graph.normalise_all()
                diversities = graph.diversities((0, 1, 2), alpha=self.alpha)
                data.loc[n_factors, f'{n_recommendations} recommendations'] = \
                    sum(diversities.values()) / len(diversities)

        data = data.reset_index().rename(columns={'index': 'n_factors'})
        data.to_csv(self.output().path)


class PlotRecommendationDiversityVsLatentFactors(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
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

    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        req = {}

        for n_factors in self.n_factors_values:
            req[n_factors] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=50
            )

        req['diversity'] = ComputeRecommendationDiversityVsLatentFactors(
            dataset=self.dataset,
            alpha=self.alpha,
            model_n_iterations=self.model_n_iterations,
            n_factors_values=self.n_factors_values,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations_values=self.n_recommendations_values,
        )

        return req

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return {
            'png': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}recommendations_diversity{self.alpha}_vs_{self.n_factors_values}factors_{self.model_regularization}reg.png'
            )),
            'latex': luigi.LocalTarget(figures.joinpath(
                f'{self.n_recommendations_values}recommendations_diversity{self.alpha}_vs_{self.n_factors_values}factors_{self.model_regularization}reg.tex'
            )),
        }

    def run(self):
        self.output()['png'].makedirs()
        data: pd.DataFrame = pd.read_csv(
            self.input()['diversity'].path, index_col=0)

        data = data[1:].set_index('n_factors')
        data['ndcg'] = 0

        for n_factors in self.n_factors_values[1:]:
            metric = pd.read_json(
                self.input()[n_factors].path,
                orient='index'
            ).transpose()
            data.loc[n_factors, 'ndcg'] = metric['ndcg'][0]

        data = data.subtract(data.min())
        data = data.divide(data.max())
        data = data.reset_index()

        _, ax = pl.subplots(1, 1)

        data.plot(
            ax=ax,
            x='n_factors',
            y='ndcg',
            xlabel='number of factors',
            ylabel='diversity (normalized)',
            logx=True,
            style='--',
            linewidth='2',
            color='black',
            legend='NDCG'
        )

        for n_reco in self.n_recommendations_values:
            data.plot(
                ax=ax,
                x='n_factors',
                y=f'{n_reco} recommendations',
                xlabel='number of factors',
                ylabel='diversity (normalized)',
                logx=True,
                style='-',
                linewidth='1',
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )
            tasks[(n_factors, 'metrics')] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
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


class PlotDiversityVsRegularization(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    model_regularization_values = luigi.parameter.ListParameter(
        description='List of regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        for regularization in self.model_regularization_values:
            tasks[(regularization, 'diversities')] = ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )
            tasks[(regularization, 'metrics')] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}reco-recommendations_diversity{self.alpha}_vs_{self.model_regularization_values}reg-{self.model_n_factors}f.png'
        ))

    def run(self):
        self.output().makedirs()

        mean_diversities = []
        metrics = pd.DataFrame()
        regularization = []

        for key, value in self.input().items():
            if key[1] == 'diversities':
                diversities = pd.read_csv(value.path)

                regularization.append(key[0])
                mean_diversities.append(diversities['diversity'].mean())

            elif key[1] == 'metrics':
                metric = pd.read_json(value.path, orient='index').transpose()
                metric['regularization'] = key[0]
                metrics = pd.concat((metrics, metric))

        metrics.set_index('regularization', inplace=True)
        metrics = metrics / metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()

        # Add plots
        div_line = ax1.semilogx(
            regularization, mean_diversities, color='green', label='diversity')
        ax1.set_xlabel('Regularization coefficient')
        ax1.set_ylabel('mean diversity')

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

        pl.title('User diversity of recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.clf()


class PlotDiversityIncreaseVsRegularization(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    alpha = luigi.parameter.FloatParameter(
        default=2, description="The true diversity order"
    )

    model_n_iterations = luigi.parameter.IntParameter(
        default=10, description='Number of training iterations'
    )
    model_n_factors = luigi.parameter.IntParameter(
        default=30, description='Number of user/item latent facors'
    )
    model_regularization_values = luigi.parameter.ListParameter(
        description='List of regularization factor for the norm of user/item factors'
    )

    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        for regularization in self.model_regularization_values:
            tasks[(regularization, 'deltas')] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                alpha=self.alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )
            tasks[(regularization, 'metrics')] = EvaluateModel(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath(
            'aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}recommendations_diversity{self.alpha}_increase_vs_{self.model_regularization_values}reg_{self.model_n_factors}f.png'
        ))

    def run(self):
        self.output().makedirs()

        mean_deltas = []
        metrics = pd.DataFrame()
        regularization = []

        for key, value in self.input().items():
            if key[1] == 'deltas':
                diversities = pd.read_csv(value.path)

                regularization.append(key[0])
                mean_deltas.append(diversities['diversity'].mean())

            elif key[1] == 'metrics':
                metric = pd.read_json(value.path, orient='index').transpose()
                metric['regularization'] = key[0]
                metrics = pd.concat((metrics, metric))

        metrics.set_index('regularization', inplace=True)
        metrics = metrics / metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()

        # Add plots
        div_line = ax1.semilogx(
            regularization, mean_deltas, color='green', label='diversity')
        ax1.set_xlabel('Regularization coefficient')
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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


class AnalyseUser(luigi.Task):
    """ Look at the items listened by a user, its diversity before and after 
        recommendation etc... """

    user_id = luigi.parameter.Parameter(
        description='The id string of the user'
    )

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model_n_iterations = luigi.parameter.IntParameter(
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
    alpha_values = luigi.parameter.ListParameter(
        description="List of diversity order values to consider"
    )

    return_info = luigi.parameter.BoolParameter(
        default=True,
        description='Whether to return the info and distributions'
    )

    def requires(self):
        req = {
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
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'recommendation_graph': BuildRecommendationGraph(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'recommendation_with_listen': BuildRecommendationsWithListeningsGraph(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'diversity': [],
            'recommendation_diversity': [],
            'diversity_after_reco': [],
        }

        for alpha in self.alpha_values:
            req['diversity'].append(ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                alpha=alpha,
                user_fraction=self.model_user_fraction,
            ))

            req['recommendation_diversity'].append(ComputeRecommendationDiversities(
                dataset=self.dataset,
                alpha=alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations,
            ))

            req['diversity_after_reco'].append(ComputeRecommendationWithListeningsUsersDiversities(
                dataset=self.dataset,
                alpha=alpha,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations,
            ))

        return req

    def output(self):
        model = Path(self.input()['recommendations'].path).parent
        return luigi.LocalTarget(model.joinpath(
            f'{self.alpha_values}alpha'
            f'-{self.n_recommendations}reco'
            f'-user_{self.user_id}-info.json'
        ))

    def run(self):
        test = pd.read_csv(self.input()['train_test']['test'].path)
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path)
        song_info = get_msd_song_info()

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
            'model_n_iterations': self.model_n_iterations,
            'model_n_factors': self.model_n_factors,
            'model_regularization': self.model_regularization,
            'model_user_fraction': self.model_user_fraction,
            'n_recommendations': self.n_recommendations,
        }

        # Organic diversity
        info['organic_diversity'] = {}
        for i, alpha in enumerate(self.alpha_values):
            diversities = pd.read_csv(
                self.input()['diversity'][i]['train'].path
            ).set_index('user')
            info['organic_diversity'][alpha] = float(
                diversities.loc[self.user_id]['diversity'])

        # Recommendation diversity
        info['recommendation_diversity'] = {}
        for i, alpha in enumerate(self.alpha_values):
            diversities = pd.read_csv(
                self.input()['recommendation_diversity'][i].path
            ).set_index('user')
            info['recommendation_diversity'][alpha] = float(
                diversities.loc[self.user_id]['diversity'])

        # Recommendation + listennings diversity
        info['diversity_after_reco'] = {}
        for i, alpha in enumerate(self.alpha_values):
            diversities = pd.read_csv(
                self.input()['diversity_after_reco'][i].path
            ).set_index('user')
            info['diversity_after_reco'][alpha] = float(
                diversities.loc[self.user_id]['diversity'])

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

        if self.return_info:
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

            # dist = np.array(listened_tag_distribution)
            # dist = dist / np.sum(dist)
            # print('ORGANIC DIVERSITY 2', 1 / np.sum(dist**2))
            # print('ORGANIC DIVERSITY 0', len(dist))

            reco_listen_graph = IndividualHerfindahlDiversities.recall(
                self.input()['recommendation_with_listen'].path
            )
            reco_listen_graph.normalise_all()
            distribution = reco_listen_graph.spread_node(
                self.user_id, (0, 1, 2)
            )
            # print('N_TAGS', len(distribution))
            # print('DIV manual', sum(1 for x in distribution.values() if x > 0))
            # print('DIVERSITY all', reco_listen_graph.diversities(
            #     (0, 1, 2), alpha=0)[self.user_id])
            # print('SUM', sum(distribution.values()))
            # print(distribution)

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

            return info, listened_tag_distribution, recommended_tag_distribution


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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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

    model_n_iterations = luigi.parameter.IntParameter(
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
                model_n_iterations=self.model_n_iterations,
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


################################################################################
# UTILS                                                                        #
################################################################################
class CollectAllModelFigures(luigi.Task):
    """Collect all figures related to a dataset in a single folder"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def run(self):
        figures = self.dataset.base_folder.joinpath('figures')
        figures.mkdir(exist_ok=True)

        for figure in self.dataset.base_folder.glob('**/model-*/figures/*'):
            destination = figures.joinpath(
                f'{figure.parent.parts[-2]}-{figure.name}'
            )
            shutil.copy(figure, destination)


class DeleteAllModelFigures(luigi.Task):
    """Delete all figures in models folders"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    priority = 4

    def run(self):
        figures = self.dataset.base_folder.joinpath('figures')
        figures.mkdir(exist_ok=True)

        for figure in self.dataset.base_folder.glob('**/figures/*'):
            figure.unlink()

    def will_delete(self):
        """Returns the paths of the files that will be deleted"""

        return iter(self.dataset.base_folder.glob('**/figures/*'))


class DeleteAllModelAnalysis(luigi.Task):
    """Delete all diversity, listening graph ... but not recommendations and
    trained models"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def run(self):

        for file in self.dataset.base_folder.glob('**/*users_diversities*.csv'):
            file.unlink()

        for file in self.dataset.base_folder.glob('**/*-graph.pk'):
            file.unlink()

    def will_delete(self):
        """Returns the paths of the files that will be deleted"""

        files = list(self.dataset.base_folder.glob(
            '**/*users_diversities*.csv'))
        files += list(self.dataset.base_folder.glob('**/*-graph.pk'))

        return iter(files)
