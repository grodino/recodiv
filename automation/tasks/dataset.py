import json
import time
from pathlib import Path
from abc import ABC, abstractmethod

import luigi
import numpy as np
import pandas as pd

from recodiv.utils import dataset_info
from recodiv.utils import plot_histogram
from recodiv.utils import generate_graph
from recodiv.triversity.graph import IndividualHerfindahlDiversities

# Path to generated folder
GENERATED = Path('generated/')


################################################################################
# DATASETS DECLARATION                                                         #
################################################################################
class Dataset(ABC):
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

    @abstractmethod
    def import_data():
        raise NotImplementedError()


class MsdDataset(Dataset):
    """The Million Songs Dataset class"""

    IMPORT_FOLDER = 'data/million_songs_dataset/'
    NAME = 'MSD-confidence-corrected'

    def __init__(self,
                 name,
                 *args,
                 n_users=0,
                 min_user_volume=10,
                 min_item_volume=10,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.NAME = name
        self.base_folder = GENERATED.joinpath(f'dataset-{self.NAME}')
        self.data_folder = self.base_folder.joinpath('data/')
        self.n_users = int(n_users)
        self.min_item_volume = min_item_volume

        self.user_item = None
        self.item_tag = None

    def import_data(self):
        """Randomly select self.n_users (all if n_users == 0) and import all
           their listenings

        Only listened songs are imported, only tags that are related to listened
        songs are imported
        """

        print('Importing dataset')
        t = time.perf_counter()

        print('Reading user->item links file')
        user_item = pd.read_csv(
            Path(self.IMPORT_FOLDER).joinpath('msd_users.txt'),
            sep=' ',
            names=['node1_level', 'user', 'node2_level', 'item', 'rating'],
            dtype={
                'node1_level': np.int,
                'node2_level': np.int,
                'user': np.str,
                'item': np.str,
                'rating': np.float
            },
            engine='c'
        )[['user', 'item', 'rating']]

        if self.min_item_volume > 0:
            print('removing invalid items')
            # For each item, count the number of users who listen to it
            item_volume = user_item[['item', 'user']] \
                .groupby(by='item') \
                .count() \
                .rename(columns={'user': 'volume'}) \
                .reset_index()

            # Detect the items that don't have a high enought volume
            invalid_items = item_volume.loc[
                item_volume.volume < self.min_item_volume
            ]['item']

            # And delete them
            user_item = user_item \
                .set_index('item') \
                .drop(invalid_items) \
                .reset_index()

            print('Reading item->tag links file')
            item_tag = pd.read_csv(
                Path(self.IMPORT_FOLDER).joinpath('msd_tags.txt'),
                sep=' ',
                names=['node1_level', 'item', 'node2_level', 'tag', 'weight'],
                dtype={
                    'node1_level': np.int,
                    'node2_level': np.int,
                    'item': np.str,
                    'tag': np.str,
                    'weight': np.int
                },
                engine='c'
            )[['item', 'tag', 'weight']]

        # Select a portion of the dataset
        if self.n_users > 0:
            print(f'Sampling {self.n_users} users')
            rng = np.random.default_rng()

            users = user_item['user'].unique()
            selected_users = rng.choice(users, self.n_users, replace=False)

            user_item.set_index('user', inplace=True)
            user_item = user_item.loc[selected_users]
            user_item.reset_index(inplace=True)

            # Only keep songs that are listened to
            # Too slow when importing the whole dataset
            print(f'Removing songs not listened to')
            item_tag.set_index('item', inplace=True)
            item_tag = item_tag \
                .loc[user_item['item']] \
                .reset_index() \
                .drop_duplicates()

        print(
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
