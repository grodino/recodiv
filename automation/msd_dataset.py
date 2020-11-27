import json
import pickle
from pathlib import Path
from functools import cached_property

import luigi
from luigi.format import Nop
import binpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from recodiv.utils import dataset_info
from recodiv.utils import generate_graph
from recodiv.model import train_model
from recodiv.model import split_dataset
from recodiv.model import generate_recommendations
from recodiv.triversity.graph import IndividualHerfindahlDiversities


# Path to generated folder
GENERATED = Path('generated/')


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
    NAME = 'MSD'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_folder = GENERATED.joinpath(f'dataset-{self.NAME}')
        self.data_folder = self.base_folder.joinpath('data/')

        self.user_item = pd.read_csv(
            Path(self.IMPORT_FOLDER).joinpath('msd_users.txt'),
            sep=' ',
            names=['node1_level', 'user', 'node2_level', 'item', 'rating'],
            dtype={
                'node1_level': np.int8,
                'node2_level': np.int8,
                'user': np.str,
                'item': np.str,
                'rating': np.int32
            },
            nrows=1_000_000,
            engine='c'
        )[['user', 'item', 'rating']]

        self.item_tag = pd.read_csv(
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
        
        self.dataset.user_item.to_csv(self.output()['user_item'].path, index=False)
        self.dataset.item_tag.to_csv(self.output()['item_tag'].path, index=False)
        

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

        del graph


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


class ComputeUsersDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('users_diversities.csv')
        )

    def requires(self):
        return BuildDatasetGraph(
            dataset=self.dataset
        )

    def run(self):
        graph_file = self.input()
        graph = IndividualHerfindahlDiversities.recall(graph_file.path)

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2))

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

    def output(self):
        figures = self.dataset.data_folder.joinpath(f'figures')
        return luigi.LocalTarget(figures.joinpath('user_diversity_histogram.png'))

    def requires(self):
        return ComputeUsersDiversities(
            dataset=self.dataset
        )
    
    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input().path)

        pl.hist(diversities['diversity'].where(lambda x: x < 40), bins=100)
        pl.xlabel('Diversity index')
        pl.ylabel('User count')
        pl.title('Histogram of user diversity index')
        pl.savefig(self.output().path, format='png', dpi=300)


class ComputeTagsDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        return luigi.LocalTarget(
            self.dataset.data_folder.joinpath('tags_diversities.csv')
        )

    def requires(self):
        return BuildDatasetGraph(
            dataset=self.dataset
        )

    def run(self):
        graph_file = self.input()
        graph = IndividualHerfindahlDiversities.recall(graph_file.path)

        graph.normalise_all()
        diversities = graph.diversities((2, 1, 0))

        pd.DataFrame({
            'tag': list(diversities.keys()), 
            'diversity': list(diversities.values())
        }).to_csv(self.output().path, index=False)

        del graph


class PlotTagsDiversitiesHistogram(luigi.Task):
    """Plot the histogram of user diversity"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    def output(self):
        figures = self.dataset.data_folder.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath('tag_diversity_histogram.png'))

    def requires(self):
        return ComputeTagsDiversities(
            dataset=self.dataset
        )
    
    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input().path)

        print('mean tag diversity', diversities['diversity'].mean())
        print('min tag diversity', diversities['diversity'].min())
        print('max tag diversity', diversities['diversity'].max())

        pl.hist(diversities['diversity'].where(lambda x: x < 1e7), bins=100)
        pl.xlabel('Diversity index')
        pl.ylabel('Tag count')
        pl.title('Histogram of tag diversity index')
        pl.savefig(self.output().path, format='png', dpi=300)

################################################################################
# MODEL TRAINING/EVALUATION                                                    #
################################################################################
class GenerateTrainTest(luigi.Task):
    """Import a dataset (with adequate format) and generate train/test data"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    test_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = test_fraction * n_total)'
    )

    def requires(self):
        return ImportDataset(dataset=self.dataset)

    def output(self):
        return {
            'train': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(f'train-{int((1 - self.test_fraction) * 100)}.parquet'), 
                format=Nop
            ),
            'test': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(f'test-{int(self.test_fraction * 100)}.parquet'), 
                format=Nop
            )
        }
    
    def run(self):
        for out in self.output().values():
            out.makedirs()

        user_item = pd.read_csv(self.input()['user_item'].path)
        train, test = split_dataset(user_item, self.test_fraction)

        train.to_parquet(self.output()['train'].path)
        test.to_parquet(self.output()['test'].path)


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
    # TODO: also implement crossfold techniques
    test_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = test_fraction * n_total)'
    )

    evaluate_iterations = luigi.parameter.BoolParameter(
        default=False, description='Create recommendations and evaluate metrics at each training iteration'
    )
    iteration_metrics = luigi.parameter.TupleParameter(
        default=('precision', 'ndcg'), description='Metrics to compute at each training iteration'
    )
    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        return GenerateTrainTest(
            dataset=self.dataset, 
            test_fraction=self.test_fraction
        )

    def output(self):
        model = self.dataset.base_folder.joinpath(
            f'model-{self.n_iterations}it-{self.n_factors}f-{str(self.regularization).replace(".", "_")}reg/'
        )
        
        return {
            'model': luigi.LocalTarget(
                model.joinpath('model.bpk')
            ),
            'training_metrics': luigi.LocalTarget(
                model.joinpath(f'training-metrics.parquet')
            )
        }
    
    def run(self):
        for out in self.output().values():
            out.makedirs()

        train_file, test_file = self.input()

        train = pd.read_parquet(self.input()['train'].path)
        test = pd.read_parquet(self.input()['test'].path)

        model, metrics = train_model(
            train, 
            test,
            n_factors=self.n_factors, 
            n_iterations=self.n_iterations, 
            regularization=self.regularization, 
            evaluate_iterations=self.evaluate_iterations,
            iteration_metrics=self.iteration_metrics,
            n_recommendations=self.n_recommendations
        )

        binpickle.dump(model, self.output()['model'].path)
        metrics.to_parquet(self.output()['training_metrics'].path)
        

class GenerateRecommendations(luigi.Task):
    """Generate recommendations for all users with a given model"""

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
    # TODO: also implement crossfold techniques
    model_test_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = test_fraction * n_total)'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return {
            'data': ImportDataset(dataset=self.dataset),
            'model': TrainModel(
                dataset=self.dataset,
                n_iterations=self.model_n_iterations, 
                n_factors=self.model_n_factors, 
                regularization=self.model_regularization,
                test_fraction=self.model_test_fraction,
                evaluate_iterations=False
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}.csv')
        )
    
    def run(self):
        self.output().makedirs()
        
        user_item = pd.read_csv(self.input()['data']['user_item'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_recommendations(
            model, 
            user_item,
            n_recommendations=self.n_recommendations,
        ).to_csv(self.output().path, index=False)


################################################################################
# RECOMMENDATIONS ANALYSIS                                                     #
################################################################################
class BuildRecommendationGraph(luigi.Task):
    """Build the user-song-tag graph for the recommendations"""

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
    # TODO: also implement crossfold techniques
    model_test_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = test_fraction * n_total)'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return GenerateRecommendations(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_test_fraction=self.model_test_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        model = Path(self.input()['model']['model'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        graph = generate_graph()
        graph.persist(self.output().path)

        print('YAYAYAYAYAYAYAYYA')


class PlotDiversityIncrease(luigi.Task):
    def output(self):
        return luigi.LocalTarget('generated/figures/diversity_increase_histogram.png')

    def requires(self):
        return [
            ComputeUsersDiversities(),
            ComputeRecommendationsDiversities()
        ]

    def run(self):
        # Recall recommendations diversities
        recommendations_diversisties_file = self.input()[1]
        reco_diversities = {}

        with recommendations_diversisties_file.open('r') as file:
            for line in file.readlines():
                user_id, diversity = line.split(',')
                reco_diversities[user_id] = float(diversity)

        # Recall users diversities and compute the difference
        users_diversisties_file = self.input()[0]
        delta_diversities = {}

        with users_diversisties_file.open('r') as file:
            for line in file.readlines():
                user_id, diversity = line.split(',')
                delta_diversities[user_id] = reco_diversities[user_id] - float(diversity)

        deltas = list(delta_diversities.values())
        deltas.remove(min(deltas))
        deltas.remove(min(deltas))
        deltas.remove(min(deltas))
        deltas.remove(max(deltas))

        pl.hist(deltas, bins=150)
        # pl.plot(deltas)
        self.output().makedirs()
        pl.savefig(self.output().path)
        pl.show()
