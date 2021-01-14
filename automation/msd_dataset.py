import json
import time
import shutil
import pickle
import logging
from pathlib import Path

import luigi
from luigi.format import Nop
import binpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from recodiv.utils import dataset_info
from recodiv.utils import plot_histogram
from recodiv.utils import generate_graph
from recodiv.model import train_model
from recodiv.model import split_dataset
from recodiv.model import generate_predictions
from recodiv.model import generate_recommendations
from recodiv.model import evaluate_model_predictions
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

    def __init__(self, *args, n_users=0,  **kwargs):
        super().__init__(*args, **kwargs)

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
            item_tag = item_tag.loc[user_item['item']].reset_index().drop_duplicates()

        logger.debug(f'Finished importing dataset in {time.perf_counter() - t}')

        self.user_item = user_item
        self.item_tag = item_tag

        del user_item
        del item_tag

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

        fig, ax = plot_histogram(diversities['diversity'].to_numpy(), min_quantile=0)
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

        del graph, diversities


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

        fig, ax = plot_histogram(diversities['diversity'].to_numpy(), min_quantile=0)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('Tag count')
        ax.set_title('Histogram of tag diversity index')
        fig.savefig(self.output().path, format='png', dpi=300)
        
        del fig, ax, diversities


################################################################################
# MODEL TRAINING/EVALUATION                                                    #
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
                self.dataset.data_folder.joinpath(f'train-{int((1 - self.user_fraction) * 100)}.csv'), 
                format=Nop
            ),
            'test': luigi.LocalTarget(
                self.dataset.data_folder.joinpath(f'test-{int(self.user_fraction * 100)}.csv'), 
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

# TODO : do all the dataset plots on the testing set

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
    
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
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
            user_fraction=self.user_fraction
        )

    def output(self):
        model = self.dataset.base_folder.joinpath(
            f'model-{self.n_iterations}it-{self.n_factors}f-{str(self.regularization).replace(".", "_")}reg/'
        )

        out = {'model': luigi.LocalTarget(model.joinpath('model.bpk'))}
        
        if self.evaluate_iterations == True:
            out['training_metrics'] = luigi.LocalTarget(
                model.joinpath(f'training-metrics.csv')
            )

        return out
    
    def run(self):
        for out in self.output().values():
            out.makedirs()

        train_file, test_file = self.input()

        train = pd.read_csv(self.input()['train'].path)
        test = pd.read_csv(self.input()['test'].path)

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

        if self.evaluate_iterations == True:
            metrics.to_csv(self.output()['training_metrics'].path)

        del train, test, model, metrics

# TODO: create a ModelInfo task

class GeneratePredictions(luigi.Task):
    """Compute the predicted rating values for the test set"""

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
                evaluate_iterations=False
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'predictions-{self.n_recommendations}.csv')
        )
    
    def run(self):
        self.output().makedirs()
        
        user_item = pd.read_csv(self.input()['data']['test'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_predictions(
            model, 
            user_item,
        ).to_csv(self.output().path, index=False)

        del user_item, model


class ComputeUserRMSE(luigi.Task):
    """Compute the user prediction RMSE for each user in the test set"""

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
        return GeneratePredictions(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )
    
    def output(self):
        model = Path(self.input().path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'user_rmse.csv'),
            format=Nop
        )
    
    def run(self):
        self.output().makedirs()

        predictions = pd.read_csv(self.input().path)
        evaluate_model_predictions(predictions).reset_index() \
            .to_csv(self.output().path, index=False)

        del predictions


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
                evaluate_iterations=False
            ),
            'predictions': GeneratePredictions(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}.csv')
        )
    
    def run(self):
        self.output().makedirs()
        
        ratings = pd.read_csv(self.input()['data']['train'].path)
        predictions = pd.read_csv(self.input()['predictions'].path)

        generate_recommendations(
            ratings,
            predictions,
            n_recommendations=self.n_recommendations,
        ).to_csv(self.output().path, index=False)

        del ratings, predictions


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
                n_recommendations=self.n_recommendations
            ),
            'user_rmse': ComputeUserRMSE(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'dataset': GenerateTrainTest(
                dataset=self.dataset,
                user_fraction=self.model_user_fraction
            )
        }
    
    def output(self):
        model = Path(self.input()['recommendations'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'{self.n_recommendations}-recommendations_model_eval.json'),
            format=Nop
        )
    
    def run(self):
        self.output().makedirs()

        recommendations = pd.read_csv(self.input()['recommendations'].path)
        user_rmse = pd.read_csv(self.input()['user_rmse'].path)
        test = pd.read_csv(self.input()['dataset']['test'].path)

        # NOTE : Issue appears when the two following condition is met :
        #   - a user had only one listening and it is put in the test set
        #
        # Then, the model will not know the user, thus will not be able to
        # recommend songs to this user. Therefore, to avoid issues, we simply
        # discard users with no recommendations from the test set
        recommendations.set_index('user', inplace=True)
        test.set_index('user', inplace=True)
        
        missing = test.index.difference(recommendations.index)
        common = test.index.intersection(recommendations.index).unique()
        print(f'In test set but not recommended : {missing}')

        recommendations = recommendations.loc[common].reset_index()
        test.reset_index(inplace=True)

        metrics_names = ['ndcg', 'precision']
        metrics = evaluate_model_recommendations(
            recommendations,
            test, 
            metrics_names
        )[metrics_names].mean()

        metrics['rmse'] = user_rmse['rmse'].mean()
        
        metrics.to_json(
            self.output().path,
            orient='index',
            indent=4
        )

        del recommendations, test, missing, common, metrics


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

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        grid = np.meshgrid(self.model_n_factors_values, self.model_regularization_values)
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
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        
        return luigi.LocalTarget(
            aggregated.joinpath(f'{self.model_n_factors_values}factors_{self.model_regularization_values}reg_{self.n_recommendations}reco_model_eval.json'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()
        metrics = pd.DataFrame()

        for (n_factors, regularization), metrics_file in self.input().items():
            metric = pd.read_json(metrics_file.path, orient='index').transpose()
            metric['n_factors'] = n_factors
            metric['regularization'] = regularization

            metrics = pd.concat((metrics, metric))

        metrics.set_index(['n_factors', 'regularization'], inplace=True)
        optimal = {}

        opt_n_factors, opt_regularization = metrics.index[metrics['ndcg'].argmax()]
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

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        grid = np.meshgrid(self.model_n_factors_values, self.model_regularization_values)
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
        aggregated = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        
        return luigi.LocalTarget(
            aggregated.joinpath(f'{self.model_n_factors_values}factors_{self.model_regularization_values}reg_{self.n_recommendations}reco_model_eval.png'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()
        metrics = pd.DataFrame()

        for (n_factors, regularization), metrics_file in self.input().items():
            metric = pd.read_json(metrics_file.path, orient='index').transpose()
            metric['n_factors'] = n_factors
            metric['regularization'] = regularization

            metrics = pd.concat((metrics, metric))

        metrics_matrix = metrics.pivot(index='n_factors', columns='regularization')['rmse']
        
        metrics_matrix_n = metrics_matrix.to_numpy()
        opt_n_factors, opt_regularization = np.unravel_index(
            metrics_matrix_n.flatten().argmin(),
            metrics_matrix_n.shape
        )

        fig, ax = pl.subplots()

        img = ax.imshow(metrics_matrix_n)
        fig.colorbar(img)

        ax.set_xticks([0,] + list(range(len(metrics_matrix.columns))))
        ax.set_xticklabels(['A', ] + list(metrics_matrix.columns))

        ax.set_yticks([0,] + list(range(len(metrics_matrix.index))))
        ax.set_yticklabels(['A', ] + list(metrics_matrix.index))

        ax.text(
            opt_regularization,
            opt_n_factors, 
            'MIN', 
            ha="center", 
            va="center", 
            color="w"
        )

        ax.set_ylabel('Number of latent factors')
        ax.set_xlabel('Regularization coefficient')
        ax.set_title('Model performance evaluation with RMSE')

        fig.savefig(self.output().path, format='png', dpi=300)
        
        del fig, ax, metrics, metrics_matrix


################################################################################
# RECOMMENDATIONS ANALYSIS                                                     #
################################################################################
# TODO: do not use 1/rank but the score as rating value
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
    
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of users whose items are selected for test data sampling'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
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
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        model = Path(self.input()['recommendations'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path)

        user_item = recommendations[['user', 'item', 'rank']]
        user_item['rating'] = 1 / user_item['rank']

        graph = generate_graph(user_item,item_tag)
        graph.persist(self.output().path)

        del graph, user_item


class ComputeRecommendationUsersDiversities(luigi.Task):
    """Compute the diversity of the songs recommended to users"""

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
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}-users_diversities.csv')
        )

    def run(self):
        graph = IndividualHerfindahlDiversities.recall(self.input().path)

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2))

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
        return ComputeRecommendationUsersDiversities(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )
    
    def output(self):
        figures = Path(self.input().path).parent.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}-recommendation_user_diversity_histogram.png'
        ))
  
    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(diversities['diversity'].to_numpy(), min_quantile=0)
        
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of recommendations diversity index')
        fig.savefig(self.output().path, format='png', dpi=300)
        
        del fig, ax, diversities


# TODO: think about the relative importance of the recommendations added to the
# graph: what weight should we give them ? Idea: recontruct the graph of
# listened songs with the scalar product of user->item factors rather than
# number of listenings
class BuildRecommendationsWithListeningsGraph(luigi.Task):
    """Consider the recommendations as all listened by the users and compute the
    corresponding diversity"""

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
            'graph': BuildDatasetGraph(
                dataset=self.dataset
            ),
            'dataset': ImportDataset(dataset=self.dataset),
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
            model.joinpath(f'listenings-recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        graph = IndividualHerfindahlDiversities.recall(
            self.input()['graph'].path
        )
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path)

        user_item = recommendations[['user', 'item', 'rank']]
        user_item['rating'] = 1 / user_item['rank']

        graph = generate_graph(user_item, item_tag, graph=graph)
        graph.persist(self.output().path)

        del graph, user_item, item_tag


class ComputeRecommendationWithListeningsUsersDiversities(luigi.Task):
    """Compute the diversity of the users who were recommended, assuming they
       listened to all recommendations"""

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
        return luigi.LocalTarget(
            model.joinpath(f'listenings-recommendations-{self.n_recommendations}-users_diversities.csv')
        )

    def run(self):
        graph = IndividualHerfindahlDiversities.recall(
            self.input().path
        )

        graph.normalise_all()
        diversities = graph.diversities((0, 1, 2))
        diversities = pd.DataFrame({
            'user': list(diversities.keys()), 
            'diversity': list(diversities.values())
        })

        diversities.to_csv(self.output().path, index=False)

        del graph, diversities


class ComputeRecommendationWithListeningsUsersDiversityIncrease(luigi.Task):
    """Compute the increase of diversity for the users who were recommended
       songs, assuming they listened to all the recommendations"""

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
            'with_recommendations': ComputeRecommendationWithListeningsUsersDiversities(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'original': ComputeUsersDiversities(dataset=self.dataset)
        }
        
    def output(self):
        model = Path(self.input()['with_recommendations'].path).parent
        return luigi.LocalTarget(
            model.joinpath(f'listenings-recommendations-{self.n_recommendations}-users_diversities_increase.csv')
        )

    def run(self):
        with_recommendations = pd.read_csv(self.input()['with_recommendations'].path) \
            .set_index('user')
        original = pd.read_csv(self.input()['original'].path) \
            .set_index('user')

        deltas = (original['diversity'] - with_recommendations['diversity']) \
            .dropna() \
            .reset_index()

        deltas.to_csv(self.output().path, index=False)

        del original, deltas


class PlotDiversitiesIncreaseHistogram(luigi.Task):
    """Plot the histogram of recommendations diversity for each user"""

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
        return ComputeRecommendationWithListeningsUsersDiversityIncrease(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations=self.n_recommendations
        )

    def output(self):
        figures = Path(self.input().path).parent.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}-recommendations_diversity_increase_histogram.png'
        ))

    
    def run(self):
        self.output().makedirs()
        deltas = pd.read_csv(self.input().path)

        fig, ax = plot_histogram(deltas['diversity'].to_numpy(), min_quantile=0, max_quantile=1, log=True)
        ax.set_xlabel('Diversity index')
        ax.set_ylabel('User count')
        ax.set_title('Histogram of diversity increase due to recommendations')
        fig.savefig(self.output().path, format='png', dpi=300)
        
        del fig, ax, deltas


class PlotUserDiversityIncreaseVsUserDiversity(luigi.Task):
    """Plot the user diversity increase with respect to the user diversity
       before recommendations"""

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
            'user_diversity': ComputeUsersDiversities(
                dataset=self.dataset
            ),
            'diversity_increase': ComputeRecommendationWithListeningsUsersDiversityIncrease(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
        }

    def output(self):
        figures = Path(self.input()['diversity_increase'].path).parent.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'{self.n_recommendations}-recommendations_diversity_increase_vs_original_diversity.png'
        ))

    def run(self):
        self.output().makedirs()
        diversities = pd.read_csv(self.input()['user_diversity'].path)
        increase = pd.read_csv(self.input()['diversity_increase'].path).rename(columns={'diversity': 'increase'})

        # inner join, only keep users for whom we calculated a diversity increase value
        merged = increase.merge(diversities, on='user')
        
        merged.plot.scatter(x='diversity', y='increase')
        pl.title('User diversity before and after recomendations')

        pl.savefig(self.output().path, format='png', dpi=300)

        del diversities, increase, merged


# TODO : correlation between diversity increase and user diversity
# TODO : correlation between recommendation diversity and user diversity (vs volume/latent factors)
# TODO : correlation between user RMSE and user recommendation diversity

################################################################################
# SPECIFIC EXPERIMENTS                                                         #
################################################################################
class PlotDiversityVsLatentFactors(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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
            tasks[(n_factors, 'diversities')] = ComputeRecommendationUsersDiversities(
                dataset=self.dataset,
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
        figures = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_vs_{self.n_factors_values}factors.png'
        ))

    def run(self):
        self.output().makedirs()

        mean_diversities = []
        metrics = pd.DataFrame()
        factors = []

        for key, value in self.input().items():
            if key[1] == 'diversities':
                diversities = pd.read_csv(value.path)

                factors.append(key[0])
                mean_diversities.append(diversities['diversity'].mean())
            
            elif key[1] == 'metrics':
                metric = pd.read_json(value.path, orient='index').transpose()
                metric['n_factors'] = key[0]
                metrics = pd.concat((metrics, metric))
        
        metrics.set_index('n_factors', inplace=True)
        # metrics = metrics / metrics.loc[metrics.index[0]]
        metrics = metrics - metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()        

        # Add plots
        div_line = ax1.plot(factors, mean_diversities, color='green', label='diversity')
        ax1.set_xlabel('number of factors')
        ax1.set_ylabel('mean diversity')

        ax2 = ax1.twinx()
        ax2.set_ylabel('metrics')
        # metrics_lines = metrics.plot(ax=ax2, legend=False, logy=True).get_lines()
        metrics_lines = metrics.plot(ax=ax2, legend=False, logy=False).get_lines()
        
        # Obscure trick to have only one legend
        lines = [*div_line, ]

        for line in metrics_lines:
            lines.append(line)

        labels = ['diversity', ] + list(metrics.columns)
        ax1.legend(lines, labels, loc='center right')

        pl.title('User diversity of recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()

        del metrics, mean_diversities, fig, ax1, ax2


class PlotDiversityIncreaseVsLatentFactors(luigi.Task):
    """Plot the mean user diversity increase after recommendation as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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
        figures = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_increase_vs_{self.n_factors_values}factors.png'
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
        metrics = metrics / metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()        

        # Add plots
        div_line = ax1.plot(factors, mean_deltas, color='green', label='diversity')
        ax1.set_xlabel('number of factors')
        ax1.set_ylabel('mean diversity increase')

        ax2 = ax1.twinx()
        ax2.set_ylabel('metrics')
        metrics_lines = metrics.plot(ax=ax2, legend=False, logy=True).get_lines()
        
        # Obscure trick to have only one legend
        lines = [*div_line, ]

        for line in metrics_lines:
            lines.append(line)

        labels = ['diversity', ] + list(metrics.columns)
        ax1.legend(lines, labels, loc='center right')

        pl.title('User diversity increase after recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


class PlotDiversityVsRegularization(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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
            tasks[(regularization, 'diversities')] = ComputeRecommendationUsersDiversities(
                dataset=self.dataset,
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
        figures = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_vs_{self.model_regularization_values}reg.png'
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
        div_line = ax1.semilogx(regularization, mean_diversities, color='green', label='diversity')
        ax1.set_xlabel('Regularization coefficient')
        ax1.set_ylabel('mean diversity')

        ax2 = ax1.twinx()
        ax2.set_ylabel('metrics')
        metrics_lines = metrics.plot(ax=ax2, legend=False, logy=True).get_lines()
        
        # Obscure trick to have only one legend
        lines = [*div_line, ]

        for line in metrics_lines:
            lines.append(line)

        labels = ['diversity', ] + list(metrics.columns)
        ax1.legend(lines, labels, loc='center right')

        pl.title('User diversity of recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


class PlotDiversityIncreaseVsRegularization(luigi.Task):
    """Plot the mean user diversity of the recommendations as a function of the
       number of latent factors"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
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
        figures = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_increase_vs_{self.model_regularization_values}reg.png'
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
        div_line = ax1.semilogx(regularization, mean_deltas, color='green', label='diversity')
        ax1.set_xlabel('Regularization coefficient')
        ax1.set_ylabel('mean diversity increase')

        ax2 = ax1.twinx()
        ax2.set_ylabel('metrics')
        metrics_lines = metrics.plot(ax=ax2, legend=False, logy=True).get_lines()
        
        # Obscure trick to have only one legend
        lines = [*div_line, ]

        for line in metrics_lines:
            lines.append(line)

        labels = ['diversity', ] + list(metrics.columns)
        ax1.legend(lines, labels, loc='center right')

        pl.title('User diversity increase after recommendations')
        fig.tight_layout()
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


class ComputeDiversityVsRecommendationVolume(luigi.Task):
    """Compute the user diversity of recommendations against the number of
       recommendations made"""

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
        aggregated = self.dataset.base_folder.joinpath('aggregated')
        return luigi.LocalTarget(aggregated.joinpath(
            f'recommendations_diversity_vs_{self.n_recommendations_values}reco.csv'
        ))
    
    def run(self):
        self.output().makedirs()
        
        item_tag = pd.read_csv(self.input()['dataset']['item_tag'].path)
        recommendations = pd.read_csv(self.input()['recommendations'].path) \
            .rename(columns={'score': 'rating'})

        mean_diversities = []

        # Not very otpimized, but constructing the graph incrementally would
        # require to modify triversity
        # TODO: optimize
        for n_recommendations in self.n_recommendations_values:
            user_item = recommendations[recommendations['rank'] <= n_recommendations]
            
            graph = generate_graph(user_item,item_tag)
            graph.normalise_all()
            diversities = graph.diversities((0, 1, 2))
            
            mean_diversities.append(
                sum(diversities.values()) / len(diversities)
            )

            del graph, diversities
        
        pd.DataFrame({
            'n_recommendations': self.n_recommendations_values,
            'diversity': mean_diversities
        }).to_csv(self.output().path, index=False)

        del item_tag, recommendations


class PlotDiversityVsRecommendationVolume(luigi.Task):
    """Plot the user diversity of recommendations against the number of
       recommendations made"""

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

    n_recommendations_values = luigi.parameter.ListParameter(
        description='List of number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        return ComputeDiversityVsRecommendationVolume(
            dataset=self.dataset,
            model_n_iterations=self.model_n_iterations,
            model_n_factors=self.model_n_factors,
            model_regularization=self.model_regularization,
            model_user_fraction=self.model_user_fraction,
            n_recommendations_values=self.n_recommendations_values
        )
    
    def output(self):
        figures = self.dataset.base_folder.joinpath('aggregated').joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_vs_{self.n_recommendations_values}reco.png'
        ))

    def run(self):
        self.output().makedirs()
        data = pd.read_csv(self.input().path)

        pl.plot(data['n_recommendations'], data['diversity'])
        pl.xlabel('Number of recommendations per user')
        pl.ylabel('Mean user diversity')
        pl.title('Evolution of diversity and recommendation volume')

        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()

        del data


################################################################################
# UTILS                                                                        #
################################################################################
class CollectAllModelFigures(luigi.Task):
    """Collect all figures related to a dataset in a single folder"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    ) 

    priority = -1

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

        for figure in self.dataset.base_folder.glob('**/model-*/figures/*'):
            figure.unlink()