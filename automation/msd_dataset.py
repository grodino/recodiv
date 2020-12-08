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
from recodiv.utils import generate_graph
from recodiv.model import train_model
from recodiv.model import split_dataset
from recodiv.model import evaluate_model
from recodiv.model import generate_recommendations
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
    NAME = 'MSD'

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
                'rating': np.int32
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


# TODO : compute and plot user volume histogram


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
        pl.close()


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
        pl.close()


################################################################################
# MODEL TRAINING/EVALUATION                                                    #
################################################################################
class GenerateTrainTest(luigi.Task):
    """Import a dataset (with adequate format) and generate train/test data"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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


class TrainTestInfo(luigi.Task):
    """Compute information about the training and testings datasets (n_users ...)"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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
    user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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
        metrics.to_parquet(self.output()['training_metrics'].path)
        

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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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
            model.joinpath(f'recommendations-{self.n_recommendations}.csv')
        )
    
    def run(self):
        self.output().makedirs()
        
        user_item = pd.read_csv(self.input()['data']['test'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_recommendations(
            model, 
            user_item,
            n_recommendations=self.n_recommendations,
        ).to_csv(self.output().path, index=False)


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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
        test = pd.read_csv(self.input()['dataset']['test'].path)

        # NOTE : Issue appears when the two following conditions are met :
        #   - the train test is split by data row (ie by removings all
        #     listenings of a song by a particular user)
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
        metrics = evaluate_model(recommendations, test, metrics_names)

        metrics[metrics_names].mean().to_json(
            self.output().path,
            orient='index',
            indent=4
        )


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
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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

        del graph


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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

        del graph


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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

        mean = diversities['diversity'].mean()

        pl.hist(
            diversities['diversity'].where(
                lambda x: x < diversities['diversity'].quantile(.99)
            ),
            bins=100
        )
        pl.axvline(mean, ls='--', color='pink')
        pl.text(mean + 1, 2, f'mean: {mean:.02f}', color='pink')
        pl.xlabel('Diversity index')
        pl.ylabel('User count')
        pl.title('Histogram of recommendations diversity index')
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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

        graph = generate_graph(user_item,item_tag, graph=graph)
        graph.persist(self.output().path)

        del graph


class ComputeRecommendationWithListeningsUsersDiversitiesIncrease(luigi.Task):
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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return {
            'with_listenings_graph': BuildRecommendationsWithListeningsGraph(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            ),
            'original_diversities': ComputeUsersDiversities(
                dataset=self.dataset
            )
        }

    def output(self):
        model = Path(self.input()['with_listenings_graph'].path).parent
        return luigi.LocalTarget(
            model.joinpath(f'listenings-recommendations-{self.n_recommendations}-users_diversities_increase.csv')
        )

    def run(self):
        rl_graph = IndividualHerfindahlDiversities.recall(
            self.input()['with_listenings_graph'].path
        )

        rl_graph.normalise_all()
        rl_diversities = rl_graph.diversities((0, 1, 2))
        rl_diversities = pd.DataFrame({
            'user': list(rl_diversities.keys()), 
            'diversity': list(rl_diversities.values())
        })

        diversities = pd.read_csv(self.input()['original_diversities'].path)
        deltas = rl_diversities
        deltas['diversity'] = rl_diversities['diversity'] - diversities['diversity']

        deltas[:10_000].to_csv(self.output().path, index=False)

        del rl_graph


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return ComputeRecommendationWithListeningsUsersDiversitiesIncrease(
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
        deltas = pd.read_csv(
            self.input().path
        )
        mean = deltas['diversity'].mean()

        pl.hist(
            deltas['diversity'].where(
                lambda x: deltas['diversity'].quantile(.05) <  x
            ).where(
                lambda x: x < deltas['diversity'].quantile(.99)
            ), 
            bins=100
        )
        pl.axvline(mean, ls='--', color='pink')
        pl.text(mean + 1, 2, f'mean: {mean:.02f}', color='pink')
        pl.xlabel('Diversity index')
        pl.ylabel('User count')
        pl.title('Histogram of diversity increase due to recommendations')
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
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
        figures = self.dataset.base_folder.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_vs_{self.n_factors_values}.png'
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
        metrics = metrics - metrics.loc[metrics.index[0]]

        fig, ax1 = pl.subplots()        

        # Add plots
        div_line = ax1.plot(factors, mean_diversities, color='green', label='diversity')
        ax1.set_xlabel('number of factors')
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
    # TODO: also implement crossfold techniques
    model_user_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = user_fraction * n_total)'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate for each user at each training iteration if evaluate_iterations==True'
    )

    def requires(self):
        tasks = {}

        for n_factors in self.n_factors_values:
            tasks[n_factors] = ComputeRecommendationWithListeningsUsersDiversitiesIncrease(
                dataset=self.dataset,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=n_factors,
                model_regularization=self.model_regularization,
                model_user_fraction=self.model_user_fraction,
                n_recommendations=self.n_recommendations
            )

        return tasks

    def output(self):
        figures = self.dataset.base_folder.joinpath('figures')
        return luigi.LocalTarget(figures.joinpath(
            f'recommendations_diversity_increase_vs_{self.n_factors_values}.png'
        ))

    def run(self):
        self.output().makedirs()
        
        mean_deltas = []
        factors = []

        for n_factors, deltas_file in self.input().items():
            deltas = pd.read_csv(deltas_file.path)

            factors.append(n_factors)
            mean_deltas.append(deltas['diversity'].mean())

        pl.plot(factors, mean_deltas)
        pl.xlabel('number of factors')
        pl.ylabel('mean diversity increase')
        pl.title('User diversity increase after recommendations')
        pl.savefig(self.output().path, format='png', dpi=300)
        pl.close()


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