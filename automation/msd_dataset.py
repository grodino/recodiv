import json
import pickle
from pathlib import Path

import luigi
from luigi.format import Nop
import binpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from recodiv.utils import dataset_info
from recodiv.model import train_model
from recodiv.model import import_and_split
from recodiv.model import generate_recommendations
from recodiv.triversity.graph import IndividualHerfindahlDiversities


# Path to generated folder
GENERATED = Path('generated/')

################################################################################
# DATASET ANALYSIS                                                             #
################################################################################
class BuildDatasetGraph(luigi.Task):
    """Build users-songs-tags graph"""
    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')

        return luigi.LocalTarget(
            dataset.joinpath('graph.pk'),
            format=luigi.format.Nop
        )

    def run(self):
        self.output().makedirs()        
        graph = IndividualHerfindahlDiversities.from_folder(
            self.dataset_folder,
            n_sets=4,
            n_entries=[0, 1_000_000]
        )
        graph.persist(self.output().path)

        del graph


class DatasetInfo(luigi.Task):
    """Save information on dataset (number of links, users ...)"""

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')
        return luigi.LocalTarget(dataset.joinpath('info.json'))

    def requires(self):
        return BuildDatasetGraph(
            dataset_name=self.dataset_name, dataset_folder=self.dataset_folder
        )
        

    def run(self):
        graph_file = self.input()
        graph = IndividualHerfindahlDiversities.recall(graph_file.path)

        with self.output().open('w') as file:
            json.dump(dataset_info(graph), file, indent=4)

        del graph


class ComputeUsersDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')
        return luigi.LocalTarget(dataset.joinpath('users_diversities.csv'))

    def requires(self):
        return BuildDatasetGraph(
            dataset_name=self.dataset_name, dataset_folder=self.dataset_folder
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

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')
        figures = dataset.joinpath(f'figures')
        return luigi.LocalTarget(figures.joinpath('user_diversity_histogram.png'))

    def requires(self):
        return ComputeUsersDiversities(
            dataset_name=self.dataset_name, dataset_folder=self.dataset_folder
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

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')
        return luigi.LocalTarget(dataset.joinpath('tags_diversities.csv'))

    def requires(self):
        return BuildDatasetGraph(
            dataset_name=self.dataset_name, dataset_folder=self.dataset_folder
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

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.dataset_name}')
        figures = dataset.joinpath(f'figures')
        return luigi.LocalTarget(figures.joinpath('tag_diversity_histogram.png'))

    def requires(self):
        return ComputeTagsDiversities(
            dataset_name=self.dataset_name, dataset_folder=self.dataset_folder
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

    name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    data_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
    )
    test_fraction = luigi.parameter.FloatParameter(
        default=.1, description='Proportion of test/train data (n_test = test_fraction * n_total)'
    )

    def output(self):
        dataset = GENERATED.joinpath(f'dataset-{self.name}/')

        return {
            'train': luigi.LocalTarget(
                dataset.joinpath(f'train-{int((1 - self.test_fraction) * 100)}.parquet'), 
                format=Nop
            ),
            'test': luigi.LocalTarget(
                dataset.joinpath(f'test-{int(self.test_fraction * 100)}.parquet'), 
                format=Nop
            )
        }
    
    def run(self):
        for out in self.output().values():
            out.makedirs()

        train, test = import_and_split(self.data_folder, self.test_fraction)

        train.to_parquet(self.output()['train'].path)
        test.to_parquet(self.output()['test'].path)


class TrainModel(luigi.Task):
    """Train a given model and save it"""
    
    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
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

    def output(self):
        model = GENERATED.joinpath(f'model-{self.n_iterations}it-{self.n_factors}f-{str(self.regularization).replace(".", "_")}reg/')
        return {
            'model': luigi.LocalTarget(
                model.joinpath('model.bpk')
            ),
            'training_metrics': luigi.LocalTarget(
                model.joinpath(f'training-metrics.parquet')
            )
        }

    def requires(self):
        return GenerateTrainTest(
            name=self.dataset_name, 
            data_folder=self.dataset_folder, 
            test_fraction=self.test_fraction
        )
    
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
    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
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
            'data': GenerateTrainTest(
                name=self.dataset_name, 
                data_folder=self.dataset_folder, 
                test_fraction=self.model_test_fraction
            ),
            'model': TrainModel(
                dataset_name=self.dataset_name,
                dataset_folder=self.dataset_folder,
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
            model.joinpath(f'recommendations-{self.n_recommendations}.parquet')
        )
    
    def run(self):
        self.output().makedirs()
        
        train = pd.read_parquet(self.input()['data']['train'].path)
        test = pd.read_parquet(self.input()['data']['test'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_recommendations(
            model, 
            train, 
            test,
            n_recommendations=self.n_recommendations,
            mode='all'
        ).to_parquet(self.output().path)

################################################################################
# RECOMMENDATIONS ANALYSIS                                                     #
################################################################################
class BuildRecommendationGraph(luigi.Task):
    """Build the user-song-tag graph for the recommendations"""

    dataset_name = luigi.parameter.Parameter(
        description='Name of the imported dataset'
    )
    dataset_folder = luigi.parameter.Parameter(
        description='Path to the dataset folder'
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
            'dataset': GenerateTrainTest(
                name=self.dataset_name,
                data_folder=self.dataset_folder,
                test_fraction=0
            ),
            'model': TrainModel(
                dataset_name=self.dataset_name,
                dataset_folder=self.dataset_folder,
                n_iterations=self.model_n_iterations, 
                n_factors=self.model_n_factors, 
                regularization=self.model_regularization,
                test_fraction=self.model_test_fraction,
                evaluate_iterations=False
            ),
            'recommendations': GenerateRecommendations(
                dataset_name=self.dataset_name,
                dataset_folder=self.dataset_folder,
                model_n_iterations=self.model_n_iterations,
                model_n_factors=self.model_n_factors,
                model_regularization=self.model_regularization,
                model_test_fraction=self.model_test_fraction,
                n_recommendations=self.n_recommendations
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}-graph.pk'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        # graph = IndividualHerfindahlDiversities.from_folder(
        #     self.dataset_folder,
        #     n_sets=4,
        #     n_entries=[0, 1_000_000]
        # )
        # graph.persist(self.output().path)
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
