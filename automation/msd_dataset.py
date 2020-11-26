import pickle
from pathlib import Path

import luigi
from luigi.format import Nop
import binpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from recodiv.utils import create_msd_graph
from recodiv.model import train_model
from recodiv.model import import_and_split
from recodiv.model import generate_recommendations
from recodiv.model import recommendations_graph


# Path to generated folder
GENERATED = Path('generated/')


class BuildUserSongGraph(luigi.Task):
    """Build users-songs-tags graph"""

    def output(self):
        return luigi.LocalTarget('generated/msd_graph', format=luigi.format.Nop)

    def run(self):
        graph, _ = create_msd_graph()
        # TODO : save dataset stats (number of users, songs, tags, links, songs not listened ...)

        self.output().makedirs()
        graph.persist(self.output().path)


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

        return [
            luigi.LocalTarget(
                dataset.joinpath(f'train-{int((1 - self.test_fraction) * 100)}.parquet'), 
                format=Nop
            ),
            luigi.LocalTarget(
                dataset.joinpath(f'test-{int(self.test_fraction * 100)}.parquet'), 
                format=Nop
            )
        ]
    
    def run(self):
        for out in self.output():
            out.makedirs()

        train, test = import_and_split(self.data_folder)
        train_file, test_file = self.output()

        train.to_parquet(train_file.path)
        test.to_parquet(test_file.path)


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
        return [
            luigi.LocalTarget(
                model.joinpath('model.bpk')
            ),
            luigi.LocalTarget(
                model.joinpath(f'training-metrics.parquet')
            )
        ]

    def requires(self):
        return [
            GenerateTrainTest(
                name=self.dataset_name, 
                data_folder=self.dataset_folder, 
                test_fraction=self.test_fraction
            )
        ]
    
    def run(self):
        for out in self.output():
            out.makedirs()

        train_file, test_file = self.input()[0]

        train = pd.read_parquet(train_file.path)
        test = pd.read_parquet(test_file.path)

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
        
        model_file, metrics_file = self.output()

        binpickle.dump(model, model_file.path)
        metrics.to_parquet(metrics_file.path)
        

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

    def output(self):
        model = Path(self.input()[1][0].path).parent
        
        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}.parquet')
        )

    def requires(self):
        return [
            GenerateTrainTest(
                name=self.dataset_name, 
                data_folder=self.dataset_folder, 
                test_fraction=self.model_test_fraction
            ),
            TrainModel(
                dataset_name=self.dataset_name,
                dataset_folder=self.dataset_folder,
                n_iterations=self.model_n_iterations, 
                n_factors=self.model_n_factors, 
                regularization=self.model_regularization,
                test_fraction=self.model_test_fraction,
                evaluate_iterations=False
            )
        ]
    
    def run(self):
        self.output().makedirs()

        (train_file, test_file), (model_file, _) = self.input()
        
        train = pd.read_parquet(train_file.path)
        test = pd.read_parquet(test_file.path)
        model = binpickle.load(model_file.path)

        generate_recommendations(
            model, 
            train, 
            test,
            n_recommendations=self.n_recommendations,
            mode='all'
        ).to_parquet(self.output().path)


class BuildCompleteGraph(luigi.Task):
    """Compute recommendations for users and append them to the users-songs-tags
    graph
    """

    def output(self):
        return luigi.LocalTarget(
            'generated/graph_with_recommendations', format=luigi.format.Nop
        )

    def requires(self):
        return [BuildUserSongGraph(), TrainCollaborativeFiltering()]

    def run(self):
        graph_file = self.input()[0]

        graph, (n_users, n_songs, n_categories) = create_msd_graph(
            recall_file=graph_file.path
        )

        with self.input()[1][0].open('rb') as file:
            model = pickle.load(file)

        graph = recommendations_graph(
            graph, model, n_users, n_songs, n_recommendations=10
        )
        graph.persist(self.output().path)


class EvaluateCFModel(luigi.Task):
    """Evaluate the Collaborative Filtering model"""

    def output(self):
        return luigi.LocalTarget('generated/metrics', format=luigi.format.Nop)

    def requires(self):
        return [
            BuildCompleteGraph(), 
            TrainCollaborativeFiltering()
        ]

    def run(self):
        graph_file = self.input()[0]

        graph, (n_users, n_songs, n_categories) = create_msd_graph(
            recall_file=graph_file.path
        )

        recommendations = []
        for user, songs in graph.graphs[0][3].items():
            sorted_songs = sorted(songs.items(), key=lambda s: s[1])
            
            recommendations.append(
                [song_id for song_id, _ in sorted_songs]
            )
        recommendations = np.array(recommendations, dtype=int)

        with self.input()[1][0].open('rb') as file:
            model = pickle.load(file)

        with self.input()[1][1].open('rb') as file:
            train_data = pickle.load(file)
            
        with self.input()[1][2].open('rb') as file:
            test_data = pickle.load(file)

        metrics = evaluate_model(
            n_users, 
            n_songs, 
            test_data,
            recommendations
        )

        with self.output().open('wb') as file:
            pickle.dump(metrics)


class ComputeRecommendationsDiversities(luigi.Task):
    """Compute the recommendations diversity from the user point of view"""

    def output(self):
        return luigi.LocalTarget('generated/recommendation_diversities.csv')

    def requires(self):
        return [BuildCompleteGraph()]

    def run(self):
        graph_file = self.input()[0]
        graph, _ = create_msd_graph(recall_file=graph_file.path)

        graph.normalise_all()
        _ = graph.diversities(
            (0, 3, 2), file_path=self.output().path
        )


class ComputeUsersDiversities(luigi.Task):
    """Compute the diversity of the songs listened by users"""

    def output(self):
        return luigi.LocalTarget('generated/user_diversities.csv')

    def requires(self):
        return [BuildUserSongGraph()]

    def run(self):
        graph_file = self.input()[0]
        graph, _ = create_msd_graph(recall_file=graph_file.path)

        graph.normalise_all()
        _ = graph.diversities(
            (0, 1, 2), file_path=self.output().path
        )


class PlotRecommendationsDiversity(luigi.Task):
    """Plot diversity of recommendations from the user point of view"""

    def output(self):
        return luigi.LocalTarget('generated/figures/recommendation_diversity_histogram.png')

    def requires(self):
        return [ComputeRecommendationsDiversities()]

    def run(self):
        recommendations_diversisties_file = self.input()[0]
        reco_diversities = {}

        with recommendations_diversisties_file.open('r') as file:
            for line in file.readlines():
                user_id, diversity = line.split(',')
                reco_diversities[user_id] = float(diversity)

        pl.hist(reco_diversities.values(), bins=150)
        self.output().makedirs()
        pl.savefig(self.output().path)
        pl.show()


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
