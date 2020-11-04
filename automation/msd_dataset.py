import pickle

import luigi
from matplotlib import pyplot as pl

from recodiv.utils import create_msd_graph
from recodiv.utils import recommendations_graph
from recodiv.utils import train_msd_collaborative_filtering


class BuildUserSongGraph(luigi.Task):
    """Build users-songs-tags graph"""

    def output(self):
        return luigi.LocalTarget('generated/msd_graph', format=luigi.format.Nop)

    def run(self):
        graph, _ = create_msd_graph()

        self.output().makedirs()
        graph.persist(self.output().path)


class TrainCollaborativeFiltering(luigi.Task):
    """Train the Collaborative Filtering for Implicit Data algorithm"""

    def output(self):
        return luigi.LocalTarget(
            'generated/msd_cf_model', format=luigi.format.Nop
        )

    def requires(self):
        return [BuildUserSongGraph()]

    def run(self):
        graph_file = self.input()[0]

        graph, (n_users, n_songs, n_categories) = create_msd_graph(
            recall_file=graph_file.path
        )
        model = train_msd_collaborative_filtering(graph, n_users, n_songs)

        with self.output().open('wb') as file:
            pickle.dump(model, file)


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

        with self.input()[1].open('rb') as file:
            model = pickle.load(file)

        graph = recommendations_graph(
            graph, model, n_users, n_songs, n_recommendations=1
        )
        graph.persist(self.output().path)


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
