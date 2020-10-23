import pickle

import luigi

from recodiv.utils import create_msd_graph
from recodiv.utils import train_msd_collaborative_filtering


class BuildGraph(luigi.Task):

    def output(self):
        return luigi.LocalTarget('generated/msd_graph', format=luigi.format.Nop)

    def run(self):
        graph, _ = create_msd_graph()

        self.output().makedirs()
        graph.persist(self.output().path)


class TrainCollaborativeFiltering(luigi.Task):

    def output(self):
        return luigi.LocalTarget('generated/msd_cf_model', format=luigi.format.Nop)

    def requires(self):
        return [BuildGraph()]

    def run(self):
        graph_file = self.input()[0]

        graph, (n_users, n_songs, n_categories) = create_msd_graph(
            recall_file=graph_file.path
        )
        model = train_msd_collaborative_filtering(graph, n_users, n_songs)

        with self.output().open('wb') as file:
            pickle.dump(model, file)

        print()
        print(model)
        print()
