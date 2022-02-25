import shutil

import luigi

from automation.tasks.dataset import Dataset, DatasetInfo


################################################################################
# UTILS                                                                        #
################################################################################
class CollectAllModelFigures(luigi.Task):
    """Collect all figures related to a dataset in a single folder"""

    dataset: DatasetInfo = luigi.parameter.Parameter(
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
