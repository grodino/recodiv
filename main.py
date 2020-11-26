import luigi

from automation.msd_dataset import TrainModel
from automation.msd_dataset import DatasetInfo
from automation.msd_dataset import BuildDatasetGraph
from automation.msd_dataset import ComputeUsersDiversities
from automation.msd_dataset import GenerateRecommendations
from automation.msd_dataset import PlotUsersDiversitiesHistogram
from automation.msd_dataset import PlotRecommendationsDiversity


def main():
    luigi.build(
        [
            GenerateRecommendations(dataset_name='msd', dataset_folder='data/million_songs_dataset'),
            DatasetInfo(dataset_name='msd', dataset_folder='data/million_songs_dataset'),
            PlotUsersDiversitiesHistogram(dataset_name='msd', dataset_folder='data/million_songs_dataset')
        ],
        local_scheduler=True,
        log_level='INFO'
    )


if __name__ == '__main__':
    main()