import luigi

from automation.msd_dataset import *


def main():
    msd_dataset = MsdDataset()
    luigi.build(
        [
            ImportDataset(dataset=msd_dataset),
            DatasetInfo(dataset=msd_dataset),
            BuildDatasetGraph(dataset=msd_dataset),
            ComputeUsersDiversities(dataset=msd_dataset),
            PlotUsersDiversitiesHistogram(dataset=msd_dataset),
            ComputeTagsDiversities(dataset=msd_dataset),
            PlotTagsDiversitiesHistogram(dataset=msd_dataset),
            GenerateTrainTest(dataset=msd_dataset),
            TrainModel(dataset=msd_dataset),
            GenerateRecommendations(dataset=msd_dataset)
            # BuildRecommendationGraph(dataset_name='msd', dataset_folder='data/million_songs_dataset'),
        ],
        local_scheduler=True,
        log_level='INFO'
    )


if __name__ == '__main__':
    main()