import luigi

from automation.msd_dataset import *


def main():
    msd_dataset = MsdDataset()
    luigi.build(
        # [
        #     ImportDataset(dataset=msd_dataset),
        #     DatasetInfo(dataset=msd_dataset),
        #     BuildDatasetGraph(dataset=msd_dataset),
        #     ComputeUsersDiversities(dataset=msd_dataset),
        #     PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        #     ComputeTagsDiversities(dataset=msd_dataset),
        #     PlotTagsDiversitiesHistogram(dataset=msd_dataset),
        #     GenerateTrainTest(dataset=msd_dataset),
        #     TrainModel(dataset=msd_dataset),
        #     GenerateRecommendations(dataset=msd_dataset),
        #     BuildRecommendationGraph(dataset=msd_dataset),
        #     ComputeRecommendationUsersDiversities(dataset=msd_dataset),
        #     PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset)
        # ],
        [
            ImportDataset(dataset=msd_dataset),
            DatasetInfo(dataset=msd_dataset),
        ] + [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n) for n in [30, 50, 70, 100, 150]],
        local_scheduler=True,
        log_level='INFO'
    )


if __name__ == '__main__':
    main()