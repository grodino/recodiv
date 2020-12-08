import os

import luigi
from lenskit.util import log_to_stderr

from automation.msd_dataset import *


def main():
    log_to_stderr()
    os.environ['LK_NUM_PROCS'] = '10'

    msd_dataset = MsdDataset(n_users=10_000)
    # msd_dataset = MsdDataset(n_users=0)
    # n_factors_values = list(range(50, 300, 20)) + [500,]
    # n_factors_values = list(range(30, 140, 10))
    n_factors_values = [30, 100, 200]

    tasks = [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset),
        BuildDatasetGraph(dataset=msd_dataset),
        ComputeUsersDiversities(dataset=msd_dataset),
        TrainTestInfo(dataset=msd_dataset),
    ]
    tasks += [
        # PlotDiversitiesIncreaseHistogram(dataset=msd_dataset),
        #PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotDiversityVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values, model_n_iterations=30),
        PlotDiversityIncreaseVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values, model_n_iterations=30),
        # CollectAllModelFigures(dataset=msd_dataset),
    ]
    tasks += [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n, model_n_iterations=30) for n in n_factors_values]
    #tasks += [PlotDiversitiesIncreaseHistogram(dataset=msd_dataset, model_n_factors=n, model_n_iterations=30) for n in n_factors_values]
    # tasks += [EvaluateModel(dataset=msd_dataset, model_n_factors=n, model_n_iterations=30) for n in n_factors_values]
    
    # tasks = [DeleteAllModelFigures(dataset=msd_dataset),]

    luigi.build(tasks, local_scheduler=False, log_level='INFO')


if __name__ == '__main__':
    main()
