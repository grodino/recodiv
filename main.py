import os

import luigi
from lenskit.util import log_to_stderr

from automation.msd_dataset import *


def main():
    # log_to_stderr()
    # os.environ['LK_NUM_PROCS'] = '10'

    msd_dataset = MsdDataset(n_users=10_000)
    # msd_dataset = MsdDataset(n_users=0)
    n_factors_values = [30, 60, 100, 150, 200, 300, 400, 500, 1_000]
    regularization_values = [.01, .02, .03, .04, .05, .1, .3, .5, .7, 1, 5]
    n_recommendations = [10, 30, 50, 100, 200, 500, 1_000]

    n_factors_values = [400, 500, 1_000]
    regularization_values = [ .05, .1, .3]
    

    tasks = [
        DatasetInfo(dataset=msd_dataset),
        PlotUserVolumeHistogram(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset),
        BuildDatasetGraph(dataset=msd_dataset),
        TrainTestInfo(dataset=msd_dataset),
    ]
    tasks += [
        # PlotDiversityVsRecommendationVolume(dataset=msd_dataset, n_recommendations_values=n_recommendations),
        PlotDiversityVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
        # PlotDiversityIncreaseVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
        # TuneModelHyperparameters(dataset=msd_dataset, model_n_factors_values=n_factors_values, model_regularization_values=regularization_values),
        # PlotDiversityVsRegularization(dataset=msd_dataset, model_n_factors=100, model_regularization_values=regularization_values),
        PlotModelTuning(dataset=msd_dataset, model_n_factors_values=n_factors_values, model_regularization_values=regularization_values),
        # PlotDiversityIncreaseVsRegularization(dataset=msd_dataset, model_n_factors=100, model_regularization_values=regularization_values),
    ]
    tasks += [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    tasks += [PlotDiversitiesIncreaseHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    # tasks += [EvaluateModel(dataset=msd_dataset, model_n_factors=n, model_n_iterations=30) for n in n_factors_values]
    
    # tasks = [DeleteAllModelFigures(dataset=msd_dataset),]
    # tasks = [CollectAllModelFigures(dataset=msd_dataset),]

    luigi.build(tasks, local_scheduler=False, log_level='INFO', scheduler_host='127.0.0.1')


if __name__ == '__main__':
    main()
