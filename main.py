import os
os.environ['LK_NUM_PROCS'] = '20,2'

import luigi
import numba
from lenskit.util import log_to_stderr

from automation.msd_dataset import *


def main():
    log_to_stderr()

    # msd_dataset = MsdDataset(n_users=10_000)
    msd_dataset = MsdDataset(n_users=0)
    n_factors_values = list(range(50, 300, 20)) + [500,]
    # n_factors_values = list(range(30, 140, 10))
    n_factors_values = [200, 300, 400, 500, 800, 1_000]
    regularization_values = [.001, .005, .01, .02, .03, .04, .05]

    tasks = [
        DatasetInfo(dataset=msd_dataset),
        
        PlotUserVolumeHistogram(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset),
        
        TrainTestInfo(dataset=msd_dataset),
    ]
    tasks += [
    #     PlotDiversityVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
    #     PlotDiversityIncreaseVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
    #     TuneModelHyperparameters(dataset=msd_dataset, model_n_factors_values=n_factors_values, model_regularization_values=regularization_values),
    #     PlotDiversityVsRegularization(dataset=msd_dataset, model_n_factors=100, model_regularization_values=regularization_values),
        PlotModelTuning(dataset=msd_dataset, model_n_factors_values=n_factors_values, model_n_iterations=30, model_regularization_values=regularization_values),
    ]
    # tasks += [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    # tasks += [PlotDiversitiesIncreaseHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    # tasks += [EvaluateModel(dataset=msd_dataset, model_n_factors=n, model_n_iterations=30) for n in n_factors_values]
    
    # tasks = [DeleteAllModelFigures(dataset=msd_dataset),]

    luigi.build(tasks, local_scheduler=False, log_level='INFO')


if __name__ == '__main__':
    main()
