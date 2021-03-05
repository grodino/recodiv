import os
# os.environ['LK_NUM_PROCS'] = '10,2'

import luigi
import numba
from lenskit.util import log_to_stderr

from automation.msd_dataset import *


def report_figures(msd_dataset):
    CONFIDENCE_FACTOR = 40
    N_ITERATIONS = 10
    N_RECOMMENDATIONS = 50
    N_FACTORS_VALUES = [5, 20, 50, 60, 70, 80, 200, 500, 1_000, 3_000]
    REGULARIZATION_VALUES = [.005, .01, 1.0, 10.0, 100.0, 200.0, 5_000.0, 1e5, 1e6]
    # TODO: change these values when using the better dataset (not the confidence pre-corrected ...)
    OPT_N_FACTORS = 500
    OPT_REGULARIZATION = 5_000.0

    tasks = []

    # General information about the dataset
    tasks += [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset)
    ]

    # General information about the train and test sets
    tasks += [
        TrainTestInfo(dataset=msd_dataset),
        PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset)
    ]

    # Model convergence plot
    tasks += [
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=3*N_ITERATIONS,
            model_n_factors=3_000,
            model_regularization=float(1e6),
            model_confidence_factor=CONFIDENCE_FACTOR
        ),
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=3*N_ITERATIONS,
            model_n_factors=200,
            model_regularization=float(1e-3),
            model_confidence_factor=CONFIDENCE_FACTOR
        ),
    ]

    # Hyper parameter tuning
    tasks += [
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='ndcg',
            tuning_best='max',
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='test_loss',
            tuning_best='min',
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # Recommendation diversity at equilibrium (optimal parameters)
    tasks += [
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
    ]
    
    # Recommendation diversity increase vs organic diversity at equilibrium and variations
    tasks += [
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
    ]

    # Recommendation diversity vs organic diversity at equilibrium and variations
    tasks += [
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        )
    ]

    # Recommendation diversity vs recommendation volume at equilibrium
    tasks += [
        PlotDiversityVsRecommendationVolume(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]   
        )
    ]

    # Diversity increase versus the number of latent factors used in the model
    tasks += [
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # # In depth analysis of outliers
    # tasks += [
    #     AnalyseUser(
    #         user_id='8a3a852d85deaa9e568c810e67a9707a414a59f4',
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     )
    # ]


    return tasks

def main():
    msd_dataset = MsdDataset(n_users=10_000)

    tasks = []
    tasks += report_figures(msd_dataset)

    luigi.build(tasks, local_scheduler=False, log_level='INFO', scheduler_host='127.0.0.1')


if __name__ == '__main__':
    main()
