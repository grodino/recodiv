from typing import List

import luigi
from luigi import task

from automation.config import *
from automation.msd_dataset import *


def paper_figures(n_users: int, name: str) -> List[luigi.Task]:
    msd_dataset = MsdDataset(name, n_users=n_users)

    tasks = []

    # General information about the dataset
    tasks += [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=0),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=2),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=float('inf')),
        PlotUserVolumeHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=0),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=2),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=float('inf')),
    ]

    # General information about the train and test sets
    tasks += [
        TrainTestInfo(dataset=msd_dataset),
        PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset, alpha=0),
        PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset, alpha=2),
        PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset, alpha=float('inf')),
    ]

    # Model convergence plot
    tasks += [
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=2*N_ITERATIONS,
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
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=3*N_ITERATIONS,
            model_n_factors=1_000,
            model_regularization=float(1e6),
            model_confidence_factor=CONFIDENCE_FACTOR
        ),
    ]

    # User recommendations evaluation scores
    tasks += [
        # Best model found
        PlotUserEvaluationHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserEvaluationHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserEvaluationHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=1_000
        ),
        # 200 factors
        PlotUserEvaluationHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=200,
            model_regularization=100.0,
            n_recommendations=N_RECOMMENDATIONS
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
        ),
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='train_loss',
            tuning_best='min',
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='recip_rank',
            tuning_best='max',
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='precision',
            tuning_best='max',
            n_recommendations=N_RECOMMENDATIONS
        ),
    ]

    # Model performance for different number of latent factors
    tasks += [
        PlotModelEvaluationVsLatentFactors(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            model_confidence_factor=CONFIDENCE_FACTOR,
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # Recommendation diversity histogram at equilibrium (optimal parameters)
    tasks += [
        # Herfindal
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=2
        ),
        # Richness
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=0
        ),
        # Berger-Parker
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=float('inf')
        ),
         # Herfindal
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=2
        ),
        # Richness
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=0
        ),
        # Berger-Parker
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=float('inf')
        ),
         # Herfindal
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=2
        ),
        # Richness
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=0
        ),
        # Berger-Parker
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=float('inf')
        ),
    ]

    # Diversity increase histogram at equilibrium
    tasks += [
        # Herfindal
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=2
        ),
        # Richness
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=0
        ),
        # Berger-Parker
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=float('inf')
        ),
         # Herfindal
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=2
        ),
        # Richness
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=0
        ),
        # Berger-Parker
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=float('inf')
        ),
         # Herfindal
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=2
        ),
        # Richness
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=0
        ),
        # Berger-Parker
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=float('inf')
        ),
    ]
    
    # Recommendation diversity increase vs organic diversity at equilibrium and variations
    tasks += [
        # Herfindal
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[0, 75, -30, 40]
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[0, 75, -30, 40]
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[0, 75, -30, 30]
        ),
        # Richness
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[-15, 800, None, None],
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[-15, 800, None, None],
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[-15, 800, None, None],
        ),
        # Berger-Parker
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[0, 25, -12, 12],
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[0, 25, -12, 12],
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[0, 25, -12, 12],
        ),
    ]

    # Recommendation diversity vs organic diversity at equilibrium and variations
    tasks += [
        # Herfindal diversity
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Richness
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Berger-Parker
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Almost Berger-Parker 1_000
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Almost Berger-Parker 500
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Almost Berger-Parker 250
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
    ]

    # Recommendation diversity vs recommendation volume at equilibrium
    tasks += [
        # Herfindal diversity
        PlotDiversityVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
        # Richness
        PlotDiversityVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
        # Berger-Parker
        PlotDiversityVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,    
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
    ]

    # Diversity increase vs recommendation volume at equilibrium
    deprecated = [
        PlotDiversityIncreaseVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
        PlotDiversityIncreaseVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
        PlotDiversityIncreaseVsRecommendationVolume(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        ),
    ]

    # Recommendation diversity versus the number of latent factors used in the model
    tasks += [
        # Herfindal
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500]
        ),
        # Richness
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500]
        ),
        # Berger-Parker
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500]
        ),
    ]

    # Recommendation diversity versus the regularization factor used in the model
    tasks += [
        PlotDiversityVsRegularization(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization_values=REGULARIZATION_VALUES,
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # Diversity increase versus the number of latent factors used in the model
    tasks += [
        # Herfindal
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
         PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Richness
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        # Berger-Parker
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
    ]

    # Diversity increase versus the regularization factor used in the model
    tasks += [
        # Herfindal
        PlotDiversityIncreaseVsRegularization(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization_values=REGULARIZATION_VALUES,
            n_recommendations=N_RECOMMENDATIONS
        ),
        # Richness
        PlotDiversityIncreaseVsRegularization(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization_values=REGULARIZATION_VALUES,
            n_recommendations=N_RECOMMENDATIONS
        ),
        # Berger-Parker
        PlotDiversityIncreaseVsRegularization(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization_values=REGULARIZATION_VALUES,
            n_recommendations=N_RECOMMENDATIONS
        ),
    ]

    # Compare the listened tags distribution to the recommended tags distribution for a user
    tasks1 = [
        # High organic div, low reco div
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='dfe4fbf7cfa359b8489f1bbe488ef3a1affe4e94',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, same reco div
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='fc262e03c598b4e89b479f33162e6408d6ba90bf',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Low organic div, high recommended div
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='2821f2302af120e18a07a3efd934c65cbccdd6f0',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, high increase
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='8a2986ddf6f0380a58638e384cc47ee0759a6369',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, low increase
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='8f37b100d7f7aa6b3b1c6a4d0e2955e625688d6b',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='8c1fb29c006c873da67927ed944ae5b9aac05355',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
        ),
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='fb55b71799561c34e6edf10ce72bfbe2520601fe',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
        ),
    ]
    
    # Compare the listened, recommended and reco+listened tags distributions
    tasks1 = [
        # High organic div, low reco div
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='dfe4fbf7cfa359b8489f1bbe488ef3a1affe4e94',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, same reco div
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='fc262e03c598b4e89b479f33162e6408d6ba90bf',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Low organic div, high recommended div
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='2821f2302af120e18a07a3efd934c65cbccdd6f0',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, high increase
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='8a2986ddf6f0380a58638e384cc47ee0759a6369',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        # Average organic div, low increase
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='8f37b100d7f7aa6b3b1c6a4d0e2955e625688d6b',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='8c1fb29c006c873da67927ed944ae5b9aac05355',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='fb55b71799561c34e6edf10ce72bfbe2520601fe',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='bad44c40d5057f4d496718d97b577acf2281c790',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='08b890a2f7278345978f9b6932709aa5468f3e41',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),

        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='c3eee61e081ea89785c4fa4a4a0f29f9f5eb5829',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='cba327fdb10439a837331261b06b26cfb717a6d2',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='e6cdf0de3904fc6f40171a55eaa871503593cb06',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
    ]

    tasks += [
        # très diversifié au départ, decrease
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='f168dfc09049c7b08d55f4afbbf12da6e7e0a10e',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        # assez diversifié au départ, decrease
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='c9fe89fbfedf046d5d61a8361b00bab841da090f',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        # très diversifié au départ, très actif, decrease
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='165300f45335433b38053f9b3617cc4eadaa2ecf',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        # très diversifié au départ, increase
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='2e57bafa1e5e39c0c5bf460aaf727e9ca4e11a35',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        # assez diversifié au départ, increase
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='f20fd75195cf378de0bb481b24936e12aabf8a19',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        # très diversifié au départ, très actif, increase. 
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='767153bf012dfe221b8bd8d45aa7d649aa37845a',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
    ]
    
    # Compare the best tag rank to the diversity increase of users
    tasks += [
        # Herfindal
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=2
        ),
        # Richness
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=0
        ),
        # Berger-Parker
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            alpha=float('inf')
        ),

        # Herfindal
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=2
        ),
        # Richness
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=0
        ),
        # Berger-Parker
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            alpha=float('inf')
        ),

        # Herfindal
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=2
        ),
        # Richness
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=0
        ),
        # Berger-Parker
        PlotHeaviestTagRankVsPercentageIncreased(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=float('inf')
        ),
    ]

    # Generate a summary of the metrics computed for a model
    tasks += [
        MetricsSummary(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500],
            alpha_values=[0, 2, float('inf')]
        ),
    ]

    return tasks