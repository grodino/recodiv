from typing import List

import luigi

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
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            alpha_values=[0, 2, float('inf')],
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
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
            n_recommendations=N_RECOMMENDATIONS,
            bounds=[0, 75, -40, 40]
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[0, 70, -10, 40]
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[0, 75, -40, 30]
        ),
        # Richness
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        # Berger-Parker
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        # Almost Berger-Parker 1_000
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=1_000,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        # Almost Berger-Parker 500
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=500,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        # Almost Berger-Parker 250
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=250,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
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
            n_recommendations_values=[10, 50, 200]
        ),
        # Richness
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 200]
        ),
        # Berger-Parker
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 200]
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
    tasks += [
        PlotUserListeningRecommendationsTagsDistributions(
            dataset=msd_dataset,
            user='8a2986ddf6f0380a58638e384cc47ee0759a6369',
            n_tags=15,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
        ),
    ]

    # Compare the listened tags distribution to the recommended tags distribution for each user
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

    return tasks