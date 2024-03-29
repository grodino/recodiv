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
        PlotTrainTestUsersDiversitiesHistogram(
            dataset=msd_dataset, alpha=float('inf')),
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
        # Herfindal
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            alpha=2
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
    ]

    # Recommendation diversity increase vs organic diversity at equilibrium and variations
    USERS = [
        '165300f45335433b38053f9b3617cc4eadaa2ecf',
        '767153bf012dfe221b8bd8d45aa7d649aa37845a',
        'e6cdf0de3904fc6f40171a55eaa871503593cb06',
        'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    ]

    tasks += [
        # Herfindal
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[0, 75, -40, 40],
            users=USERS,
            show_colorbar=False
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[0, 75, -40, 40],
            users=USERS,
            show_colorbar=False
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[0, 75, -40, 40],
            users=USERS,
            show_colorbar=False
        ),
        # Richness
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[-25, 800, -10, 1_000],
            users=USERS,
            show_colorbar=True
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[-15, 800, -10, 1_000],
            users=USERS,
            show_colorbar=False
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[-15, 800, -10, 1_000],
            users=USERS,
            show_colorbar=False
        ),
        # Berger-Parker
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
            bounds=[0, 25, -25, 15],
            users=USERS,
            show_colorbar=False
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500,
            bounds=[0, 25, -25, 15],
            users=USERS,
            show_colorbar=False
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10,
            bounds=[0, 25, -25, 15],
            users=USERS,
            show_colorbar=False
        ),
    ]

    # Analyse individual users
    for n_recommendations in (10, 50, 500):
        for user in USERS:
            tasks.append(AnalyseUser(
                dataset=msd_dataset,
                user_id=user,
                model_n_iterations=N_ITERATIONS,
                model_n_factors=OPT_N_FACTORS,
                model_regularization=OPT_REGULARIZATION,
                n_recommendations=n_recommendations,
                alpha_values=[0, 2, float('inf')],
                return_info=False,
            ))

    # Recommendation diversity vs organic diversity at equilibrium and variations
    tasks += [
        # Herfindal diversity
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS,
            bounds=[None, 80, None, None]
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
        # Berger-Parker
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            alpha=float('inf'),
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
    ]

    # Recommendation diversity vs recommendation volume at equilibrium
    deprecated = [
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
    canceled = [
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=2,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500]
        ),
        PlotRecommendationDiversityVsLatentFactors(
            dataset=msd_dataset,
            alpha=0,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 50, 500]
        ),
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
    canceled = [
        PlotDiversityVsRegularization(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization_values=REGULARIZATION_VALUES,
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # Diversity increase versus the number of latent factors used in the model
    canceled = [
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
    canceled = [
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

    # Compare the listened, recommended and reco+listened tags distributions
    tasks += [
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='165300f45335433b38053f9b3617cc4eadaa2ecf',
            n_tags=20,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='767153bf012dfe221b8bd8d45aa7d649aa37845a',
            n_tags=20,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='e6cdf0de3904fc6f40171a55eaa871503593cb06',
            n_tags=20,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
            n_tags=20,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=50,
        ),
        PlotUserTagHistograms(
            dataset=msd_dataset,
            user='f20fd75195cf378de0bb481b24936e12aabf8a19',
            n_tags=20,
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
