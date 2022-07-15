from typing import List

import luigi
import numpy as np
from matplotlib import pyplot as pl

from automation.config import *
from automation.tasks.dataset import DatasetInfo, MsdDataset
from automation.tasks.traintest import ComputeTrainTestUserDiversity, GenerateTrainTest, TrainTestInfo
from automation.tasks.hyperparameter import PlotRecommendationDiversityVsHyperparameter
from automation.tasks.model import EvaluateModel, EvaluateUserRecommendations, GeneratePredictions, GenerateRecommendations, PlotModelTuning, PlotTrainLoss, TrainModel
from automation.tasks.recommendations import BuildRecommendationGraph, BuildRecommendationsWithListeningsGraph, ComputeRecommendationDiversities, ComputeRecommendationWithListeningsUsersDiversities, ComputeRecommendationWithListeningsUsersDiversityIncrease, PlotRecommendationDiversityVsUserDiversity, PlotRecommendationsUsersDiversitiesHistogram, PlotUserDiversityIncreaseVsUserDiversity
from automation.tasks.user import PlotUserTagHistograms, AnalyseUser, ComputeUserListenedRecommendedTagsDistribution, ComputeUserRecommendationsTagsDistribution, ComputeRecommendationWithListeningsUsersDiversityIncrease, ComputeTrainTestUserTagsDistribution
from recodiv.utils import axes_to_grid


pl.rcParams.update({
    # "figure.figsize": [.6*6.4, .6*4.8],     # change figure default size
    # "figure.figsize": [1.2*6.4, 1.2*4.8],     # change figure default size
    "savefig.bbox": "tight",                # image fitted to the figure
    # grid lines for major and minor ticks
    "lines.linewidth": .7,                  # reduce linewidth to better see the points
    "font.family": "serif",                 # use serif/main font for text elements
    "font.size": 9,
    "legend.title_fontsize": 9,
    "legend.fontsize": 9,
    "mathtext.fontset": "dejavuserif",      # use serif font for math elements
    # "text.usetex": True,                    # use inline math for ticks
    "pgf.rcfonts": False,                   # don't setup fonts from rc parameters
    # "pgf.preamble": "\n".join([
    #     r"\usepackage{url}",                # load additional packages
    #     r"\usepackage{unicode-math}",       # unicode math setup
    #     # r"\setmainfont{DejaVu Serif}",      # serif font via preamble
    #     r"\renewcommand{\rmdefault}{ptm}",
    #     r"\renewcommand{\sfdefault}{phv}",
    #     r"\usepackage{pifont}",
    #     r"\usepackage{mathptmx}",
    # ])
})


def dev_tasks(n_users: int, name: str) -> List[luigi.Task]:
    """Tasks used to develop the models and test things out"""

    msd_dataset = MsdDataset(
        name,
        n_users=n_users,
        min_item_volume=10
    )

    split = dict(
        name='leave-one-out',
        n_fold=5,
        row_fraction=.1
    )

    # Small models, lots of values
    # latent_factors = np.logspace(3, 7, 5, base=2, dtype=int)
    # regularizations = np.logspace(-3, 3, 7)

    # Big models, few values
    latent_factors = np.array([128, 256, 512], dtype=int)
    regularizations = np.array([0.0001, 0.005, 0.1], dtype=float)
    grid = axes_to_grid(latent_factors, regularizations)

    # Optimal model found using grid search
    OPTIMAL_LATENT_FACTORS = 512
    OPTIMAL_REGULARIZATION = 0.005
    model = dict(
        name='implicit-MF',
        n_iterations=10,
        n_factors=512,
        regularization=0.005,
        confidence_factor=40,
    )

    users = [
        '57741fb06dc6557cf4748cc939ef27092174629b',  # 50, -20
        '805e941cd75a5891c99e15fa1ba19b13d089c908',  # 49, 16
        'fdeb46dea41366de2e745ecfacafadcbd9a93787',  # 12, -5
        'dbc1378cd624fb9f9ab1c7424fd75283f99538b0',  # 12, 25
    ]

    def data_info():
        return [
            DatasetInfo(dataset=msd_dataset),
            GenerateTrainTest(dataset=msd_dataset, split=split),
            TrainTestInfo(dataset=msd_dataset, split=split),
            ComputeTrainTestUserDiversity(
                dataset=msd_dataset, split=split, alpha=2
            ),
        ]

    def test_single_model():
        return [
            PlotRecommendationDiversityVsUserDiversity(
                dataset=msd_dataset,
                split=split,
                model=model,
                n_recommendations=10,
                alpha_values=[0, 2, float('inf')],
                fold_id=0,
                users=users,
            ),
            PlotUserDiversityIncreaseVsUserDiversity(
                dataset=msd_dataset,
                split=split,
                fold_id=0,
                model=model,
                n_recommendations_values=[10, 50, 100],
                alpha_values=[0, 2, float('inf')],
                users=users,
            ),
        ]

    def test_hyperparameter_grid():
        models = []
        for n_factors, regularization in grid:
            models.append(dict(
                name='implicit-MF',
                n_iterations=10,
                n_factors=int(n_factors),
                regularization=float(regularization),
                confidence_factor=40,
            ))

        metrics = {
            'ndcg': 'max',
            'recip_rank': 'max',
            'recall': 'max',
            'train_loss': 'min',
            'test_loss': 'min'
        }
        tasks = []

        for metric, tuning_best in metrics.items():
            tasks.append(PlotModelTuning(
                dataset=msd_dataset,
                models=models,
                split=split,
                n_recommendations=10,
                tuning_metric=metric,
                tuning_best=tuning_best,
            ))

        return tasks

    def diversity_vs_parameters():
        tasks = []

        # Diversity vs n factors
        models = []
        for n_factors in latent_factors:
            models.append(dict(
                name='implicit-MF',
                n_iterations=10,
                n_factors=int(n_factors),
                regularization=OPTIMAL_REGULARIZATION,
                confidence_factor=40,
            ))

        tasks.append(PlotRecommendationDiversityVsHyperparameter(
            dataset=msd_dataset,
            hyperparameter='n_factors',
            models=models,
            split=split,
            fold_id=2,
            alpha_values=[0, 2, float('inf')],
            n_recommendations_values=[10, 50, 100],
            # n_recommendations_ndcg=10,
            n_recommendations_ndcg=50,
        ))

        # Diversity vs regularization
        models = []
        for regularization in regularizations:
            models.append(dict(
                name='implicit-MF',
                n_iterations=10,
                n_factors=OPTIMAL_LATENT_FACTORS,
                regularization=regularization,
                confidence_factor=40,
            ))

        tasks.append(PlotRecommendationDiversityVsHyperparameter(
            dataset=msd_dataset,
            hyperparameter='regularization',
            models=models,
            split=split,
            fold_id=2,
            alpha_values=[0, 2, float('inf')],
            n_recommendations_values=[10, 50, 100],
            # n_recommendations_ndcg=10
            n_recommendations_ndcg=10
        ))

        return tasks

    def user_analysis():
        tasks = [
            AnalyseUser(
                dataset=msd_dataset,
                user_id=user_id,
                model=model,
                split=split,
                alpha_values=[0, 2, float('inf')],
                n_recommendation_values=[10, 50, 100],
                fold_id=0
            ) for user_id in users
        ]

        tasks += [
            PlotUserTagHistograms(
                dataset=msd_dataset,
                alpha=2,
                user_id=user_id,
                model=model,
                split=split,
                n_recommendations=N_RECOMMENDATIONS,
                fold_id=0,
                n_tags=30
            ) for user_id in users
        ]

        return tasks

    # return user_analysis()
    return (
        test_single_model()
        + user_analysis()
        + diversity_vs_parameters()
        + test_hyperparameter_grid()
        + data_info()
    )


def paper_figures(n_users: int, name: str) -> List[luigi.Task]:
    msd_dataset = MsdDataset(name, n_users=n_users)

    tasks = []

    # # General information about the dataset
    # tasks += [
    #     DatasetInfo(dataset=msd_dataset),
    #     PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=0),
    #     PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=2),
    #     PlotUsersDiversitiesHistogram(dataset=msd_dataset, alpha=float('inf')),
    #     PlotUserVolumeHistogram(dataset=msd_dataset),
    #     PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=0),
    #     PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=2),
    #     PlotTagsDiversitiesHistogram(dataset=msd_dataset, alpha=float('inf')),
    # ]

    # # General information about the train and test sets
    # tasks += [
    #     TrainTestInfo(dataset=msd_dataset),
    #     PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset, alpha=0),
    #     PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset, alpha=2),
    #     PlotTrainTestUsersDiversitiesHistogram(
    #         dataset=msd_dataset, alpha=float('inf')),
    # ]

    # # Model convergence plot
    # tasks += [
    #     PlotTrainLoss(
    #         dataset=msd_dataset,
    #         model_n_iterations=2*N_ITERATIONS,
    #         model_n_factors=3_000,
    #         model_regularization=float(1e6),
    #         model_confidence_factor=CONFIDENCE_FACTOR
    #     ),
    #     PlotTrainLoss(
    #         dataset=msd_dataset,
    #         model_n_iterations=3*N_ITERATIONS,
    #         model_n_factors=200,
    #         model_regularization=float(1e-3),
    #         model_confidence_factor=CONFIDENCE_FACTOR
    #     ),
    #     PlotTrainLoss(
    #         dataset=msd_dataset,
    #         model_n_iterations=3*N_ITERATIONS,
    #         model_n_factors=1_000,
    #         model_regularization=float(1e6),
    #         model_confidence_factor=CONFIDENCE_FACTOR
    #     ),
    # ]

    # # User recommendations evaluation scores
    # tasks += [
    #     # Best model found
    #     PlotUserEvaluationHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotUserEvaluationHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500
    #     ),
    #     PlotUserEvaluationHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=1_000
    #     ),
    #     # 200 factors
    #     PlotUserEvaluationHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=200,
    #         model_regularization=100.0,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    # ]

    # # Hyper parameter tuning
    # tasks += [
    #     PlotModelTuning(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         tuning_metric='ndcg',
    #         tuning_best='max',
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotModelTuning(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         tuning_metric='test_loss',
    #         tuning_best='min',
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotModelTuning(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         tuning_metric='train_loss',
    #         tuning_best='min',
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotModelTuning(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         tuning_metric='recip_rank',
    #         tuning_best='max',
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotModelTuning(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         tuning_metric='precision',
    #         tuning_best='max',
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    # ]

    # # Model performance for different number of latent factors
    # tasks += [
    #     PlotModelEvaluationVsLatentFactors(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         model_confidence_factor=CONFIDENCE_FACTOR,
    #         n_recommendations=N_RECOMMENDATIONS
    #     )
    # ]

    # # Recommendation diversity histogram at equilibrium (optimal parameters)
    # tasks += [
    #     # Herfindal
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=float('inf')
    #     ),
    #     # Herfindal
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=float('inf')
    #     ),
    #     # Herfindal
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotRecommendationsUsersDiversitiesHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=float('inf')
    #     ),
    # ]

    # # Diversity increase histogram at equilibrium
    # tasks += [
    #     # Herfindal
    #     PlotDiversitiesIncreaseHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=2
    #     ),
    #     # Herfindal
    #     PlotDiversitiesIncreaseHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=2
    #     ),
    #     # Herfindal
    #     PlotDiversitiesIncreaseHistogram(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=2
    #     ),
    # ]

    # # Recommendation diversity increase vs organic diversity at equilibrium and variations
    # tasks += [
    #     # Herfindal
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #         bounds=[0, 75, -40, 40],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         bounds=[0, 75, -40, 40],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         bounds=[0, 75, -40, 40],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     # Richness
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #         bounds=[-25, 800, -10, 1_000],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=True
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         bounds=[-15, 800, -10, 1_000],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         bounds=[-15, 800, -10, 1_000],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     # Berger-Parker
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #         bounds=[0, 25, -25, 15],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         bounds=[0, 25, -25, 15],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    #     PlotUserDiversityIncreaseVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         bounds=[0, 25, -25, 15],
    #         users=[
    #             '165300f45335433b38053f9b3617cc4eadaa2ecf',
    #             '767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #             'e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #             'c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         ],
    #         show_colorbar=False
    #     ),
    # ]

    # # Recommendation diversity vs organic diversity at equilibrium and variations
    # tasks += [
    #     # Herfindal diversity
    #     PlotRecommendationDiversityVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         bounds=[None, 80, None, None]
    #     ),
    #     # Richness
    #     PlotRecommendationDiversityVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     # Berger-Parker
    #     PlotRecommendationDiversityVsUserDiversity(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    # ]

    # # Recommendation diversity vs recommendation volume at equilibrium
    # tasks += [
    #     # Herfindal diversity
    #     PlotDiversityVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    #     # Richness
    #     PlotDiversityVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    #     # Berger-Parker
    #     PlotDiversityVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    # ]

    # # Diversity increase vs recommendation volume at equilibrium
    # deprecated = [
    #     PlotDiversityIncreaseVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    #     PlotDiversityIncreaseVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    #     PlotDiversityIncreaseVsRecommendationVolume(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=[5, 20, 500, 3_000],
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=N_RECOMMENDATIONS_VALUES
    #     ),
    # ]

    # # Recommendation diversity versus the number of latent factors used in the model
    # tasks += [
    #     PlotRecommendationDiversityVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=[10, 50, 500]
    #     ),
    #     PlotRecommendationDiversityVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=[10, 50, 500]
    #     ),
    #     PlotRecommendationDiversityVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=[10, 50, 500]
    #     ),
    # ]

    # # Recommendation diversity versus the regularization factor used in the model
    # tasks += [
    #     PlotDiversityVsRegularization(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         n_recommendations=N_RECOMMENDATIONS
    #     )
    # ]

    # # Diversity increase versus the number of latent factors used in the model
    # tasks += [
    #     # Herfindal
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500
    #     ),
    #     # Richness
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500
    #     ),
    #     # Berger-Parker
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     PlotDiversityIncreaseVsLatentFactors(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         n_factors_values=N_FACTORS_VALUES,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500
    #     ),
    # ]

    # # Diversity increase versus the regularization factor used in the model
    # tasks += [
    #     # Herfindal
    #     PlotDiversityIncreaseVsRegularization(
    #         dataset=msd_dataset,
    #         alpha=2,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     # Richness
    #     PlotDiversityIncreaseVsRegularization(
    #         dataset=msd_dataset,
    #         alpha=0,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    #     # Berger-Parker
    #     PlotDiversityIncreaseVsRegularization(
    #         dataset=msd_dataset,
    #         alpha=float('inf'),
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization_values=REGULARIZATION_VALUES,
    #         n_recommendations=N_RECOMMENDATIONS
    #     ),
    # ]

    # # Compare the listened, recommended and reco+listened tags distributions
    # tasks += [
    #     PlotUserTagHistograms(
    #         dataset=msd_dataset,
    #         user='165300f45335433b38053f9b3617cc4eadaa2ecf',
    #         n_tags=20,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #     ),
    #     PlotUserTagHistograms(
    #         dataset=msd_dataset,
    #         user='767153bf012dfe221b8bd8d45aa7d649aa37845a',
    #         n_tags=20,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #     ),
    #     PlotUserTagHistograms(
    #         dataset=msd_dataset,
    #         user='e6cdf0de3904fc6f40171a55eaa871503593cb06',
    #         n_tags=20,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #     ),
    #     PlotUserTagHistograms(
    #         dataset=msd_dataset,
    #         user='c0d9b4c9ca33db5a3a90fcf0072727ee0758a9c0',
    #         n_tags=20,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #     ),
    #     PlotUserTagHistograms(
    #         dataset=msd_dataset,
    #         user='f20fd75195cf378de0bb481b24936e12aabf8a19',
    #         n_tags=20,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=50,
    #     ),
    # ]

    # # Compare the best tag rank to the diversity increase of users
    # tasks += [
    #     # Herfindal
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS,
    #         alpha=float('inf')
    #     ),

    #     # Herfindal
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=500,
    #         alpha=float('inf')
    #     ),

    #     # Herfindal
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=2
    #     ),
    #     # Richness
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=0
    #     ),
    #     # Berger-Parker
    #     PlotHeaviestTagRankVsPercentageIncreased(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=10,
    #         alpha=float('inf')
    #     ),
    # ]

    # # Generate a summary of the metrics computed for a model
    # tasks += [
    #     MetricsSummary(
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations_values=[10, 50, 500],
    #         alpha_values=[0, 2, float('inf')]
    #     ),
    # ]

    return tasks
