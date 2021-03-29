# import os
# os.environ['LK_NUM_PROCS'] = '10,2'

from socket import MsgFlag
import click

from automation.config import *
from automation.interactive import *
from automation.msd_dataset import *


@click.group()
@click.option(
    '--n-users', 
    default=10_000, 
    type=int,
    help='Number of users to sample from the datasest'
)
@click.option(
    '--local-scheduler', 
    default=False, 
    type=bool,
    help='Use a luigi local scheduler for the tasks execution'
)
@click.option(
    '--name',
    default='MSD-10_000-users',
    type=str,
    help='The name of the folder where to save the experiments'
)
@click.pass_context
def cli(context: click.Context, n_users, local_scheduler, name):
    context.ensure_object(dict)

    context.obj['n_users'] = n_users
    context.obj['local_scheduler'] = local_scheduler
    context.obj['name'] = name


@cli.command()
@click.pass_context
def report_figures(context):
    """Lauch luigi to generate the report figures"""

    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']

    msd_dataset = MsdDataset(name, n_users=n_users)

    tasks = []

    # General information about the dataset
    tasks += [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotUserVolumeHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset),
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
        PlotDiversitiesIncreaseHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        )
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

    # # Recommendation diversity vs organic diversity at equilibrium and variations
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
            n_recommendations_values=N_RECOMMENDATIONS_VALUES
        )
    ]

    # Recommendation diversity versus the number of latent factors used in the model
    tasks += [
        PlotDiversityVsLatentFactors(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
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


    luigi.build(tasks, local_scheduler=local_scheduler, log_level='INFO', scheduler_host='127.0.0.1')


@cli.group()
def interactive():
    pass


@interactive.command()
@click.option(
    '--animated', 
    type=click.Choice(['latent-factors', 'reco-volume']),
    default='reco-volume', 
    help='Choose the variable to change during the animation'
)
@click.pass_context
def recommendation_diversity(context, animated):
    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']
    
    msd_dataset = MsdDataset(name, n_users=n_users)

    if animated == 'latent-factors':
        reco_div_vs_user_div_vs_latent_factors(msd_dataset, local_scheduler)
    
    elif animated == 'reco-volume':
        reco_div_vs_user_div_vs_reco_volume(msd_dataset, local_scheduler)


@interactive.command()
@click.option(
    '--animated', 
    type=click.Choice(['reco-volume']),
    default='reco-volume', 
    help='Choose the variable to change during the animation'
)
@click.pass_context
def diversity_increase(context, animated):
    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']
    msd_dataset = MsdDataset(name, n_users=n_users)

    div_increase_vs_user_div_vs_reco_volume(msd_dataset, local_scheduler)


if __name__ == '__main__':
    cli(obj={})
