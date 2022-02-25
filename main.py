import click

from automation.config import *
from automation.interactive import *
from automation.paper import dev_tasks
from automation.paper import paper_figures
from automation.tasks.dataset import MsdDataset
from automation.tasks.utils import DeleteAllModelFigures
from automation.tasks.utils import DeleteAllModelAnalysis


@click.group()
@click.option(
    '--n-users',
    default=10_000,
    type=int,
    help='Number of users to sample from the datasest'
)
@click.option(
    '--local-scheduler/--no-local-scheduler',
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

    tasks = paper_figures(n_users, name)

    luigi.build(tasks, local_scheduler=local_scheduler,
                log_level='INFO', scheduler_host='127.0.0.1')


@cli.command()
@click.pass_context
def dev(context):
    """Run tasks in development"""

    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']

    tasks = dev_tasks(n_users, name)

    luigi.build(tasks, local_scheduler=local_scheduler,
                log_level='INFO', scheduler_host='127.0.0.1')


@cli.command()
@click.pass_context
def clean_models(context):
    """Keep the models, clear the models' folders"""

    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']

    msd_dataset = MsdDataset(name, n_users=n_users)

    task = DeleteAllModelAnalysis(dataset=msd_dataset)
    for file in task.will_delete():
        print(f'\t{file}')

    input('ARE YOU SURE YOU WANT TO DELETE THE FILES ? Press Enter to continue')
    luigi.build([task], local_scheduler=local_scheduler,
                log_level='INFO', scheduler_host='127.0.0.1')


@cli.command()
@click.pass_context
def clear_figures(context):
    """Clear the generated figures"""

    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']

    msd_dataset = MsdDataset(name, n_users=n_users)

    task = DeleteAllModelFigures(dataset=msd_dataset)
    for file in task.will_delete():
        print(f'\t{file}')

    input('ARE YOU SURE YOU WANT TO DELETE THE FILES ? Press Enter to continue')
    luigi.build([task], local_scheduler=local_scheduler,
                log_level='INFO', scheduler_host='127.0.0.1')


@cli.group()
@click.option(
    '--animated',
    type=click.Choice(['latent-factors', 'reco-volume']),
    default='reco-volume',
    help='Choose the variable to change during the animation'
)
@click.option(
    '--alpha',
    type=float,
    default=2,
    help='The order of the diversity to use.'
)
@click.pass_context
def interactive(context: click.Context, animated: str, alpha: float):
    """Lauch the interactive graphs server"""
    context.ensure_object(dict)

    # Avoid issues where 0.0 and 0 lead to different file titles
    alpha = float(alpha)
    alpha = int(alpha) if alpha.is_integer() else alpha

    context.obj['animated'] = animated
    context.obj['alpha'] = alpha


@interactive.command()
@click.pass_context
def recommendation_diversity(context):
    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']
    animated = context.obj['animated']

    msd_dataset = MsdDataset(name, n_users=n_users)

    if animated == 'latent-factors':
        reco_div_vs_user_div_vs_latent_factors(msd_dataset, local_scheduler)

    elif animated == 'reco-volume':
        reco_div_vs_user_div_vs_reco_volume(msd_dataset, local_scheduler)


@interactive.command()
@click.pass_context
def diversity_increase(context):
    n_users = context.obj['n_users']
    local_scheduler = context.obj['local_scheduler']
    name = context.obj['name']
    animated = context.obj['animated']
    alpha = context.obj['alpha']

    msd_dataset = MsdDataset(name, n_users=n_users)

    if animated == 'latent-factors':
        div_increase_vs_user_div_vs_latent_factors(
            msd_dataset, local_scheduler, alpha)

    elif animated == 'reco-volume':
        div_increase_vs_user_div_vs_reco_volume(
            msd_dataset, local_scheduler, alpha)


if __name__ == '__main__':
    cli(obj={})
