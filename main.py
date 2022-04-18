import click
import luigi

from automation.config import *
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


if __name__ == '__main__':
    cli(obj={})
