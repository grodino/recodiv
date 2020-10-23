import luigi
from automan.api import Automator

from automation.msd_dataset import BuildGraph
from automation.msd_dataset import TrainCollaborativeFiltering


def main():
    # automator = Automator(
    #     simulation_dir='outputs',
    #     output_dir='figures',
    #     all_problems=[TrainCollaborativeFiltering]
    # )
    # automator.run()

    luigi.build([TrainCollaborativeFiltering(), BuildGraph()], local_scheduler=True)


if __name__ == '__main__':
    main()