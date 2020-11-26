import luigi

from automation.msd_dataset import TrainModel
from automation.msd_dataset import GenerateRecommendations
from automation.msd_dataset import EvaluateCFModel
from automation.msd_dataset import PlotDiversityIncrease
from automation.msd_dataset import PlotRecommendationsDiversity

from recodiv.utils import create_msd_graph


def main():
    luigi.build(
        [
            #TrainCollaborativeFiltering(),
            # TrainModel(evaluate_iterations=True),
            GenerateRecommendations(dataset_name='msd', dataset_folder='recodiv/data/million_songs_dataset'),
        ],
        local_scheduler=True,
        log_level='INFO'
    )


if __name__ == '__main__':
    main()