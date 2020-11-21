import luigi

from automation.msd_dataset import EvaluateCFModel
from automation.msd_dataset import PlotDiversityIncrease
from automation.msd_dataset import PlotRecommendationsDiversity



def main():
    luigi.build(
        [
            EvaluateCFModel(),
            PlotRecommendationsDiversity(),
            PlotDiversityIncrease()
        ],
        local_scheduler=True,
        log_level='INFO'
    )


if __name__ == '__main__':
    main()