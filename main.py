import luigi

from automation.msd_dataset import PlotRecommendationsDiversity
from automation.msd_dataset import PlotDiversityIncrease


def main():
    luigi.build(
        [
            PlotRecommendationsDiversity(),
            PlotDiversityIncrease()
        ],
        local_scheduler=True
    )


if __name__ == '__main__':
    main()