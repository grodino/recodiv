import luigi

from automation.msd_dataset import *

def main():
    msd_dataset = MsdDataset()
    n_factors_values = list(range(30, 150, 10))

    tasks = [
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        DatasetInfo(dataset=msd_dataset),
        PlotDiversitiesIncreaseHistogram(dataset=msd_dataset),
        CollectFigures(dataset=msd_dataset),
        PlotDiversityVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
        PlotDiversityIncreaseVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
    ]
    tasks += [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    tasks += [PlotDiversitiesIncreaseHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    
    luigi.build(tasks, local_scheduler=False, log_level='INFO')


if __name__ == '__main__':
    main()