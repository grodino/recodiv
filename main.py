import luigi

from automation.msd_dataset import *

def main():
    msd_dataset = MsdDataset(n_users=10_000)
    n_factors_values = list(range(30, 150, 10))

    tasks = [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset),
        PlotTagsDiversitiesHistogram(dataset=msd_dataset)
    ]
    tasks += [
        PlotDiversitiesIncreaseHistogram(dataset=msd_dataset),
        PlotDiversityVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
        PlotDiversityIncreaseVsLatentFactors(dataset=msd_dataset, n_factors_values=n_factors_values),
        CollectFigures(dataset=msd_dataset),
        # EvaluateModel(dataset=msd_dataset),
    ]
    tasks += [PlotRecommendationsUsersDiversitiesHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    tasks += [PlotDiversitiesIncreaseHistogram(dataset=msd_dataset, model_n_factors=n) for n in n_factors_values]
    
    luigi.build(tasks, local_scheduler=False, log_level='INFO')


if __name__ == '__main__':
    main()