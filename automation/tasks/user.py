import json
from pathlib import Path

import luigi
import numpy as np
import pandas as pd

from recodiv.utils import get_msd_song_info
from recodiv.triversity.graph import IndividualHerfindahlDiversities
from automation.tasks.dataset import Dataset, ImportDataset
from automation.tasks.model import GenerateRecommendations
from automation.tasks.traintest import BuildTrainTestGraphs, ComputeTrainTestUserDiversity, GenerateTrainTest
from automation.tasks.recommendations import BuildRecommendationGraph, BuildRecommendationsWithListeningsGraph, ComputeRecommendationDiversities, ComputeRecommendationWithListeningsUsersDiversityIncrease


class AnalyseUser(luigi.Task):
    """ Look at the items listened by a user, its diversity before and after
        recommendation etc... """

    user_id = luigi.parameter.Parameter(
        description='The id string of the user'
    )

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )
    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )
    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    alpha_values = luigi.parameter.ListParameter(
        description='Values of the true diversity order to use'
    )
    fold_id = luigi.parameter.IntParameter(
        description='Which fold to use for the data'
    )
    n_recommendation_values = luigi.parameter.ListParameter(
        description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = {}

        for alpha in self.alpha_values:
            req[alpha] = {}
            req[alpha]['diversity'] = ComputeTrainTestUserDiversity(
                dataset=self.dataset,
                split=self.split,
                alpha=alpha,
            )

            for n_recommendations in self.n_recommendation_values:
                req[alpha][n_recommendations] = {}
                req[alpha][n_recommendations]['reco_diversity'] = ComputeRecommendationDiversities(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    alpha=alpha,
                    n_recommendations=n_recommendations,
                )
                req[alpha][n_recommendations]['diversity_increase'] = ComputeRecommendationWithListeningsUsersDiversityIncrease(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    alpha=alpha,
                    n_recommendations=n_recommendations,
                )

        return req

    def output(self):
        model = Path(
            self.input()
            [self.alpha_values[0]]
            [self.n_recommendation_values[0]]
            ['reco_diversity']
            [self.fold_id]
            .path
        ).parent

        return luigi.LocalTarget(model.joinpath(
            f'users/user_{self.user_id}'
            f'-n_recommendations_{self.n_recommendation_values}'
            f'-alpha_values_{self.alpha_values}'
            f'-info.json'
        ))

    def run(self):
        self.output().makedirs()
        info = {}

        for alpha in self.alpha_values:
            # Organic diversity
            organic_diversities = pd.read_csv(
                self.input()[alpha]['diversity'][self.fold_id]['test'].path
            ).set_index('user')
            info[f'alpha {alpha}'] = {
                'organic diversity': float(organic_diversities.loc[self.user_id]['diversity']),
                'recommendation diversity': {},
                'diversity increase': {}
            }

            for n_recommendations in self.n_recommendation_values:
                # Recommendation diversity
                reco_diversities = pd.read_csv(
                    self.input()
                    [alpha]
                    [n_recommendations]
                    ['reco_diversity']
                    [self.fold_id]
                    .path
                ).set_index('user')
                info[f'alpha {alpha}']['recommendation diversity'][n_recommendations] = \
                    float(reco_diversities.loc[self.user_id]['diversity'])

                # Diversity increase
                diversity_increase = pd.read_csv(
                    self.input()
                    [alpha]
                    [n_recommendations]
                    ['diversity_increase']
                    [self.fold_id]
                    .path
                ).set_index('user')
                info[f'alpha {alpha}']['diversity increase'][n_recommendations] = \
                    float(diversity_increase.loc[self.user_id]['diversity'])

        with self.output().open('w') as file:
            json.dump(info, file, indent=4)
