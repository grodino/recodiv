import json
from pathlib import Path

import luigi
from luigi.format import Nop
import binpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from recodiv.model import train_model
from recodiv.model import evaluate_model_loss
from recodiv.model import generate_predictions
from recodiv.model import generate_recommendations
from recodiv.model import evaluate_model_recommendations
from automation.tasks.dataset import Dataset
from automation.tasks.traintest import GenerateTrainTest


################################################################################
# MODEL TRAINING/EVALUATION, RECOMMENDATION GENERATION                         #
################################################################################
class TrainModel(luigi.Task):
    """Train a given model and save it"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    save_training_loss = luigi.BoolParameter(
        default=False, description='Save the value of the training loss at each iteration'
    )

    def requires(self):
        return GenerateTrainTest(
            dataset=self.dataset,
            split=self.split
        )

    def output(self):
        model_str = '-'.join(
            '_'.join((key, str(val))) for key, val in self.model.items() if key != 'name'
        )
        model_path = self.dataset.base_folder.joinpath(
            f'models/{self.model["name"]}/{self.split["name"]}/' + model_str
        ).joinpath(f'fold_{self.fold_id}/')

        out = {'model': luigi.LocalTarget(
            model_path.joinpath('model.bpk')
        )}

        if self.save_training_loss == True:
            out['train_loss'] = luigi.LocalTarget(
                model_path.joinpath(f'train_loss.csv')
            )

        return out

    def run(self):
        for out in self.output().values():
            out.makedirs()

        train = pd.read_csv(self.input()[self.fold_id]['train'].path)

        if self.model['name'] == 'implicit-MF':
            model, loss = train_model(
                train,
                n_factors=self.model['n_factors'],
                n_iterations=self.model['n_iterations'],
                confidence_factor=self.model['confidence_factor'],
                regularization=self.model['regularization'],
                save_training_loss=self.save_training_loss
            )
        else:
            raise NotImplementedError('The asked model is not implemented')

        binpickle.dump(model, self.output()['model'].path)

        if self.save_training_loss == True:
            pd.DataFrame({'iteration': np.arange(loss.shape[0]), 'train_loss': loss}) \
                .to_csv(self.output()['train_loss'].path, index=False)

        del train, model, loss


class PlotTrainLoss(luigi.Task):
    """Plot the loss of a model for each iteration step"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    def requires(self):
        return TrainModel(
            dataset=self.dataset,
            model=self.model,
            split=self.split,
            fold_id=self.fold_id,
            save_training_loss=True
        )

    def output(self):
        model = Path(self.input()['model'].path).parent

        return luigi.LocalTarget(
            model.joinpath(f'fold_{self.fold_id}-training-loss.png'),
            format=Nop
        )

    def run(self):
        loss = pd.read_csv(self.input()['train_loss'].path)

        iterations = loss['iteration'].to_numpy()
        loss = loss['train_loss'].to_numpy()

        fig, ax = pl.subplots()
        ax.semilogy(iterations, loss)
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        fig.savefig(self.output().path, format='png', dpi=300)


class GenerateRecommendations(luigi.Task):
    """Generate recommendations for users in test dataset with a given model"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        return {
            'data': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split
            ),
            'model': TrainModel(
                dataset=self.dataset,
                model=self.model,
                save_training_loss=False,
                split=self.split,
                fold_id=self.fold_id
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent

        return luigi.LocalTarget(
            model.joinpath(f'recommendations-{self.n_recommendations}.csv')
        )

    def run(self):
        self.output().makedirs()

        model = binpickle.load(self.input()['model']['model'].path)
        ratings = pd.read_csv(self.input()['data'][self.fold_id]['test'].path)

        generate_recommendations(
            model,
            ratings,
            n_recommendations=self.n_recommendations
        ).to_csv(self.output().path, index=False)

        del ratings


class GeneratePredictions(luigi.Task):
    """Compute the predicted rating values for the user-item pairs in the test set"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )
    fold_id = luigi.parameter.IntParameter(
        default=0, description='Select the fold_id\'th train/test pair'
    )

    train_predictions = luigi.parameter.BoolParameter(
        default=False, description='Whether or not to compute predictions for the train set'
    )

    def requires(self):
        return {
            'data': GenerateTrainTest(
                dataset=self.dataset,
                split=self.split
            ),
            'model': TrainModel(
                dataset=self.dataset,
                model=self.model,
                save_training_loss=False,
                split=self.split,
                fold_id=self.fold_id
            )
        }

    def output(self):
        model = Path(self.input()['model']['model'].path).parent

        out = {}
        out['test'] = luigi.LocalTarget(
            model.joinpath(f'test-predictions.csv'))

        if self.train_predictions:
            out['train'] = luigi.LocalTarget(
                model.joinpath(f'train-predictions.csv')
            )

        return out

    def run(self):
        self.output()['test'].makedirs()

        test_user_item = pd.read_csv(
            self.input()['data'][self.fold_id]['test'].path)
        model = binpickle.load(self.input()['model']['model'].path)

        generate_predictions(
            model,
            test_user_item,
        ).to_csv(self.output()['test'].path, index=False)

        if self.train_predictions:
            train_user_item = pd.read_csv(
                self.input()['data'][self.fold_id]['train'].path)

            generate_predictions(
                model,
                train_user_item,
            ).to_csv(self.output()['train'].path, index=False)

            del train_user_item

        del test_user_item, model


class EvaluateUserRecommendations(luigi.Task):
    """Compute evaluations metrics on a trained model over all the crossfolds,
    user by user"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = []
        for fold_id in range(self.split['n_fold']):
            req.append({
                'recommendations': GenerateRecommendations(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    fold_id=fold_id,
                    n_recommendations=self.n_recommendations,
                ),
                'split': GenerateTrainTest(
                    dataset=self.dataset,
                    split=self.split
                )
            })

        return req

    def output(self):
        model = Path(self.input()[0]['recommendations'].path).parent.parent

        return luigi.LocalTarget(
            model.joinpath(
                f'users_eval-{self.n_recommendations}_reco.csv'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()

        metrics_names = ['ndcg', 'precision', 'recip_rank', 'recall']
        metrics = pd.DataFrame()

        for fold_id, data in enumerate(self.input()):
            recommendations = pd.read_csv(data['recommendations'].path)
            test = pd.read_csv(data['split'][fold_id]['test'].path)

            fold_metrics = evaluate_model_recommendations(
                recommendations,
                test,
                metrics_names
            )[metrics_names].reset_index()
            fold_metrics['fold_id'] = fold_id

            metrics = metrics.append(fold_metrics)

        metrics.to_csv(self.output().path, index=False)

        del recommendations, test, metrics


class EvaluateModel(luigi.Task):
    """Compute evaluations metrics on a trained model over each crossfolds
    averaged on all the users

    TODO: create an Avg version of this task
    """

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    model = luigi.parameter.DictParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        req = {
            'user_eval': EvaluateUserRecommendations(
                dataset=self.dataset,
                model=self.model,
                split=self.split,
                n_recommendations=self.n_recommendations
            ),
        }

        req['folds'] = []

        for fold_id in range(self.split['n_fold']):
            req['folds'].append({
                'model': TrainModel(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    fold_id=fold_id,
                    save_training_loss=False,
                ),
                'predictions': GeneratePredictions(
                    dataset=self.dataset,
                    model=self.model,
                    split=self.split,
                    train_predictions=True,
                    fold_id=fold_id,
                ),
            })

        return req

    def output(self):
        model_path = Path(self.input()['user_eval'].path).parent

        return luigi.LocalTarget(
            model_path.joinpath(
                f'model_eval-{self.n_recommendations}_reco.json'),
            format=Nop
        )

    def run(self):
        self.output().makedirs()
        user_metrics: pd.DataFrame = pd.read_csv(
            self.input()['user_eval'].path)
        user_metrics = user_metrics.set_index('user')

        metrics = pd.DataFrame()

        for fold_id, fold in enumerate(self.input()['folds']):
            # Average the user metrics over the users
            fold_metrics = user_metrics[user_metrics['fold_id'] == fold_id]
            fold_metrics = fold_metrics.mean()

            test_predictions = pd.read_csv(
                fold['predictions']['test'].path)
            train_predictions = pd.read_csv(
                fold['predictions']['train'].path)

            # Get the model trained with this fold
            model = binpickle.load(fold['model']['model'].path)

            # Evaluate the model loss on train and test data
            fold_metrics['test_loss'] = evaluate_model_loss(
                model, test_predictions)
            fold_metrics['train_loss'] = evaluate_model_loss(
                model, train_predictions)

            metrics = metrics.append(fold_metrics, ignore_index=True)

        metrics.to_json(
            self.output().path,
            orient='index',
            indent=4
        )

        del metrics


class TuneModelHyperparameters(luigi.Task):
    """Evaluate a model on a hyperparameter grid and get the best combination

    Currently, only the 'implicit-MF' models are supported. Each given model
    must only differ in the 'n_factors' and 'regularization' values.
    """

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    models = luigi.parameter.ListParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    def requires(self):
        required = {}

        for model in self.models:
            required[(model['n_factors'], model['regularization'])] = EvaluateModel(
                dataset=self.dataset,
                model=model,
                split=self.split,
                n_recommendations=self.n_recommendations
            )

        return required

    def output(self):
        path = self.dataset.base_folder.joinpath('aggregated/tuning')

        factors = list(set(model['n_factors'] for model in self.models))
        regularizations = list(set(
            model['regularization'] for model in self.models
        ))
        factors_str = ','.join(map(str, factors))
        reg_str = ','.join(map(str, regularizations))

        return {
            'optimal': luigi.LocalTarget(
                path.joinpath('-'.join((
                    f'{factors_str}_factors',
                    f'{reg_str}_reg',
                    f'{self.n_recommendations}_reco',
                    f'{self.models[0]["confidence_factor"]}_weight',
                    f'optimal_ndcg_param.json')),
                ),
                format=Nop
            ),
            'metrics': luigi.LocalTarget(
                path.joinpath('-'.join((
                    f'{factors_str}_factors',
                    f'{reg_str}_reg',
                    f'{self.n_recommendations}_reco',
                    f'{self.models[0]["confidence_factor"]}_weight',
                    f'metrics.csv')),
                ),
                format=Nop
            ),
        }

    def run(self):
        for folder in self.output().values():
            folder.makedirs()

        metrics = pd.DataFrame()

        for (n_factors, regularization), metrics_file in self.input().items():
            metric = pd.DataFrame()
            # Average over the different crossfolds
            metric = metric.append(pd.read_json(
                metrics_file.path, orient='index'
            ).mean(axis=0), ignore_index=True)

            metric['n_factors'] = n_factors
            metric['regularization'] = regularization

            metrics = pd.concat((metrics, metric))

        metrics = metrics.drop(columns='fold_id')
        metrics.set_index(['n_factors', 'regularization'], inplace=True)
        metrics.to_csv(self.output()['metrics'].path)

        optimal = {}
        opt_n_factors, opt_regularization = metrics.index[metrics['ndcg'].argmax(
        )]
        optimal['n_factors'] = float(opt_n_factors)
        optimal['regularization'] = float(opt_regularization)

        with open(self.output()['optimal'].path, 'w') as file:
            json.dump(optimal, file, indent=4)

        del metrics


class PlotModelTuning(luigi.Task):
    """Plot the 2D matrix of the model performance (ndcg value) on a
       hyperparameter grid"""

    dataset: Dataset = luigi.parameter.Parameter(
        description='Instance of the Dataset class or subclasses'
    )

    models = luigi.parameter.ListParameter(
        description='The parameters of the model, passed to the model training function'
    )

    split = luigi.parameter.DictParameter(
        description='Name and parameters of the split to use'
    )

    n_recommendations = luigi.parameter.IntParameter(
        default=50, description='Number of recommendation to generate per user'
    )

    tuning_metric = luigi.parameter.Parameter(
        default='ndcg', description='Which metric to use to tune the hyperparameters'
    )
    tuning_best = luigi.parameter.ChoiceParameter(
        choices=('min', 'max'),
        default='max',
        description='Whether the metric should be maximized or minimized'
    )

    def requires(self):
        return TuneModelHyperparameters(
            dataset=self.dataset,
            models=self.models,
            split=self.split,
            n_recommendations=self.n_recommendations,
        )

    def output(self):
        path = self.dataset.base_folder.joinpath(
            'aggregated/tuning').joinpath('figures')

        factors = list(set(model['n_factors'] for model in self.models))
        regularizations = list(set(
            model['regularization'] for model in self.models
        ))
        factors_str = ','.join(map(str, factors))
        reg_str = ','.join(map(str, regularizations))

        return luigi.LocalTarget(
            path.joinpath('-'.join((
                f'{factors_str}_factors',
                f'{reg_str}_reg',
                f'{self.n_recommendations}_reco',
                f'{self.models[0]["confidence_factor"]}_weight',
                f'{self.tuning_metric}_tuning.png')),
            ),
            format=Nop
        )

    def run(self):
        self.output().makedirs()
        metrics = pd.read_csv(self.input()['metrics'].path)

        metrics_matrix = metrics.pivot(
            index='n_factors', columns='regularization')[self.tuning_metric]
        metrics_matrix_n = metrics_matrix.to_numpy()

        fig, ax = pl.subplots()

        # Display the matrix as a heatmap
        img = ax.imshow(metrics_matrix_n)

        # Create the color bar
        cbar = fig.colorbar(img)
        cbar.ax.set_ylabel(self.tuning_metric.replace(
            "_", " "), rotation=-90, va="bottom")

        # Set the x and y axis values
        ax.set_xticks(list(range(len(metrics_matrix.columns))))
        ax.set_xticklabels(
            [f'{value:.0e}' for value in metrics_matrix.columns])

        ax.set_yticks(list(range(len(metrics_matrix.index))))
        ax.set_yticklabels(list(metrics_matrix.index))

        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
        pl.setp(ax.get_xticklabels(), rotation=-40,
                rotation_mode="anchor", ha="right")

        # Annotate the best value
        if self.tuning_best == 'min':
            opt_n_factors, opt_regularization = np.unravel_index(
                metrics_matrix_n.flatten().argmin(),
                metrics_matrix_n.shape
            )
            opt_text = 'MIN'
            opt_color = 'white'
        else:
            opt_n_factors, opt_regularization = np.unravel_index(
                metrics_matrix_n.flatten().argmax(),
                metrics_matrix_n.shape
            )
            opt_text = 'MAX'
            opt_color = 'black'

        ax.text(
            opt_regularization,
            opt_n_factors,
            opt_text,
            ha="center",
            va="center",
            color=opt_color
        )

        ax.set_ylabel('Number of latent factors')
        ax.set_xlabel('Regularization coefficient')

        fig.savefig(self.output().path, format='png', dpi=300)
        # tikzplotlib.save(self.output()['latex'].path)

        pl.clf()

        del fig, ax, metrics, metrics_matrix
