import os
# os.environ['LK_NUM_PROCS'] = '10,2'

import luigi
import numba
from lenskit.util import log_to_stderr

from automation.msd_dataset import *

CONFIDENCE_FACTOR = 40
N_ITERATIONS = 10
N_RECOMMENDATIONS = 50
N_FACTORS_VALUES = [5, 20, 50, 60, 70, 80, 200, 500, 1_000, 3_000]
REGULARIZATION_VALUES = [.005, .01, 1.0, 10.0, 100.0, 200.0, 5_000.0, 1e5, 1e6]
# TODO: change these values when using the better dataset (not the confidence pre-corrected ...)
OPT_N_FACTORS = 500
OPT_REGULARIZATION = 5_000.0


def report_figures(msd_dataset):
    tasks = []

    # General information about the dataset
    tasks += [
        DatasetInfo(dataset=msd_dataset),
        PlotUsersDiversitiesHistogram(dataset=msd_dataset)
    ]

    # General information about the train and test sets
    tasks += [
        TrainTestInfo(dataset=msd_dataset),
        PlotTrainTestUsersDiversitiesHistogram(dataset=msd_dataset)
    ]

    # Model convergence plot
    tasks += [
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=3*N_ITERATIONS,
            model_n_factors=3_000,
            model_regularization=float(1e6),
            model_confidence_factor=CONFIDENCE_FACTOR
        ),
        PlotTrainLoss(
            dataset=msd_dataset,
            model_n_iterations=3*N_ITERATIONS,
            model_n_factors=200,
            model_regularization=float(1e-3),
            model_confidence_factor=CONFIDENCE_FACTOR
        ),
    ]

    # Hyper parameter tuning
    tasks += [
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='ndcg',
            tuning_best='max',
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotModelTuning(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors_values=N_FACTORS_VALUES,
            model_regularization_values=REGULARIZATION_VALUES,
            model_confidence_factor=CONFIDENCE_FACTOR,
            tuning_metric='test_loss',
            tuning_best='min',
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # Recommendation diversity at equilibrium (optimal parameters)
    tasks += [
        PlotRecommendationsUsersDiversitiesHistogram(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
    ]
    
    # Recommendation diversity increase vs organic diversity at equilibrium and variations
    tasks += [
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        ),
        PlotUserDiversityIncreaseVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
    ]

    # Recommendation diversity vs organic diversity at equilibrium and variations
    tasks += [
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=10
        ),
        PlotRecommendationDiversityVsUserDiversity(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            model_n_factors=OPT_N_FACTORS,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=500
        )
    ]

    # Recommendation diversity vs recommendation volume at equilibrium
    tasks += [
        PlotDiversityVsRecommendationVolume(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=[5, 20, 500, 3_000],
            model_regularization=OPT_REGULARIZATION,
            n_recommendations_values=[10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]   
        )
    ]

    # Diversity increase versus the number of latent factors used in the model
    tasks += [
        PlotDiversityIncreaseVsLatentFactors(
            dataset=msd_dataset,
            model_n_iterations=N_ITERATIONS,
            n_factors_values=N_FACTORS_VALUES,
            model_regularization=OPT_REGULARIZATION,
            n_recommendations=N_RECOMMENDATIONS
        )
    ]

    # # In depth analysis of outliers
    # tasks += [
    #     AnalyseUser(
    #         user_id='8a3a852d85deaa9e568c810e67a9707a414a59f4',
    #         dataset=msd_dataset,
    #         model_n_iterations=N_ITERATIONS,
    #         model_n_factors=OPT_N_FACTORS,
    #         model_regularization=OPT_REGULARIZATION,
    #         n_recommendations=N_RECOMMENDATIONS
    #     )
    # ]


    return tasks

def interactive(dataset):
    n_iterations = N_ITERATIONS
    regularization = OPT_REGULARIZATION
    n_recommendations = N_RECOMMENDATIONS

    merged = ComputeRecommendationDiversityVsUserDiversityVsLatentFactors(
        dataset=dataset,
        model_n_iterations=n_iterations,
        model_n_factors_values=N_FACTORS_VALUES,
        model_regularization=regularization,
        n_recommendations=n_recommendations
    ).run()

    fig = px.scatter(
        merged, 
        x='diversity', 
        y='reco_diversity',
        hover_data=['user'],
        custom_data=['n_factors'],
        animation_frame='n_factors',
        animation_group='user',
        marginal_x='histogram',
        marginal_y='histogram',
        color='volume', 
        color_continuous_scale=px.colors.sequential.Viridis,
        width=float('inf'),
        height=900,
        title='Effect of user diversity on their recommendation diversity',
        labels={
            'diversity': 'User individual diversity',
            'reco_diversity': 'Recommendation diversity',
            'volume': 'log10(volume)',
        }
    )
    fig.update_layout(coloraxis_colorbar=dict(
        title='volume',
        tickvals=[1, 1.477, 2, 2.477, 3],
        ticktext=['10', '30', '100', '300', '1000'],
    ))
    
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(
            id='basic-interactions',
            figure=fig
        ),

        html.Div(className='row', children=[
            html.Div([
                dcc.Markdown("""
                    **Click Data**

                    Click on points in the graph.
                """),
                html.Pre(id='click-data'),
            ], className='three columns'),
        ]),
    ])

    @app.callback(
        Output('click-data', 'children'),
        Input('basic-interactions', 'clickData'))
    def display_click_data(click_data):
        if not click_data:
            return

        point = click_data['points'][0]
        user_id = point['id']
        n_factors = int(point['customdata'][0])

        user_info = AnalyseUser(
            user_id=user_id,
            dataset=dataset,
            model_n_iterations=n_iterations,
            model_n_factors=n_factors,
            model_regularization=regularization,
            n_recommendations=n_recommendations
        ).run()

        return json.dumps(user_info, indent=2)

    app.run_server(debug=True, use_reloader=False) 

def main():
    msd_dataset = MsdDataset(n_users=10_000)

    tasks = []
    tasks += report_figures(msd_dataset)

    # luigi.build(tasks, local_scheduler=False, log_level='INFO', scheduler_host='127.0.0.1')
    interactive(msd_dataset)

if __name__ == '__main__':
    main()
