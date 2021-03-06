import json

import luigi
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input
from dash.dependencies import Output
import plotly.express as px

from automation.config import *
from automation.msd_dataset import AnalyseUser
from automation.msd_dataset import ComputeRecommendationDiversityVsUserDiversityVsRecoVolume
from automation.msd_dataset import ComputeRecommendationDiversityVsUserDiversityVsLatentFactors


def reco_div_vs_user_div_vs_latent_factors(msd_dataset, local_scheduler=False):
    """Interactive plot of the recommendation diversity vs the user diversity 
    for different number of latent factors"""

    n_iterations = N_ITERATIONS
    regularization = OPT_REGULARIZATION
    n_recommendations = N_RECOMMENDATIONS

    # make sure that all the dependencies of the task are completed by running
    # it in the scheduler
    data_task = ComputeRecommendationDiversityVsUserDiversityVsLatentFactors(
        dataset=msd_dataset,
        model_n_iterations=n_iterations,
        model_n_factors_values=N_FACTORS_VALUES,
        model_regularization=regularization,
        n_recommendations=n_recommendations
    )
    luigi.build([data_task], local_scheduler=local_scheduler, log_level='INFO')
    # then retrieve the data
    merged = data_task.run()

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
            dataset=msd_dataset,
            model_n_iterations=n_iterations,
            model_n_factors=n_factors,
            model_regularization=regularization,
            n_recommendations=n_recommendations
        ).run()

        return json.dumps(user_info, indent=2)

    app.run_server(debug=True, use_reloader=False)


def reco_div_vs_user_div_vs_reco_volume(msd_dataset, local_scheduler=False):
    """Interactive plot of the recommendation diversity vs the user diversity 
    for different number of latent factors"""

    n_iterations = N_ITERATIONS
    n_factors = OPT_N_FACTORS
    regularization = OPT_REGULARIZATION
    n_recommendations_values = N_RECOMMENDATIONS_VALUES

    # make sure that all the dependencies of the task are completed by running
    # it in the scheduler
    data_task = ComputeRecommendationDiversityVsUserDiversityVsRecoVolume(
        dataset=msd_dataset,
        model_n_iterations=n_iterations,
        model_n_factors=n_factors,
        model_regularization=regularization,
        n_recommendations_values=n_recommendations_values
    )
    luigi.build([data_task], local_scheduler=local_scheduler, log_level='INFO')
    # then retrieve the data
    merged = data_task.run()

    fig = px.scatter(
        merged, 
        x='diversity', 
        y='reco_diversity',
        hover_data=['user'],
        custom_data=['n_recommendations'],
        animation_frame='n_recommendations',
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
        n_recommendations = int(point['customdata'][0])

        user_info = AnalyseUser(
            user_id=user_id,
            dataset=msd_dataset,
            model_n_iterations=n_iterations,
            model_n_factors=n_factors,
            model_regularization=regularization,
            n_recommendations=n_recommendations
        ).run()

        # reorder the dict for better readability in the webpage
        keys = [
            'user_id',
            'model_n_iterations',
            'model_n_factors',
            'model_regularization',
            'model_user_fraction',
            'n_listened',
            'n_listened_tags',
            'n_recommended_items',
            'n_recommended_tags',
            'n_common_tags',
            'listened_items',
            'listened_tags',
            'recommended_items',
            'recommended_tags',
            'common_tags'
        ]
        user_info = {key: user_info[key] for key in keys}

        return json.dumps(user_info, indent=2)

    app.run_server(debug=True, use_reloader=False) 
