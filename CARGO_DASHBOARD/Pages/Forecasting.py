import dash
from dash import html, dcc, dash_table, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, path='/dashboard', name='FORECASTING')

# Import data
cargo_throughput_data = pd.read_excel(r"C:\Users\qianh\OneDrive\Desktop\UUM\SEM 5\Python(1)\cargo data (2018-2023).xlsx",)
cargo_throughput_data['Date'] = pd.to_datetime(cargo_throughput_data['Date'])
print('Dataset of Cargo Throughput at Selected Ports - Peninsular Malaysia:')
print(cargo_throughput_data)

# Create Dash app
app = dash.Dash(__name__)

# Define layout
layout = html.Div([
    html.Div(children=[html.H1('Time Series Forecasting for Each Port', style={'color':'white','font-size': '24px', 'margin-top': '20px','padding-left': '15px'})]),
    dcc.RadioItems(
        id='port-radio',style={'color':'white','padding-left': '15px'},
        options=[
            {'label': 'Penang', 'value': 'Penang'},
            {'label': 'Klang', 'value': 'Klang'},
            {'label': 'Kuantan', 'value': 'Kuantan'},
            {'label': 'Port Dickson', 'value': 'Port Dickson'}
        ],
        value= 'Penang',
        labelStyle={'display': 'block'}
    ),
    dcc.Graph(id='forecast-plot'),

html.Div([
        html.Div([
            dash_table.DataTable(id='data-table',
                                 columns=[
                                     {'name': 'Date', 'id': 'Date'},
                                     {'name': 'Export Value Predict', 'id': 'Export Value Predict'},
                                     {'name': 'Import Value Predict', 'id': 'Import Value Predict'}
                                 ],
                                 style_table={'height': '100%', 'overflowY': 'auto', 'width': '80%', 'border': '1px solid white'},
                                 style_cell={'font_size': '15px', 'textAlign': 'center'},  
                                 style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                                 style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},),
        ], style={'width': '60%', 'display': 'inline-block','float':'center','padding-left': '250px'}),
        
        html.Div([
            dash_table.DataTable(id='metrics-table',
                                 columns=[
                                     {'name': 'Column', 'id': 'Column'},
                                     {'name': 'RMSE', 'id': 'RMSE'},
                                     {'name': 'MAPE(%)', 'id':'MAPE(%)'}
                                 ],
                                 style_table={'height': '100%', 'overflowY': 'auto', 'width': '80%', 'border': '1px solid white'},
                                 style_cell={'font_size': '15px', 'textAlign': 'center'},  
                                 style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                                 style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},),
        ], style={'width': '30%', 'display': 'inline-block','float':'center','margin':'auto'})
    ]),
])

# Define callback to update the graph and RMSE table
@callback(
    [Output('forecast-plot', 'figure'),
     Output('data-table', 'columns'),
     Output('data-table', 'data'),
     Output('metrics-table', 'data')],
    [Input('port-radio', 'value')]
)
def update_graph(selected_port):
    forecast_data = pd.DataFrame({'Date': []})
    metrics_data = []

    export_col = f'Export({selected_port})'
    import_col = f'Import({selected_port})'

   #Set training set and testing set
    training_set = cargo_throughput_data[:round(len(cargo_throughput_data)*70/100)]
    testing_set = cargo_throughput_data[round(len(cargo_throughput_data)*70/100):]

    # Plot forecasts for both export and import
    forecast_figure_export, predict_test_export, predicted_data_export = set_forecast(training_set, testing_set, export_col)
    forecast_figure_import, predict_test_import, predicted_data_import = set_forecast(training_set, testing_set, import_col)

    # Combine traces for export and import
    traces = forecast_figure_export['data'] + forecast_figure_import['data']

    # Combine predicted data for export and import
    forecast_data = pd.concat([predicted_data_export.rename(columns={'Predict': 'Export Value Predict'}),
                               predicted_data_import.rename(columns={'Predict': 'Import Value Predict'})], axis=1)
    
    forecast_data['Export Value Predict'] = forecast_data['Export Value Predict'].round(2)
    forecast_data['Import Value Predict'] = forecast_data['Import Value Predict'].round(2)

    # Calculate RMSE for export and import using the custom function
    rmse_export = round(calculate_rmse(testing_set[export_col], predict_test_export['Predict']),2)
    rmse_import = round(calculate_rmse(testing_set[import_col], predict_test_import['Predict']),2)

    #Calculate MAPE for export and import
    mape_export = calculate_mape(testing_set[export_col], predict_test_export['Predict'])
    mape_import = calculate_mape(testing_set[import_col], predict_test_import['Predict'])

    metrics_data.append({'Column': 'Export', 'RMSE': rmse_export,'MAPE(%)': mape_export})
    metrics_data.append({'Column': 'Import', 'RMSE': rmse_import,'MAPE(%)': mape_import})

    # Create combined graph
    forecast = go.Figure(data=traces, layout=forecast_figure_export['layout'])
    
    forecast.update_layout(xaxis_title='Date', yaxis_title='\'000 Tan Metrik(Freightweight)', 
                           paper_bgcolor="#1f1f1f",  
                           plot_bgcolor="#1f1f1f",   
                           font=dict(color="#ffffff"),
                           xaxis=dict(showgrid=False),  
                           yaxis=dict(showgrid=False), )

    # Convert combined predicted data to dictionary format for DataTable
    prediction_table = forecast_data.to_dict('records')

    selected_data = [{'name': 'Date', 'id': 'Date', 'type':'datetime'},
                     {'name': 'Export Value Predict', 'id': 'Export Value Predict'},
                     {'name': 'Import Value Predict', 'id': 'Import Value Predict'}]

    return forecast, selected_data, prediction_table, metrics_data

def set_forecast(train_dframe,test_dframe, col):
    order_dict = {
      'Export(Penang)': {'order': (0, 1, 1), 'seasonal_order': (0, 1, 1, 12)},
        'Import(Penang)': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
        'Export(Klang)': {'order': (1, 1, 1), 'seasonal_order': (0, 1, 1, 12)},
        'Import(Klang)': {'order': (0, 1, 1), 'seasonal_order': (0, 1, 1, 12)},
        'Export(Kuantan)': {'order': (0, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
        'Import(Kuantan)': {'order': (0, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
        'Export(Port Dickson)': {'order': (1, 0, 1), 'seasonal_order': (0, 1, 1, 12)},
        'Import(Port Dickson)': {'order': (0, 0, 0), 'seasonal_order': (0, 1, 1, 12)}
    }

    if col in order_dict:
        order = order_dict[col]['order']
        seasonal_order = order_dict[col]['seasonal_order']
    else:
        order = (0, 1, 1)
        seasonal_order = (0, 1, 1, 12)
    model_selected_column = sm.tsa.statespace.SARIMAX(train_dframe[col], order=order, seasonal_order=seasonal_order,
                                                    enforce_stationarity=False, enforce_invertibility=False)
    result = model_selected_column.fit()
    forecast_steps = 15
    future_pred = result.get_forecast(steps=forecast_steps)
    forecast_values = future_pred.predicted_mean

    cargo_throughput_data['Date'] = pd.to_datetime(cargo_throughput_data['Date'])
    cargo_throughput_data['Date'] = cargo_throughput_data['Date'].dt.date
    start_date = cargo_throughput_data['Date'].max()
    predicted_data = pd.DataFrame({
        'Date': pd.date_range(start=start_date, periods=forecast_steps + 1, freq='M')[1:],
        'Predict': forecast_values.values
    })

    predicted_data['Date'] = predicted_data['Date'].dt.date

    # Create forecast plot

    forecast_dates = pd.date_range(start=cargo_throughput_data['Date'].max(), periods=forecast_steps + 1, freq='M')
    if 'Export' in col:
        forecast_trace = go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name=f'Forecast-{col}', line=dict(dash='dash', color= px.colors.qualitative.Light24[7]))
        original_trace = go.Scatter(x=cargo_throughput_data['Date'], y=cargo_throughput_data[col], mode='lines', name=f'Original Data-{col}', line=dict(color= px.colors.qualitative.Plotly[1]))
    else:
        forecast_trace = go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name=f'Forecast-{col}', line=dict(dash='dash', color= px.colors.qualitative.Light24[6]))
        original_trace = go.Scatter(x=cargo_throughput_data['Date'], y=cargo_throughput_data[col], mode='lines', name=f'Original Data-{col}', line=dict(color= px.colors.qualitative.Light24[19]))

    # Create layout
    layout = dict(xaxis=dict(title='Date'), yaxis=dict(title='\'000 Tan Metrik(Freightweight)'),title='Future Cargo Export and Import Value Prediction')

    # Combine all traces into the final figure
    forecast_figure = dict(data=[original_trace,forecast_trace], layout=layout)

    #Set forecasting to calculate RMSE
    test_pred = result.get_forecast(steps=len(test_dframe))
    pred_mean = test_pred.predicted_mean
    predicted_testing_data = pd.DataFrame({
        'Date': pd.date_range(start=test_dframe['Date'].max(), periods=len(test_dframe)+1, freq='M')[1:],
        'Predict': pred_mean.values
    })
    return forecast_figure, predicted_testing_data, predicted_data

def calculate_rmse(actual,predicted):
    rmse = round(np.sqrt(mean_squared_error(actual,predicted)),2)
    return rmse

def calculate_mape(actual,predicted):
    mape = round((mean_absolute_percentage_error(actual,predicted))*100,2)
    return mape


