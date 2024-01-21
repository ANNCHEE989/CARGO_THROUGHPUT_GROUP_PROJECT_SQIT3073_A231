import os
os.system('cls')
import pandas as pd
import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output
from dash_table.Format import Format
import plotly.express as px

dash.register_page(__name__, path='/Trend', name='TREND')


cargo_throughput_data = pd.read_excel(r"C:\Users\qianh\OneDrive\Desktop\UUM\SEM 5\Python(1)\cargo data (2018-2023).xlsx",)
cargo_throughput_data['Date'] = pd.to_datetime(cargo_throughput_data['Date'])
print('Dataset of Cargo Through at Selected Ports-Peningsular Malaysia')
print(cargo_throughput_data)

app = dash.Dash(__name__)

# Define layout
layout = html.Div([
    html.H1("Cargo Throughput Trend from 2018 to 2023", style={'color':'white','font-size': '24px', 'margin-top': '25px','padding-left': '15px'}),
    dcc.Dropdown(
     id='year-dropdown',
     options=[{'label': year, 'value': year} for year in cargo_throughput_data['Date'].dt.year.unique()],
     multi=False,
     value=cargo_throughput_data['Date'].dt.year.max(),
     placeholder="Select Year",
     style={'margin-bottom': '25px','fontSize': '20px', 'width': '30%','padding-left': '15px'}
    ),
    dcc.RadioItems(
        id='port-radio',
        options=[
            {'label': 'Penang', 'value': 'Penang'},
            {'label': 'Klang', 'value': 'Klang'},
            {'label': 'Kuantan', 'value': 'Kuantan'},
            {'label': 'Port Dickson', 'value': 'Port Dickson'}
        ],
        value = 'Penang',
        labelStyle={'display': 'inline-block', 'margin-right': '10px', 'color':'white'},
        style={'fontSize': '20px','padding-left': '15px'}
    ),
    dcc.Graph(id='time-series-graph'),

    html.Div([
        html.Div([
            dcc.Graph(id='bar-chart', style={'backgroundColor': '#303030', 'margin-right': '45px'}),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            dash_table.DataTable(
                id='diff-table',
                columns=[
                    {'name': 'Date', 'id': 'Date', 'presentation': 'dropdown'},
                    {'name': 'Export Value', 'id': 'Export Value'},
                    {'name': 'Import Value', 'id': 'Import Value'},
                    {'name': 'Difference', 'id': 'Difference'}
                ],
                style_table={'height': '100%', 'overflowY': 'auto', 'width': '80%', 'border': '1px solid white'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'center'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                style_cell={'font_size': '15px', 'textAlign': 'center'}
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ], style={'margin-top': '25px', 'margin-bottom': '25px', 'width': '100%'}),
         
])

@callback(
    [Output('time-series-graph', 'figure'),
     Output('bar-chart','figure'),
     Output('diff-table','data')],
    [Input('year-dropdown','value'),
    Input('port-radio', 'value')]
)

def update_graph(selected_year, selected_port):
    if selected_year is not None and selected_port is not None:
        # Convert selected_year to int if needed
        selected_year = int(selected_year)
    
        if selected_year == 2023:
            # For 2023, set the end date to the last available date in September
            end_date = pd.to_datetime(f'2023-09-30')
        else:
            # For other years, set the end date to December 31
            end_date = pd.to_datetime(f'{selected_year}-12-31')

        start_date = pd.to_datetime(f'{selected_year}-01-31')

        export_col = f'Export({selected_port})'
        import_col = f'Import({selected_port})'

        # Use the complete list of months in the selected year
        all_months = pd.date_range(start=start_date, end=end_date, freq='M')
        all_months_df = pd.DataFrame({'Date': all_months})
        all_months_df['Month'] = all_months_df['Date'].dt.month_name()

        # Merge with the actual data, using outer join to include all months
        filter_df = pd.merge(all_months_df, cargo_throughput_data, on='Date', how='left')

        # Fill missing values with 0
        filter_df.fillna(0, inplace=True)

        # Sort DataFrame by Date
        filter_df.sort_values(by='Date', inplace=True)

        # Extract specific columns
        filter_df = filter_df[['Date', export_col, import_col]]
        print("Columns in filter_df:", filter_df.columns)
        
        # Create 'Export Value' and 'Import Value' columns
        filter_df['Export Value'] = filter_df[export_col]
        filter_df['Import Value'] = filter_df[import_col]

    fig = px.line(filter_df, x='Date', y=[export_col, import_col],
                  title=f'Cargo Throughput Over Time for {selected_port}',
                  labels={'value': 'Cargo Throughput (000 Tonnes)', 'Date': 'Date'},
                  color_discrete_sequence=[px.colors.qualitative.Plotly[1],px.colors.qualitative.Light24[19],],
                  line_dash_sequence=['solid', 'dot'])
    fig.update_xaxes(tickformat='%b', tickmode='array', tickvals=all_months, ticktext=all_months_df['Month'])
                  
    sum_of_export = filter_df[export_col].sum()
    sum_of_import = filter_df[import_col].sum()

    bar_chart = px.bar(x=['Export', 'Import'], y=[sum_of_export, sum_of_import],
                        title=f'Sum of Export and Import for {selected_port} in {selected_year}',
                        color_discrete_sequence=[px.colors.qualitative.Plotly[1],px.colors.qualitative.Light24[19],],
                        labels={'x': 'Operation'},color=['Export', 'Import'])
    bar_chart.update_yaxes(title_text='Cargo Throughput (000 Tonnes)')
   
    table_data = []
    for index, row in filter_df.iterrows():
        table_data.append({
            'Date': row['Date'],
            'Export Value': round(row[export_col],2),
            'Import Value': round(row[import_col],2),
            'Difference': round(row[export_col] - row[import_col],2)
        })
    
    for entry in table_data:
        entry['Date'] = entry['Date'].strftime('%b')

    return fig, bar_chart, table_data

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False)

