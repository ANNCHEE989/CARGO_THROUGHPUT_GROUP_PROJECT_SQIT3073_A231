import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

dash.register_page(__name__, path='/', name='OVERALL')

cargo_throughput_data = pd.read_excel(r"C:\Users\qianh\OneDrive\Desktop\UUM\SEM 5\Python(1)\cargo data (2018-2023).xlsx",)
cargo_throughput_data['Date'] = pd.to_datetime(cargo_throughput_data['Date'])
cargo_throughput_data['Year'] = cargo_throughput_data['Date'].dt.year
new_cargo_data = cargo_throughput_data.groupby('Year').agg({'Export(Penang)': 'sum', 'Import(Penang)': 'sum', 'Export(Klang)': 'sum', 'Import(Klang)': 'sum',
                                                            'Export(Kuantan)':'sum', 'Import(Kuantan)':'sum', 'Export(Port Dickson)':'sum', 'Import(Port Dickson)':'sum'}).reset_index()

port_latitudes = {'Penang': 5.4098, 'Klang': 2.999, 'Kuantan': 3.9767, 'Port Dickson': 2.5225}
port_longitudes = {'Penang': 100.3679, 'Klang': 101.3928, 'Kuantan': 103.4242, 'Port Dickson': 101.7963}

melted_cargo_data = pd.melt(new_cargo_data, id_vars=['Year'], var_name='Type', value_name='Tan Metrik(,000)')
melted_cargo_data['Port'] = melted_cargo_data['Type'].str.extract(r'\((.*?)\)')  # Extract port name from Type
melted_cargo_data['Type'] = melted_cargo_data['Type'].str.split('(').str[0]  # Remove port name from Type
melted_cargo_data['Lat'] = melted_cargo_data['Port'].map(port_latitudes)
melted_cargo_data['Lon'] = melted_cargo_data['Port'].map(port_longitudes)

final_cargo_data = melted_cargo_data[['Year', 'Port', 'Type', 'Tan Metrik(,000)', 'Lat', 'Lon']]
print(final_cargo_data)

app = dash.Dash(__name__)

layout = html.Div(children=[
    html.H1("Total Export and Import for Each Port",style={'color':'white','font-size': '24px', 'margin-top': '25px','padding-left': '15px'}),

    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': year, 'value': year} for year in final_cargo_data['Year'].unique()],
        value=final_cargo_data['Year'].max(),
        style={'width': '30%','padding-left': '15px'}
    ),

    dcc.Graph(id='cargo-bar-chart', style={'width': '52%', 'display': 'inline-block'}),
    dcc.Graph(id='scatter-geo-map', style={'width': '47%', 'display': 'inline-block'})
])

@callback(
    [Output('cargo-bar-chart', 'figure'),
     Output('scatter-geo-map', 'figure')],
    [Input('year-dropdown', 'value')]
)


def update_bar_chart(selected_year):
    filtered_df = final_cargo_data[final_cargo_data['Year'] == selected_year]
    grouped_df = filtered_df.groupby('Port').agg({'Tan Metrik(,000)': 'sum', 'Lat':'first', 'Lon': 'first'}).reset_index()
    grouped_df['Percentage(%)'] = (grouped_df['Tan Metrik(,000)'] / grouped_df['Tan Metrik(,000)'].sum() * 100).round(2)
    
    fig = px.bar(
        filtered_df,
        x='Port',
        y='Tan Metrik(,000)',
        color='Type',
        color_discrete_sequence=[px.colors.qualitative.Plotly[1],px.colors.qualitative.Light24[19],],
        barmode='group',
        labels={'Value': 'Cargo Value'},
        title=f'Cargo Throughput in {selected_year}'
    )

    center_lat = grouped_df['Lat'].mean()
    center_lon = grouped_df['Lon'].mean()

    scatter_geo_fig = px.scatter_geo(
    grouped_df,
    lat='Lat',
    lon='Lon',
    text='Port',
    size='Percentage(%)',
    color='Port',
    projection='natural earth',
    size_max=37,
    title=f'Percentage for total cargo throughput in {selected_year}',
    color_discrete_sequence=[
                 px.colors.qualitative.Dark24[0],
                 px.colors.qualitative.Alphabet[16],
                 px.colors.qualitative.G10[2],
                 px.colors.qualitative.Antique[1]] 
)
      
    scatter_geo_fig.update_geos(
    resolution=110,
    showcoastlines=True, coastlinecolor="Black",
    showland=True, landcolor="white",
    showocean=True, oceancolor="LightBlue",
    projection_scale=30,
    center=dict(lat=center_lat, lon=center_lon),
)
    
    scatter_geo_fig.update_traces(
    textfont_color='black'  
)
    
    scatter_geo_fig.update_layout(showlegend=False)
    
    return fig, scatter_geo_fig

