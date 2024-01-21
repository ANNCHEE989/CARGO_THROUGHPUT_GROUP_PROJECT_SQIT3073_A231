from dash import Dash, html, dcc
import dash
import plotly.express as px

# Define external CSS stylesheets
external_css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css", ]

# Set the default template for Plotly Express
px.defaults.template = 'plotly_dark'

app = Dash(__name__, pages_folder='Pages', use_pages=True, external_stylesheets=external_css)

app.layout = html.Div(
    [
        html.Br(),
        html.P('CARGO THROUGHPUT AT FOUR MAJOR PORTS IN MALAYSIA', className='text-light text-center fw-bold fs-1'),
        html.Div(
            children=[
                dcc.Link(page['name'], href=f"{page['relative_path']}", className='btn btn-light m-2 fs-5 text-dark')
                for page in dash.page_registry.values()
            ],
            className='col-8 ms-2 mb-7'
        ),
        dash.page_container
    ],
    className='bg-dark' 
)


if __name__ == '__main__':
    app.run(debug=True)
