from click import style
from dash import Dash, html
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Proyecto Final - Dashboard", style={"textAlign": "center"}),

    html.H3("Análisis de Datos y Modelo Predictivo",
            style={"textAlign": "center"}),

    html.Hr(),

    html.P(
        "Dashboard interactivo del Proyecto Final.",
        style={"textAlign": "center"}
    )
])
if __name__ == '__main__':
    app.run(debug=True)