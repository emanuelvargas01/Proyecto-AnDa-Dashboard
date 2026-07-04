from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import joblib
import geopandas as gpd
##Lectura del DF
df = pd.read_csv("data/gold_dataset_limpio.csv")
df_inecMap = pd.read_csv("data/datos_sociodemográfica_población_título_universitario_con_título_universitario_distritos.csv")
geo_df = gpd.read_file("maps/geoBoundaries-PAN-ADM2_simplified.geojson")
NumericC1 = df.select_dtypes(include="number").columns
modelo = joblib.load("models/random_forest_regressor.joblib")

df_inecMap = df_inecMap.dropna()
geo_df = geo_df.dropna()

##LIMPIEZA DE DATOS PARA EL MAPA
def limpiar(texto):
    return (
        str(texto)
        .strip()
        .lower()
        .replace("á","a")
        .replace("é","e")
        .replace("í","i")
        .replace("ó","o")
        .replace("ú","u")
    )

geo_df["key"] = geo_df["shapeName"].apply(limpiar)
df_inecMap["key"] = df_inecMap["Nombre Distrito"].apply(limpiar)


df_inec_agg = df_inecMap.groupby("key", as_index=False)["Valor"].mean()


mapa_df = geo_df.merge(df_inec_agg, on="key", how="left")



##Creacion de app web
app = Dash(__name__)
## Graficos

fig = px.histogram(
    df,
    x=NumericC1[0],
    title="Distribución de la primera variable del dataset"
)
## Grafica 2 Boxplot
fig2 = px.box (
    df,
    y=df.columns[0],
    title= "Analisis de Dispersion Graf 2."
)
##Grafica 3
correlacion = df.corr(numeric_only=True)
fig3 = px.imshow(
    correlacion,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Matriz de Correlación",
)
fig3.update_layout(height=600)
## Grafica 4

if len(NumericC1) >= 2:
    fig4 = px.scatter(
        df,
        x=NumericC1[0],
        y=NumericC1[1],
        title= "Relacion entre variables numericas Graf.4"
    )
else:
    fig4 = px.scatter(title="No hay suficientes variables numéricas")



import json
import plotly.express as px

geojson = json.loads(mapa_df.to_json())

fig_map = px.choropleth_map(
    mapa_df,
    geojson=geojson,
    locations="shapeID",
    featureidkey="properties.shapeID",
    color="Valor",
    hover_name="shapeName",
    hover_data={"Valor": True},
    center={"lat": 8.5, "lon": -80},
    zoom=6,
    map_style="carto-positron",
    opacity=0.7,
    color_continuous_scale="Viridis"
)

fig_map.update_layout(
    margin=dict(l=0, r=0, t=40, b=0),
    height=700
)

top10 = (
    mapa_df
    .sort_values("Valor", ascending=False)
    .head(10)
)

fig_resumen = px.bar(
    top10,
    x="shapeName",
    y="Valor",
    color="Valor",
    title="Top 10 distritos con mayor población con título universitario",
    labels={
        "shapeName": "Distrito",
        "Valor": "Cantidad de personas"
    },
    text_auto=True,
    color_continuous_scale="Viridis"
)

fig_resumen.update_layout(
    xaxis_tickangle=-45,
    height=500
)


# ======================
# 4. LAYOUT
# ======================
app.layout = html.Div([

    html.H1("📈 Dashboard de Análisis del Precio del Oro"),

    # ================= CONTROLES =================
    html.Div([

        html.Div([
            html.Label("Variable"),
            dcc.Dropdown(
                id="variable-selector",
                options=[{"label":c,"value":c} for c in NumericC1],
                value="NASDAQXAU" if "NASDAQXAU" in NumericC1 else NumericC1[0],
                clearable=False
            )
        ], style={"width":"30%"}),

        html.Div([
            html.Label("Tipo de gráfico"),
            dcc.Dropdown(
                id="graph-selector",
                options=[
                    {"label":"Histograma","value":"hist"},
                    {"label":"Boxplot","value":"box"},
                    {"label":"Scatter","value":"scatter"},
                    {"label":"Correlación","value":"heat"}
                ],
                value="hist",
                clearable=False
            )
        ], style={"width":"30%"}),

        html.Div([
            html.Label("Año"),
            dcc.Dropdown(
                id="year-selector",
                options=[{"label":"Todos","value":"Todos"}] +
                        [{"label":str(a),"value":a} for a in sorted(df["Año"].unique())],
                value="Todos",
                clearable=False
            )
        ], style={"width":"20%"})

    ],
    className="control-panel",
    style={
        "display":"flex",
        "justifyContent":"space-around",
        "alignItems":"center",
        "gap":"20px"
    }),


    # ================= GRÁFICA =================

    html.Div([

        dcc.Graph(id="main-graph")

    ], className="graph-panel"),

    html.Br(),

    # ================= TARJETAS =================

    html.Div([

        html.Div([
            html.H3("Registros"),
            html.H2(id="card-registros")
        ], className="card"),

        html.Div([
            html.H3("Promedio"),
            html.H2(id="card-promedio")
        ], className="card"),

        html.Div([
            html.H3("Máximo"),
            html.H2(id="card-maximo")
        ], className="card"),

        html.Div([
            html.H3("Mínimo"),
            html.H2(id="card-minimo")
        ], className="card"),

    ],
    style={
        "display":"grid",
        "gridTemplateColumns":"repeat(4,1fr)",
        "gap":"20px"
    }),

    html.Br(),

    html.Div([

        html.H2("Predicción de NASDAQXAU para el día siguiente"),

        html.Label("Seleccione una fecha"),

        dcc.Dropdown(
            id="fecha-selector",
            options=[
                {"label": fecha, "value": fecha}
                for fecha in df["Fecha"].astype(str).unique()
            ],
            value=str(df["Fecha"].iloc[0]),
            clearable=False
        ),

        html.Br(),

        html.Button(
            "Predecir",
            id="btn-predecir",
            n_clicks=0
        ),

        html.Br(),
        html.Br(),

        html.H3(id="resultado-prediccion")

    ], className="graph-panel"),

    html.Br(),

    html.Div([

        html.H2("📊 Mapa Socioeconómico (INEC)"),

        dcc.Graph(
            id="map-inec",
            figure=fig_map
        ),

        html.Br(),

        html.H2("📈 Resumen del mapa"),

        dcc.Graph(
            id="grafica-resumen",
            figure=fig_resumen
        )

    ])

], style={
    "padding":"30px"
})
@app.callback(

    Output("main-graph","figure"),
    Output("card-registros","children"),
    Output("card-promedio","children"),
    Output("card-maximo","children"),
    Output("card-minimo","children"),

    Input("variable-selector","value"),
    Input("graph-selector","value"),
    Input("year-selector","value")

)
def actualizar_dashboard(variable, tipo, anio):

    # -----------------------------
    # Filtrar por año
    # -----------------------------

    df_filtrado = df.copy()

    if anio != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Año"] == anio]

    # -----------------------------
    # Crear la gráfica
    # -----------------------------

    if tipo == "hist":

        fig = px.histogram(
            df_filtrado,
            x=variable,
            title=f"Distribución de {variable}"
        )

    elif tipo == "box":

        fig = px.box(
            df_filtrado,
            y=variable,
            title=f"Boxplot de {variable}"
        )

    elif tipo == "scatter":

        # Elegimos una variable objetivo para comparar
        objetivo = "NASDAQXAU"

        if variable == objetivo and len(NumericC1) > 1:
            objetivo = NumericC1[1]

        fig = px.scatter(
            df_filtrado,
            x=variable,
            y=objetivo,
            title=f"{variable} vs {objetivo}"
        )

    else:

        correlacion = df_filtrado.select_dtypes(include="number").corr()

        fig = px.imshow(
            correlacion,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Matriz de Correlación"
        )

    # -----------------------------
    # Tarjetas
    # -----------------------------

    registros = f"{len(df_filtrado):,}"

    promedio = f"{df_filtrado[variable].mean():.2f}"

    maximo = f"{df_filtrado[variable].max():.2f}"

    minimo = f"{df_filtrado[variable].min():.2f}"

    return (
        fig,
        registros,
        promedio,
        maximo,
        minimo
    )
FEATURES = [
    "GVZCLS",
    "NASDAQXAU",
    "Año",
    "Mes_num",
    "Trimestre",
    "Mes",
    "Dia_semana",
    "GVZCLS_ret",
    "NASDAQXAU_ret",
    "GVZCLS_lag1",
    "NASDAQXAU_lag1",
    "GVZCLS_MA5",
    "NASDAQXAU_MA5",
    "GVZCLS_MA20",
    "NASDAQXAU_MA20",
    "GVZCLS_vol_5d",
    "NASDAQXAU_vol_5d",
    "Rango_Volatilidad_Oro"
]
@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    Input("fecha-selector", "value")
)
def predecir(n_clicks, fecha):

    if n_clicks == 0:
        return "Seleccione una fecha y presione Predecir."

    fila = df[df["Fecha"].astype(str) == str(fecha)]

    if fila.empty:
        return "No se encontró la fecha."

    datos = fila[FEATURES]

    prediccion = modelo.predict(datos)[0]

    return f"Predicción de NASDAQXAU para el día siguiente: {prediccion:.2f}"
# ======================
# 5. EJECUTAR
# ======================

if __name__ == "__main__":
    app.run(debug=True)