
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#  HERRAMIENTA DASH - PROYECTO MINERÍA DE DATOS: PREDICCIÓN DE REGISTROS SÓNICOS A PARTIR DE REGISTROS BÁSICOS   
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# CARGA DE LIBRERÍAS
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import base64
import io
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# *************************************************************************************************************
#                              PROCESAMIENTO DE DATOS PARA VISUALIZACIÓN 
# *************************************************************************************************************

# Carga de Información general de pozos
df = pd.read_csv('Info_general_pozos.csv', sep = "\t")
df['Tipo'] = df.apply(lambda row: 'Sónico OH' if row['OPEN HOLE'] == 'X' else 'Sónico CH' , axis=1)

# Creación de dataframes para visualización de tablas en el dash con las métricas de desempeño
# Registro DTS
data_DTS = {
    "Modelo": ["XGBoost", "Random Forest", "LightGBM", "Gradient Boosting", "GAM (Splines + Inter)", "Red Neuronal", "GAM (Splines)", "Regression Lasso", "ElasticNet Regression", "LSTM"],
    "RMSE": [9.41, 9.82, 10.46, 10.48, 16.07, 16.42, 17.97, 19.28, 19.28, 17.57],
    "MAE": [6.61, 6.89, 7.61, 7.73, 12.36, 12.62, 14.03, 15.10, 15.10, 14.38],
    "R2": ["87.91%", "86.81%", "85.06%", "84.96%", "64.70%", "63.12%", "55.88%", "49.20%", "49.20%", "29.00%"]
}
data_DTS = pd.DataFrame(data_DTS)

# Registros DTC
data_DTC = {
    "Modelo": ["XGBoost", "Random Forest", "LightGBM", "Gradient Boosting", "GAM (Splines + Inter)", "Red Neuronal", "GAM (Splines)", "Regression Lasso", "ElasticNet Regression", "LSTM"],
    "RMSE": [3.27, 3.54, 3.57, 3.61, 5.14, 5.55, 5.89, 6.64, 6.64, 6.62],
    "MAE": [2.25, 2.41, 2.54, 2.58, 3.85, 4.23, 4.51, 5.09, 5.09, 5.23],
    "R2": ["87.97%", "85.93%", "85.63%", "85.31%", "70.24%", "65.22%", "60.89%", "50.31%", "50.31%", "19.41%"]
}
data_DTC = pd.DataFrame(data_DTC)

# Creación de lista de nombres de modelos para listas desplegables
Lista_Nombres = ["XGBoost","Random Forest","LightGBM","Gradient Boosting","GAM(Splines + Inter)","Red Neuronal","GAM(Splines)","Regresion Lasso","Elastic Net","Red Neuronal LSTM"]


# *************************************************************************************************************
#                              DEFINICIÓN DE FUNCIONES
# *************************************************************************************************************


# Función para graficar el desempeño de los modelos sobre el set de prueba
def grafica_desempeno_modelos(NOMBRE_MODELO, sample_size=1000):
    """
    La función permite obtener las gráficas de diagnóstico de desempeño sobre el set de prueba:
        - Valores predichos vs valores reales
        - Residuos vs valores predichos
        - Histograma de residuos
    
    input: 
        NOMBRE_MODELO: Nombre del modelo seleccionado para visualizar sus resultados.
        sample_size: tamaño de muestra a ser graficado, empleado para mejorar el rendimiento de la herramienta, valor por defecto = 1000
    
    output:
        gráficas de desempeño para la predicción de las variables objetivo sobre el set de prueba.

    """
    
    # Cargar las predicciones obtenidas para cada variable
    PREDDTC = pd.read_excel('y_test_DTC_Predicciones.xlsx')
    PREDDTS = pd.read_excel('y_test_DTS_Predicciones.xlsx')

    # Utilizar una muestra de los datos para mejorar el rendimiento
    PREDDTC_sample = PREDDTC.sample(n=min(sample_size, len(PREDDTC)))
    PREDDTS_sample = PREDDTS.sample(n=min(sample_size, len(PREDDTS)))


    # Se realizan las gráficas de desempeño de acuero a los modelos seleccionados.
    if NOMBRE_MODELO == "Red Neuronal LSTM":
        PREDDTC_sample['Model1_Residuals'] = PREDDTC_sample['DTC_REAL_SEQ'] - PREDDTC_sample[NOMBRE_MODELO]
        PREDDTS_sample['Model2_Residuals'] = PREDDTS_sample['DTS_REAL_SEQ'] - PREDDTS_sample[NOMBRE_MODELO]

        fig = make_subplots(rows=2, cols=3, subplot_titles=[
            '<b>DTC <br> Valores Reales  vs Predicciones<b>',
            '<b>DTC <br> Valores Reales vs Residuales<b>',
            '<b>DTC <br> Histograma de Residuales<b>',
            '<b>DTS <br> Valores Reales vs Predicciones<b>',
            '<b>DTS <br> Valores Reales vs Residuales<b>',
            '<b>DTS <br> Histograma de Residuales<b>' 
        ])

        # Scatter plot for Model1
        fig.add_trace(go.Scatter(x=PREDDTC_sample['DTC_REAL_SEQ'], y=PREDDTC_sample[NOMBRE_MODELO], mode='markers', 
                                 marker=dict(color='skyblue', line=dict(color='white', width=0.5)), showlegend=False),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[PREDDTC_sample['DTC_REAL_SEQ'].min(), PREDDTC_sample['DTC_REAL_SEQ'].max()], 
                                 y=[PREDDTC_sample['DTC_REAL_SEQ'].min(), PREDDTC_sample['DTC_REAL_SEQ'].max()], 
                                 mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=1, col=1)

        # Scatter plot of residuals for Model1
        fig.add_trace(go.Scatter(x=PREDDTC_sample['DTC_REAL_SEQ'], y=PREDDTC_sample['Model1_Residuals'], mode='markers', 
                                 marker=dict(color='skyblue', line=dict(color='white', width=0.5)), showlegend=False),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=[PREDDTC_sample['DTC_REAL_SEQ'].min(), PREDDTC_sample['DTC_REAL_SEQ'].max()], 
                                 y=[0, 0], mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=1, col=2)

        # Histogram of residuals for Model1
        fig.add_trace(go.Histogram(x=PREDDTC_sample['Model1_Residuals'], nbinsx=40, histnorm='density',
                                   marker=dict(color='lightgreen', line=dict(color='black', width=0.5)), showlegend=False),
                      row=1, col=3)

        # Scatter plot for Model2
        fig.add_trace(go.Scatter(x=PREDDTS_sample['DTS_REAL_SEQ'], y=PREDDTS_sample[NOMBRE_MODELO], mode='markers',
                                  marker=dict(color='lightgreen', line=dict(color='white', width=0.5)), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=[PREDDTS_sample['DTS_REAL_SEQ'].min(), PREDDTS_sample['DTS_REAL_SEQ'].max()], 
                                 y=[PREDDTS_sample['DTS_REAL_SEQ'].min(), PREDDTS_sample['DTS_REAL_SEQ'].max()], 
                                 mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=2, col=1)

        # Scatter plot of residuals for Model2
        fig.add_trace(go.Scatter(x=PREDDTS_sample['DTS_REAL_SEQ'], y=PREDDTS_sample['Model2_Residuals'], mode='markers', 
                                 marker=dict(color='lightgreen', line=dict(color='white', width=0.5)), showlegend=False),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=[PREDDTS_sample['DTS_REAL_SEQ'].min(), PREDDTS_sample['DTS_REAL_SEQ'].max()], 
                                 y=[0, 0], mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=2, col=2)

        # Histogram of residuals for Model2
        fig.add_trace(go.Histogram(x=PREDDTS_sample['Model2_Residuals'], nbinsx=40, marker=dict(color='lightgreen', line=dict(color='black', width=0.5)), showlegend=False),
                      row=2, col=3)

        fig.update_layout(height=800, width=1200, title_text="Gráficas de Desempeño del Modelo " + NOMBRE_MODELO, font=dict(size=15))
    
        # Update individual subplot properties
        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=1, col=1)
        fig.update_yaxes(title_text="<b>Predicciones<b>",title_font=dict(size=12), row=1, col=1)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=1, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=1, tickcolor='black',  ticks='outside', mirror="all")
     

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=1, col=2)
        fig.update_yaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=1, col=2)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=2, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=2, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=1, col=3)
        fig.update_yaxes(title_text="<b>Frecuencia<b>",title_font=dict(size=12), row=1, col=3)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=3, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=3, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=2, col=1)
        fig.update_yaxes(title_text="<b>Predicciones<b>",title_font=dict(size=12), row=2, col=1)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=1, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=1, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=2, col=2)
        fig.update_yaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=2, col=2)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=2, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=2, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=2, col=3)
        fig.update_yaxes(title_text="<b>Frecuencia<b>",title_font=dict(size=12), row=2, col=3)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=3, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=3, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_layout(plot_bgcolor='white')
        fig.update_annotations(font=dict(size=12))

        return fig

    else:
        PREDDTC_sample['Model1_Residuals'] = PREDDTC_sample['DTC_REAL'] - PREDDTC_sample[NOMBRE_MODELO]
        PREDDTS_sample['Model2_Residuals'] = PREDDTS_sample['DTS_REAL'] - PREDDTS_sample[NOMBRE_MODELO]

        fig = make_subplots(rows=2, cols=3, subplot_titles=[
            '<b>DTC <br> Valores Reales  vs Predicciones<b>',
            '<b>DTC <br> Valores Reales vs Residuales<b>',
            '<b>DTC <br> Histograma de Residuales<b>',
            '<b>DTS <br> Valores Reales vs Predicciones<b>',
            '<b>DTS <br> Valores Reales DTS vs Residuales<b>',
            '<b>DTS <br> Histograma de Residuales<b>' 
        ])

        # Scatter plot for Model1
        fig.add_trace(go.Scatter(x=PREDDTC_sample['DTC_REAL'], y=PREDDTC_sample[NOMBRE_MODELO], mode='markers', 
                                 marker=dict(color='skyblue',line=dict(color='white', width=0.5)), showlegend=False),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=[PREDDTC_sample['DTC_REAL'].min(), PREDDTC_sample['DTC_REAL'].max()], 
                                 y=[PREDDTC_sample['DTC_REAL'].min(), PREDDTC_sample['DTC_REAL'].max()], 
                                 mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=1, col=1)
        
        # Scatter plot of residuals for Model1
        fig.add_trace(go.Scatter(x=PREDDTC_sample['DTC_REAL'], y=PREDDTC_sample['Model1_Residuals'], mode='markers', 
                                 marker=dict(color='skyblue',line=dict(color='white', width=0.5)), showlegend=False),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=[PREDDTC_sample['DTC_REAL'].min(), PREDDTC_sample['DTC_REAL'].max()], 
                                 y=[0, 0], mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=1, col=2)

        # Histogram of residuals for Model1
        fig.add_trace(go.Histogram(x=PREDDTC_sample['Model1_Residuals'], nbinsx=40,  
                                   marker=dict(color='skyblue', line=dict(width=0.5, color='black')), showlegend=False),
                      row=1, col=3)

        # Scatter plot for Model2
        fig.add_trace(go.Scatter(x=PREDDTS_sample['DTS_REAL'], y=PREDDTS_sample[NOMBRE_MODELO], 
                                 mode='markers', marker=dict(color='lightgreen',line=dict(color='white', width=0.5)), showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=[PREDDTS_sample['DTS_REAL'].min(), PREDDTS_sample['DTS_REAL'].max()], 
                                 y=[PREDDTS_sample['DTS_REAL'].min(), PREDDTS_sample['DTS_REAL'].max()], 
                                 mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=2, col=1)

        # Scatter plot of residuals for Model2
        fig.add_trace(go.Scatter(x=PREDDTS_sample['DTS_REAL'], y=PREDDTS_sample['Model2_Residuals'], 
                                 mode='markers', marker=dict(color='lightgreen',line=dict(color='white', width=0.5)), showlegend=False),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=[PREDDTS_sample['DTS_REAL'].min(), PREDDTS_sample['DTS_REAL'].max()], 
                                 y=[0, 0], mode='lines', line=dict(color='red', dash='dot'), showlegend=False),
                      row=2, col=2)

        # Histogram of residuals for Model2
        fig.add_trace(go.Histogram(x=PREDDTS_sample['Model2_Residuals'], nbinsx=40, 
                                   marker=dict(color='lightgreen', line=dict(color='black', width=0.5)), showlegend=False),
                      row=2, col=3)

        fig.update_layout(height=800, width=1200, title_text="Gráficas de Desempeño del Modelo " + NOMBRE_MODELO, font=dict(size=15))

        # Update individual subplot properties
        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=1, col=1)
        fig.update_yaxes(title_text="<b>Predicciones<b>",title_font=dict(size=12), row=1, col=1)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=1, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=1, tickcolor='black',  ticks='outside', mirror="all")
     

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=1, col=2)
        fig.update_yaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=1, col=2)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=2, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=2, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=1, col=3)
        fig.update_yaxes(title_text="<b>Frecuencia<b>",title_font=dict(size=12), row=1, col=3)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=3, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=1, col=3, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=2, col=1)
        fig.update_yaxes(title_text="<b>Predicciones<b>",title_font=dict(size=12), row=2, col=1)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=1, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=1, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Valores Reales<b>",title_font=dict(size=12), row=2, col=2)
        fig.update_yaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=2, col=2)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=2, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=2, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_xaxes(title_text="<b>Residuales<b>",title_font=dict(size=12), row=2, col=3)
        fig.update_yaxes(title_text="<b>Frecuencia<b>",title_font=dict(size=12), row=2, col=3)
        fig.update_xaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=3, tickcolor='black',  ticks='outside', mirror="all")
        fig.update_yaxes(tickfont=dict(size=10), showgrid=False, linecolor='black', showline=True, row=2, col=3, tickcolor='black',  ticks='outside', mirror="all")

        fig.update_layout(plot_bgcolor='white', dragmode="pan", uirevision=True, autosize=True)
        fig.update_annotations(font=dict(size=12))


        fig.update_layout(newshape_line_color='blue', 
                        modebar_add=['drawline','drawopenpath','drawclosedpath',
                        'drawcircle','drawrect','eraseshape'])
        

        return fig



# Función para predecir las variables de interés a partir de un registro ingresado por el usuario y un modelo seleccionado.
def generar_predic(ESTADO, RAW_DATA):
    """
    La función permite obtener la predicción de los registros sónicos a partir de registros básicos ingresados por el usuario.

    input: 
        ESTADO: el estado indica que modelo emplear entre las tres opciones disponibles
        RAW_DATA: corresponde a los registro básicos empleados como variables predictoras.

    output:
        archivo dataframe con las variables predichas referenciadas en profundidad
    """

    ##### Preprocesamiento del archivo ingresado ######

    # Homologación del nombre de variables
    dicc_curv = {
        "MD": "Depth",
        "GRC": "GR",
        "BS": "BITSIZE",
        "HCAL": "CALI",
        "PE": "PEF",
        "PEFZ": "PEF",
        "HDRA": "DRHO",
        "TNPH": "NPHI",
        "RHOZ": "RHOB",
        "AT10": "RT10",
        "AT20": "RT20",
        "AT30": "RT30",
        "AT60": "RT60",
        "AT90": "RT90",
        "AHT10": "RT10",
        "AHT20": "RT20",
        "AHT30": "RT30",
        "AHT60": "RT60",
        "AHT90": "RT90"
    }
    RAW_DATA=RAW_DATA.copy()
    
    # Renombrar las columnas del DataFrame utilizando el diccionario
    RAW_DATA.rename(columns=dicc_curv, inplace=True)
    
    # Se filtran las columnas de interés (predictores seleccionados en los modelos)
    columnas_interes = ['Depth', 'SP', 'GR', "NPHI", "RHOB", "PEF","RT30"]
    DATOSFILTRADOSRAW = RAW_DATA[columnas_interes]

    # Renombrar las columnas temporalmente para que sean únicas
    DATOSFILTRADOSRAW.columns = [f"{col}_{i}" for i, col in enumerate(DATOSFILTRADOSRAW.columns)]

    # Crear una lista para almacenar los nombres de las columnas a remover
    columnas_a_eliminar = []

    # Recorrer las columnas por su índice de posición
    for i in range(DATOSFILTRADOSRAW.shape[1]):
        if DATOSFILTRADOSRAW.iloc[:, i].isnull().sum() == DATOSFILTRADOSRAW.shape[0]:
            columnas_a_eliminar.append(DATOSFILTRADOSRAW.columns[i])

    # Eliminar las columnas utilizando los nombres almacenados en la lista
    DATOSFILTRADOSRAW = DATOSFILTRADOSRAW.drop(columns=columnas_a_eliminar)

    # Revertir los nombres de las columnas si es necesario
    DATOSFILTRADOSRAW.columns = [col.split('_')[0] for col in DATOSFILTRADOSRAW.columns]

    # Se eliminan los faltantes. 
    DATOSFILTRADOS = DATOSFILTRADOSRAW.dropna()

    # Se aplica transformacion logaritimica sobre RT30
    DATOSFILTRADOS["log_RT30"] = DATOSFILTRADOS["RT30"].apply(lambda x: np.log(x ))
    DATOSFILTRADOS= DATOSFILTRADOS.drop(columns="RT30")
    
    # Se conserva la variable depth sin normalzar para la exportación
    depth = DATOSFILTRADOS["Depth"]

    # Se normalizan los datos
    scaler_X_loaded = joblib.load('./scaler_X.pkl')

    # Aplicar la transformación a los nuevos datos
    Datos_limpios = scaler_X_loaded.transform(DATOSFILTRADOS)
    Datos_limpios = pd.DataFrame(Datos_limpios, columns=DATOSFILTRADOS.columns)

    # Una vez generado el pipeline de procesamiento de datos, es posible emplear los modelos predictivos

    # Estado 1: Modelo XGBoost
    if ESTADO ==1 :
        
        # DTS
        # Cargar el modelo desde el archivo
        modelo_cargadoDTS = joblib.load('./best_model_XGB_DTS.pkl')
        # Realizar predicciones
        prediccionesDTS = modelo_cargadoDTS.predict(Datos_limpios)

        # DTC
        # Cargar el modelo desde el archivo
        modelo_cargadoDTC = joblib.load('./best_model_XGB_DTC.pkl')
        # Realizar predicciones
        prediccionesDTC = modelo_cargadoDTC.predict(Datos_limpios)

        # Se crea un dataframe con las predicciones referenciadas en profundidad
        Predicciones = pd.DataFrame({
            'Depth': depth,
            'PrediccionesDTC': prediccionesDTC,
            'PrediccionesDTS': prediccionesDTS,
        })

    # # Estado 2: Modelo Random Forest
    # if ESTADO ==2 :
        
    #     # DTS
    #     # Cargar el modelo desde el archivo
    #     modelo_cargadoDTS = joblib.load('./joblib_best_model_RF_DTS.sav')
    #     # Realizar predicciones
    #     prediccionesDTS = modelo_cargadoDTS.predict(Datos_limpios)

    #     # DTC
    #     # Cargar el modelo desde el archivo
    #     modelo_cargadoDTC = joblib.load('./joblib_best_model_RF_DTC.sav')
    #     # Realizar predicciones
    #     prediccionesDTC = modelo_cargadoDTC.predict(Datos_limpios)

    #     # Se crea un dataframe con las predicciones referenciadas en profundidad
    #     Predicciones = pd.DataFrame({
    #         'Depth': depth,
    #         'PrediccionesDTC': prediccionesDTC,
    #         'PrediccionesDTS': prediccionesDTS,
    #     })

    # Estado 2: Modelo LightGBM
    if ESTADO ==2 :
        
        # DTS
        # Cargar el modelo desde el archivo
        modelo_cargadoDTS = joblib.load('./best_model_lgb_DTS.pkl')
        # Realizar predicciones
        prediccionesDTS = modelo_cargadoDTS.predict(Datos_limpios)

        # DTC
        # Cargar el modelo desde el archivo
        modelo_cargadoDTC = joblib.load('./best_model_lgb_DTC.pkl')
        # Realizar predicciones
        prediccionesDTC = modelo_cargadoDTC.predict(Datos_limpios)

        # Se crea un dataframe con las predicciones referenciadas en profundidad
        Predicciones = pd.DataFrame({
            'Depth': depth,
            'PrediccionesDTC': prediccionesDTC,
            'PrediccionesDTS': prediccionesDTS,
        })
    return Predicciones



# Función para graficar los registros ingresados por el usuario junto con aquellos predichos.
def plot_prediccion(df, df_pred):
    """
    Funcion para graficar los registros usados como predictores
    y los registros predichos.

    Input:
        df: dataframe creado a partir del archivo cargado por el usuario
        df_pred: dataframe generado por la función de predicción conteniendo los registros predichos.
    """

    # Identificar columnas no vacías para el pozo actual
    na_cols = df.isna().sum() == len(df)
    non_empty_cols = na_cols[~na_cols].index.tolist()

    # Dataframe con columnas no vacías
    df = df[non_empty_cols].copy()

    # Creación de diccionario de nombres para homologación de registros
    dicc_curv = {
        "MD": "Depth",
        "GRC": "GR",
        "BS": "BITSIZE",
        "HCAL": "CALI",
        "PE": "PEF",
        "PEFZ": "PEF",
        "HDRA": "DRHO",
        "TNPH": "NPHI",
        "RHOZ": "RHOB",
        "AT10": "RT10",
        "AT20": "RT20",
        "AT30": "RT30",
        "AT60": "RT60",
        "AT90": "RT90",
        "AHT10": "RT10",
        "AHT20": "RT20",
        "AHT30": "RT30",
        "AHT60": "RT60",
        "AHT90": "RT90"
    }

    # Renombrar las columnas del DataFrame utilizando el diccionario
    df.rename(columns=dicc_curv, inplace=True)

    # Crear los subplots para cada registros
    fig = make_subplots(rows=1, cols=7, shared_yaxes=True, horizontal_spacing=0.01)

    # Se agrega una grafica por cada registro
    fig.add_trace(go.Scatter(x=df['GR'], y=df['Depth'], name='GR', line=dict(color='darkred', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['SP'], y=df['Depth'], name='SP', line=dict(color='black', width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['RHOB'], y=df['Depth'], name='RHOB', line=dict(color='peru', width=1)), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['NPHI'], y=df['Depth'], name='NPHI', line=dict(color='blue', width=1)), row=1, col=4)
    fig.add_trace(go.Scatter(x=df['PEF'], y=df['Depth'], name='PEF', line=dict(color='navy', width=1)), row=1, col=5)
    fig.add_trace(go.Scatter(x=df['RT30'], y=df['Depth'], name='RT30', line=dict(color='green', width=1)), row=1, col=6)
    fig.add_trace(go.Scatter(x=df_pred['PrediccionesDTC'], y=df_pred['Depth'], name='DTC', line=dict(color='black', width=1)), row=1, col=7)
    fig.add_trace(go.Scatter(x=df_pred['PrediccionesDTS'], y=df_pred['Depth'], name='DTS', line=dict(color='red', width=1)), row=1, col=7)

    # Definir el formato común para eje X
    xaxis_style = dict(
        showgrid=True,
        gridcolor='lightgrey',
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
        side='top'
    )

    names = ["<b>GR<b>", "<b>SP<b>", "<b>RHOB<b>", "<b>NPHI<b>", "<b>PEF<b>", "<b>RT30<b>", "<b>DTC / DTS<br>Predicted<b>"]

    # Se aplica el formato común a todos los ejes x
    for col in range(1, 8):  # Se itera sobre las columnas
        fig.update_xaxes(title_text=names[col-1], row=1, col=col, **xaxis_style)

    # Se define el formato común para el eje Y
    yaxis_style = dict(
        showgrid=True,
        gridcolor='lightgrey',
        zeroline=False,
        showline=True,
        linecolor='black',
        mirror=True,
    )

    # Se aplica el formato común a todos los ejes Y
    fig.update_yaxes(title_text="Depth", row=1, col=1, autorange='reversed', **yaxis_style)
    for col in range(2, 8):  # Loop through y-axes for the second row
        fig.update_yaxes(row=1, col=col, autorange='reversed', **yaxis_style)

    # Se define escala logaritmica para la variable de resistividad
    fig.update_xaxes(type='log', row=1, col=6)

    # Actualización del formato
    fig.update_layout(
        height=900,
        width=1200,
        template='plotly_white',
        font=dict(size=12, color='black'),
        title_font=dict(size=16, color='black', family='Arial'),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig



# ***************************************************
#               APP LAYOUT - FRONTEND
# ***************************************************

external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Predicción Registros Sónicos"
server = app.server

app.layout = html.Div([

    html.Div([
        html.H1(children=['PREDICCIÓN DE REGISTROS SÓNICOS DE POZOS A PARTIR DE REGISTROS BÁSICOS', html.Br(), 'EMPLEANDO ALGORITMOS DE APRENDIZAJE DE MAQUINA'], style={'background-color': '#003f5c', 'color':'white', 'font-weight':'bold', 'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '100%'}),
    ], style={'background-color': '#003f5c', 'height': '100px'}),
    
    dbc.Container([       
        dbc.Row([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                    dbc.Col(html.H1("Descripción del problema", className='text-center text-primary mb-4')),
                ]),
                    html.Div([
                        html.Div('El análisis e interpretación de datos de registros de pozos (well logs) es una práctica común y fundamental en la caracterización cuantitativa de reservorios, evaluación de formaciones y completaciones. Diversas mediciones en pozos proporcionan información crucial para determinar la composición del reservorio, como las fracciones de volumen sólido y de fluido, así como los tipos de roca. Estos datos, cuando se integran con datos sísmicos, ayudan a reducir la ambigüedad en la interpretación geológica y a mejorar los modelos de reservorios de hidrocarburos. Los registros de pozos convencionales como gamma ray (GR), resistividad, densidad y porosidad de neutrones se consideran de fácil adquisición y se implementan en la mayoría de los pozos. Sin embargo, debido a consideraciones de costo o limitaciones de acceso, otros tipos de registros, como la resonancia magnética nuclear (NMR), dispersión dieléctrica, espectroscopía elemental y registros sónicos dipolares/cizallantes, se utilizan en un número limitado de pozos y no siempre están disponibles en todas las profundidades de interés.', className="text-muted text-black fs-9"), 
                        html.Br(),
                        html.Div('Los registros sónicos, que miden el tiempo de viaje de una onda elástica a través de la formación, son particularmente importantes en muchas aplicaciones. Por ejemplo, la integración sísmica-pozo requiere registros sónicos y de densidad como insumos. Además, los parámetros geomecánicos, que son desafiantes de medir directamente, pueden derivarse de los registros sónicos con la ayuda de información adicional como los tipos de roca. Los registros sónicos también proporcionan datos para derivar la porosidad de la formación, realizar correlaciones estratigráficas e identificar litologías, facies, fracturas y compactaciones. Tradicionalmente, se han aplicado diversos métodos de regresión y algoritmos basados en modelos estadísticos, deterministas o empíricos para reconstruir datos de registros faltantes a partir de los registros medidos en pozos cercanos. Sin embargo, estos enfoques pueden ser sensibles a los tipos de roca específicos en los intervalos de profundidad y requieren calibraciones laboriosas y que consumen mucho tiempo con conocimientos expertos. La relación entre los registros convencionales y los registros sónicos puede ser altamente compleja y sensible a muchos factores, lo que introduce incertidumbres significativas en los modelos deterministas.',
                                className="text-muted text-black fs-9")
                    ], style={'max-width': '1800px'})
                ]),
                style={'background-color': 'white', 'border': '1px solid lightgray', 'border-radius': '8px', 'padding': '10px'}
            )
        ]),
        
        html.Br(),

        dbc.Row([
            dbc.Card(
                dbc.CardBody([
            html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col(html.H1("Descripción de los Datos", className='text-center text-primary mb-4')),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.P(
                            """
                            Los datos empleados incluyen: información general de los pozos, información de topes y registros eléctricos. La fuente de esta información corresponde a bases de datos de pozos del campo, interpretaciones geológicas de topes y los registros tomados en cada pozos.
                            """
                        ),
                        html.H5("Información general de pozos"),
                        html.P(
                            """
                            Corresponde a la información básica de los pozos, la cual incluye su ubicación en términos de coordenadas y sector dentro del campos, entre otras descritas a continuación.
                            """
                        ),
                        html.Ul([
                            html.Li("POZO: nombre o identificador de cada pozo."),
                            html.Li("BLOQUE: bloque o sector del campo donde se encuentre ubicado el pozo."),
                            html.Li("LOCACIÓN: grupo de pozos o cluster al que pertenece cada pozo. Cluster en este contexto representa grupos de pozos que compartir ubicación en superficie."),
                            html.Li("X(m) y Y(m): coordenadas x y y de la ubicación del pozo en superficie."),
                            html.Li("MD (FT): profundidad total medida a lo largo del pozo en pies."),
                            html.Li("TVD (FT): profundidad total vertical del pozo en pies."),
                            html.Li("OPEN HOLE: indica si el pozo tiene registros Sónicos hueco abierto (sin revestimiento)."),
                            html.Li("CASED-HOLE: indica si el pozo tiene registros Sónicos hueco revestido."),
                            html.Li("FECHA: fecha de registro sónico del pozo."),
                            html.Li("INTERVALO REGISTRADO (FT-FT): intervalo sobre el cual se corrieron los registros sónicos."),
                            html.Li("NPHI, RHOB, PEFZ, GR, RESISTIVOS: indica si el pozo cuenta con cada tipo de registro así: neutrón, densidad, factor fotoeléctrico, gamma ray y resistivos."),
                            html.Li("FECHA: fecha de registros del pozo."),
                            html.Li("INTERVALO REGISTRADO (FT-FT).1: intervalo sobre el cual se corrieron los registros básicos.")
                        ]),
                        html.H5("Información de topes por pozo"),
                        html.P(
                            """
                            Esta información hace referencia a la identificación de diferentes formaciones a lo largo del pozo. Cada formación corresponde a cuerpos de arena de propiedades geológicas particulares.
                            """
                        ),
                        html.Ul([
                            html.Li("POZO: identificador o nombre de pozo."),
                            html.Li("FORMACIÓN: nombre de la formación."),
                            html.Li("TOPE MD (FT): profundidad a partir de la cual comienza la formación específica."),
                            html.Li("BASE MD (FT): profundidad a la cual termina la formación específica.")
                        ]),

                        html.H5("Información de Registros"),
                        html.P(
                            """
                            Esta información corresponde a los perfiles tomados a lo largo de la trayectoria del pozo. Cada pozo cuenta con un set de registros, en general se cuenta con:
                            """
                        ),
                        html.Div([
                            html.Ul([
                                html.Li([
                                    html.Strong("Gamma Ray (GR):"),
                                    " Mide la radioactividad natural de las formaciones para identificar lutitas (alta radioactividad) y formaciones no arcillosas como areniscas y calizas. Utilizado para la correlación de capas y la identificación de zonas con contenido de arcilla."
                                ]),
                                html.Li([
                                    html.Strong("Resistivos:"),
                                    " Miden la resistividad eléctrica de las formaciones, ayudando a identificar formaciones con hidrocarburos (alta resistividad) en contraste con formaciones saturadas de agua (baja resistividad). Esencial para la interpretación de saturación de hidrocarburos."
                                ]),
                                html.Li([
                                    html.Strong("Sónicos: (DTC/DTS)"),
                                    " Miden la velocidad de propagación de ondas acústicas para calcular la porosidad de las rocas, identificar litología y evaluar propiedades mecánicas. Útiles para la correlación de capas y detección de fracturas."
                                ]),
                                html.Li([
                                    html.Strong("Densidad (RHOB):"),
                                    " Mide la densidad de las formaciones utilizando rayos gamma para proporcionar información sobre la porosidad y litología. Ayuda en la identificación de diferentes tipos de rocas y la evaluación de propiedades del reservorio."
                                ]),
                                html.Li([
                                    html.Strong("Neutrón (NPHI):"),
                                    " Mide la porosidad de las formaciones a través de la interacción de neutrones con los núcleos de hidrógeno presentes. Utilizado para estimar la porosidad efectiva y diferenciar entre fluidos presentes en el reservorio."
                                ]),
                                html.Li([
                                    html.Strong("Caliper:"),
                                    " Mide el diámetro del pozo a lo largo de su profundidad para identificar cambios en el diámetro que pueden indicar problemas de perforación o zonas de fractura. Útil para la calibración de otros registros y la evaluación de la calidad del pozo."
                                ]),
                                html.Li([
                                    html.Strong("Efecto Fotoeléctrico (PEF):"),
                                    " Mide la absorción de rayos gamma en la formación para diferenciar entre tipos de rocas, como carbonatos y silicatos. Ayuda a la identificación litológica y la evaluación de la mineralogía del reservorio."
                                ])
                            ])
                        ]),
                        html.Div([
                            html.H3("Preprocesamiento de Datos"),
                            html.Ol([
                                html.Li("Homologación de nombre de registros: los nombres de los registros varían entre pozos, por tanto, se realizó una homologación de las variables sobre un nombre en común."),
                                html.Li("Para cada pozo se eliminaron aquellas variables que no contaran con datos."),
                                html.Li("Se unieron los registros de todos los pozos en un solo set de datos."),
                                html.Li("Se empleó la información de topes para crear una nueva variable que indicara la formación a la que corresponde cada observación en profundidad.")
                            ]),

                           html.H3("Procesamiento de Datos"),
                            html.H4("Exploración de Datos"),
                            html.Ul([
                                html.Li([
                                    html.Strong("Distribuciones y Limpieza de Datos Faltantes:"),
                                    " La exploración inicial de los datos incluyó el análisis de distribuciones de las variables, la identificación de datos faltantes y su limpieza."
                                ]),
                                html.Li([
                                    html.Strong("Correlaciones:"),
                                    " Se calcularon las correlaciones entre las variables de entrada (gamma ray, densidad, porosidad de neutrones) y las variables objetivo (DTC y DTS) para identificar relaciones significativas que podrían ser útiles para el modelado."
                                ]),
                                html.Li([
                                    html.Strong("Relevancia de características:"),
                                    " Se implementó el algoritmo de Random Forest para estimar la importancia de las variables y realizar una preselección de las variables más importantes."
                                ])
                            ]),
                            html.H4("Detección y Eliminación de Outliers"),
                            html.Ul([
                                html.Li("Se emplearon los algoritmos One-Class Support Vector Machine (One-Class SVM) y DBSCAN.")
                            ]),

                            html.H4("Transformación y Normalización de Variables"),
                            html.Ul([
                                html.Li("Se implementó una transformación logarítmica sobre las variables de resistividad con base en su comportamiento"),
                                html.Li("Normalización de los datos.")
                            ]),

                            html.Div([html.H4("Campo de Estudio"),
                            html.Li('El campo cuenta con 263 pozos perforados, 169 de ellos activos.'),
                            html.Li('​El registro sónico de interés se corrió en 12 pozos del campo, tanto DTC, como DTS, lo cuales corresponden a mediciones sónicas compresionales y tensionales, respectivamente.'),
                            html.Li('El dataset inicial al unir la información de todos los pozos contiene 222.443 observaciones. Una vez realizado un preprocesamiento y limpieza de los datos el dataset resultante contiene 69.086 observaciones.​'),
                        
                        ])

                    ])
                ])
            ])
        ]),
        ], style={'max-width': '1800px'})
        ]),
        style={'background-color': 'white', 'border': '1px solid lightgray', 'border-radius': '8px', 'padding': '10px'}
            )
        ]),

        html.Br(),
        html.Hr(),
        html.Div((html.H1("Visualizaciones", className='text-center text-primary mb-4'))),
        html.Hr(),
        
        html.Div(['Esta sección permite ingresar las coordenadas del pozo sobre el cual se desea realizar la predicción de los registros sónicos (DTC y DTS), de esta manera obtener su ubicación geográfica respecto a los pozos cuyos registros fueron empleados para entrenar el modelo de predicción. Una mayor cercanía sobre los pozos de entrenamiento debería representar una mayor confiabilidad sobre la predicción.']),
        html.Br(),
        dbc.Row([
            dbc.Col([
            dcc.Graph(figure={}, id='pozos_location_graph', config = {'toImageButtonOptions': {
                                                    'format': 'jpeg', # one of png, svg, jpeg, webp
                                                    'height': None, 'filename': 'Location',
                                                    'width': None,'scale': 15,  # Image quality
                                                    }, 'scrollZoom': True,'displaylogo': False}, style={'height' : '95vh',})
            ], width=8),
            dbc.Col([
                html.Div('Coordenadas pozo a predecir'),
                dbc.Row([
                dbc.Input(placeholder='Write X coordinate', type='number', id='x_input_box'),
                dbc.Input(placeholder='Write Y coordinate', type='number', id='y_input_box'),
                dbc.Button('Plot',id='plot_button')    
            ]),
            ],width=4),

        ]),
        
        html.Br(),
        html.Hr(),
        html.H3("Análisis Exploratorio"),
        html.Hr(),
        html.Div(['En las siguientes gráficas se presenta: la importancia de las variables predictoras, las correlaciones entre las variables y las respuestas, y la distribución de las variables más relevantes']),
        html.Br(),
        dbc.Card(
        dbc.Row([
            dbc.Col([
                dbc.Col([
                    html.Div('Importancia de las Características', style={'text-align':'center', 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Div('DTC', style={'position' : 'absolute','left': '12.5%'}),
                    html.Div('DTS', style={'position' : 'absolute', 'left': '37.5%'}),
                    html.Br(),
                    html.Img(src = 'https://i.postimg.cc/mrJz5qbp/feature-importance-DTC.png', style={'width': '50%', 'height': 'auto', 'justify-content': 'center'}),
                    html.Img(src = 'https://i.postimg.cc/DZGJkg0v/feature-importance-DTS.png', style={'width': '50%', 'height': 'auto', 'justify-content': 'center'})
                ]),
                html.Br(),
                dbc.Col([
                    html.Div('Correlaciones', style={'text-align':'center'}),
                    html.Img(src = 'https://i.postimg.cc/xTnkhL2Q/DTC-correlations.png', style={'width': '50%', 'height': 'auto', 'alignItems': 'center'}),
                    html.Img(src = 'https://i.postimg.cc/8cVF16fh/DTS-correlations.png', style={'width': '50%', 'height': 'auto', 'alignItems': 'center'})
                ])
            ],width=6),
                
            dbc.Col([
                html.Div('Distribución de Variables Revelantes',  style={'text-align': 'center','fontWeight': 'bold'},),
                html.Br(),
                html.Br(),
                dbc.Col([
                    html.Img(src = 'https://i.postimg.cc/nLC9TZtQ/Depth.png', style={'width':'50%','height': 'auto'}),
                    html.Img(src = 'https://i.postimg.cc/Jzhs4m0d/RT30.png', style={'width':'50%','height': 'auto'})
                ]),
                html.Br(),
                dbc.Col([
                    html.Img(src = 'https://i.postimg.cc/tCk1j7qx/RHOB.png', style={'width':'50%','height': 'auto'}),
                    html.Img(src = 'https://i.postimg.cc/RZbNW8nC/SP.png', style={'width':'50%','height': 'auto',})
                ]),
                html.Br(),
                dbc.Col([
                    html.Img(src = 'https://i.postimg.cc/L4zYxNGv/NPHI.png', style={'width':'50%','height': 'auto'}),
                    html.Img(src = 'https://i.postimg.cc/C5HRdxb9/GR.png', style={'width':'50%','height': 'auto'})
                ])
            ],width=6)
            
        ]),
        style={'background-color': 'white', 'border': '1px solid lightgray', 'border-radius': '8px', 'padding': '10px'}
            ),

        html.Br(),
        html.Hr(),
        html.Div((html.H1("Desempeño de Modelos", className='text-center text-primary mb-4'))),
        html.Hr(),
        
    dbc.Container([html.Div(['Las siguientes tablas presentan el desempeño de los modelos predictivos evaluados sobre el set de prueba']), html.Br(),
        dbc.Row([ 
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.Div(['Métricas desempeño', html.Br(), 'Predicción DTS - Set de Prueba'], className="text-black fs-3"),
                        html.Div([
                            dash_table.DataTable(
                                id='table_DTS',
                                columns=[{"name": i, "id": i} for i in data_DTS.columns],
                                data=data_DTS.to_dict('records'),
                                style_table={'overflowX': 'auto', 'width': '100%'}
                            )
                        ], style={'max-width': '800px'}),
                    ]),
                    style={'background-color': '#f8f9fa', 'border': '1px solid lightgray', 'border-radius': '8px', 'padding': '10px'}
                )
            ], width=6),
            
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.Div(['Métricas desempeño', html.Br(), 'Predicción DTC - Set de Prueba'], className="text-black fs-3"),
                        html.Div([
                            dash_table.DataTable(
                                id='table_DTC',
                                columns=[{"name": i, "id": i} for i in data_DTC.columns],
                                data=data_DTC.to_dict('records'),
                                style_table={'overflowX': 'auto', 'width': '100%'}
                            )
                        ], style={'max-width': '800px'}),
                    ]),
                    style={'background-color': '#f8f9fa', 'border': '1px solid lightgray', 'border-radius': '8px', 'padding': '10px'}
                )
            ], width=6)  
        ]),
    ]),
        
        html.Br(),
        
   
    dbc.Container([html.Div(['Seleccione el modelo de la lista desplegable para mostrar las gráficas de diagnóstico.']),
                 html.Br(),
        dbc.Row([
            
            dbc.Col([
                
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': model, 'value': model} for model in Lista_Nombres],
                    value=Lista_Nombres[0]  
                ),
            ], width=4),  
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='model-performance-graph', config = {'toImageButtonOptions': {
                                                    'format': 'jpeg', # one of png, svg, jpeg, webp
                                                    'height': None, 'filename': 'Desempeño',
                                                    'width': None,'scale': 15,  # Image quality
                                                    }, 'scrollZoom': True,'displaylogo': False},)# style={'height' : '95vh',})# 'width' : '56vw' }),
            ], width=16),  
        ]),
    ])
    ,
    html.Br(),
    html.Hr(),
    html.Div((html.H1("Motor de Predicción", className='text-center text-primary mb-4'))),
    html.Hr(),
    
    html.Div([
    dbc.Container([
        html.Div(['Esta sección permite la carga de los registros básicos de un pozo para la predicción de los registros sónicos, y seleccionar el modelo predictivo entre los tres modelos con mejor desempeño.', html.Br(), 'Una vez realizada la predicción, se visualizan los registros, y se habilita la opción de descarga']),
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.B("Carga de Registros"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Arrastra y suelta o ',
                        html.A('Selecciona archivo de registros en formato csv')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False,
                    accept='.csv'
                ),
                html.Div(id='output_data_upload'),
                html.Br(), html.Hr(), html.Br(),
                html.Div(['Seleccione el modelo predictivo:']),
                dcc.RadioItems(
                    id='model-radioitems',
                    options=[
                        {'label': 'XGBoost', 'value': 1},
                        # {'label': 'Random Forest', 'value': 2},
                        {'label': 'LightGBM', 'value': 2}
                    ],
                    value=1,  # Default value
                    labelStyle={'display': 'block'}
                ),
                html.Div([
                    html.Button('Predecir', id='predict-button', n_clicks=0, className='btn btn-primary mx-1'),
                    html.Button('Descargar', id='download-button', n_clicks=0, className='btn btn-success mx-1'),
                ], style={'textAlign': 'center', 'marginTop': '10px'}),
                dcc.Download(id='download-predictions'),
                html.Br(), html.Br(), 
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="output-predictions")
                ),
                dcc.Store(id='prediction-store', data={}, storage_type='memory')  # Variable de estado oculta
            ])
        ]),
        html.Br(),html.Br(),
        dbc.Row([
            dbc.Col(dcc.Graph(id='prediction-graph', config = {'toImageButtonOptions': {
                                                    'format': 'jpeg', # one of png, svg, jpeg, webp
                                                    'height': None, 'filename': 'Desempeño',
                                                    'width': None,'scale': 15,  # Image quality
                                                    }, 'scrollZoom': False,'displaylogo': False},))
        ])
        ])
    ])

        
    ], fluid=True, style={'background-color': '#e0e0e0', 'padding': '40px', 'max-width': '1800px'})  # Set light gray background for the entire dashboard excluding title
])




# ************************************************************************************
# ************************************************************************************
# ************************************************************************************
# ********************************* CALLBACKS*****************************************
# ************************************************************************************
# ************************************************************************************
# ************************************************************************************

# Callback para graficar coordenadas de pozo
@callback(
    Output(component_id='pozos_location_graph', component_property='figure'),
    Input(component_id='plot_button', component_property='n_clicks'),
    State('x_input_box','value'),
    State('y_input_box','value')
)
def update_graph(n_clicks, x, y):
    fig = px.scatter(df, x='X (m)', y='Y (m)', title='Mapa de ubicación de los pozos', text = 'POZO', color='Tipo', 
                     labels={'Tipo': 'Tipo Registro Sonico','X (m)' : 'Coordenada X (m)','Y (m)' : 'Coordenada Y (m)' })
    fig.update_traces(textposition='top center', textfont=dict(color='black', size=7))
    fig.update_yaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside", scaleanchor = "x", scaleratio = 1,)
    fig.update_xaxes(tickformat = ".f", showline=True, linecolor='black', gridcolor='aliceblue', mirror="all",ticks="inside",)
    fig.update_annotations(font=dict(color='black', size=8),bgcolor='white', showarrow=False,)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False, yaxis_title=None, xaxis_title=None, 
            coloraxis_colorbar_x=-0.15,)
    fig.update_layout(newshape_line_color='blue', 
                    modebar_add=['drawline','drawopenpath','drawclosedpath',
                    'drawcircle','drawrect','eraseshape'])

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=True, yaxis_title=None, xaxis_title=None, dragmode="pan", uirevision=True, autosize=True,  
            coloraxis_colorbar_x=-0.15,)
    


    if x is not None and y is not None:
        new_point_df = pd.DataFrame({'X': [x], 'Y': [y]})
        fig.add_scatter(x=new_point_df['X'], y=new_point_df['Y'], mode='markers+text', 
                        marker=dict(size=10, color='darkgreen', symbol="triangle-up"), text = 'WELL', name='Pozo a predecir', textposition='top center'
                        )
        # Se definen los radios alrededor de la coordenada del pozo
        radii = [150, 500, 1000]

        # Se definen los colores para cada circunferencia
        colors = ['lightgray', 'gray', 'black']

        # Se agregan las circunferencias a la figura
        for radius, color in zip(radii, colors):
            fig.add_shape(
                type='circle',
                xref='x', yref='y',
                x0=x - radius, y0=y - radius,  
                x1=x + radius, y1=y + radius,  
                line=dict(color=color, width=0.5, dash='dot')
            )
            # Se agrega la medida de radio de cada circunferencia
            fig.add_annotation(
                x=x + radius,
                y=y,
                text=f'{radius}m',
                showarrow=False,
                yshift=10,
                xshift=12,
                font=dict(size=7)
            )
    return fig


# Callback para actualizar gráfico de performance
@app.callback(
    Output('model-performance-graph', 'figure'),
    Input('model-dropdown', 'value'),
)
def update_graph_1(selected_model):
    return grafica_desempeno_modelos(selected_model, sample_size=1000)


# Callback para mostrar el archivo ingresado por el usuario
@app.callback(Output('output_data_upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
            #   State('upload-data', 'last_modified'),
              )
def update_output(content, name_file):
        if content is not None:
            children = html.P('Archivo cargado: ' + name_file)

            return children 
        else:
            return None


# Callback para generar predicciones y actualizar el gráfico
@app.callback(
    [Output('output-predictions', 'children'),
     Output('prediction-graph', 'figure'),
     Output('prediction-store', 'data')],
    [Input('predict-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename'),
     State('model-radioitems', 'value')]
)
def update_output_1(n_clicks, contents, filename, selected_model):
    if n_clicks > 0 and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.BytesIO(decoded))

        # Generar predicciones
        predictions = generar_predic(selected_model, df)
        
        # Generar figura
        fig = plot_prediccion(df, predictions)
        
        return 'Predicción generada con éxito', fig, predictions.to_dict()
    return '', go.Figure(), {}


#  Callback para descargar las predicciones
@app.callback(
    Output('download-predictions', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('prediction-store', 'data')]
)
def download_predictions(n_clicks, data):
    if n_clicks > 0 and data:
        predictions = pd.DataFrame.from_dict(data)
        return dcc.send_data_frame(predictions.to_csv, "predicciones.csv", index=False)
    return None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


