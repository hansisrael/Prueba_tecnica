# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 23:46:30 2021

@author: tequi
"""
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
# ==============================================================================
# Cargamos el dataset
df = pd.read_csv("score_de_jamonosidad.csv")
#convertimos a tipo de dato numérico las columnas v1, v2, v3
df[['v1','v2','v3']] = df[['v1','v2','v3']].apply(pd.to_numeric)
df=df.drop(['jamon'],axis=1)
# Seleccionamos el valor de v1 y score para hacer regresion lineal simple y obtener R^2
#===============================================================================
X = df[['v1']]
y = df['score']
# Creación del modelo1
# ==============================================================================
modelo1 = LinearRegression()
modelo1.fit(X = X, y = y)
R2_1=modelo1.score(X,y)
# Seleccionamos el valor de v2 y score para hacer regresion lineal simple y obtener R^2
#===============================================================================
X = df[['v2']]
y = df['score']
# Creación del modelo2
# ==============================================================================
modelo2 = LinearRegression()
modelo2.fit(X = X, y = y)
R2_2=modelo2.score(X,y)
# Seleccionamos el valor de v3 y score para hacer regresion lineal simple y obtener R^2
#===============================================================================
X = df[['v3']]
y = df['score']
# Creación del modelo3
# ==============================================================================
modelo3 = LinearRegression()
modelo3.fit(X = X, y = y)
R2_3=modelo3.score(X,y)
# Correlación entre columnas numéricas
# ==============================================================================

def tidy_corr_matrix(corr_mat):

    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(10))
# Observamos una baja correlación entre score y las variables v1, v2 y v3
# (la más alta corresponde al 66% de los datos)

# Preparación de datos para regresión lineal múltiple
# ==============================================================================
X = df[['v1', 'v2', 'v3']]
y = df['score']

# Creación del modelo de regresión lineal múltiple
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de unos para el intercept del modelo
X = sm.add_constant(X, prepend=True)
modelo = sm.OLS(endog=y, exog=X,)
modelo = modelo.fit()
R2=modelo.rsquared
print('Correlación entre score y v1: ',R2_1**0.5)
print('Correlación entre score y v2: ',R2_2**0.5)
print('Correlación entre score y v3: ',R2_3**0.5)
print('Correlación entre score y las tres variables: ',R2**0.5)
#===============================================================================
# Como la correlación entre score y las tres variables con un modelo de regresión 
# lineal múltiple es alta, procedemos a cargar los jamones por calificar
df1 = pd.read_csv("jamones_por_calificar.csv")
#convertimos a tipo de dato numérico las columnas v1, v2, v3
df1[['v1','v2','v3']] = df1[['v1','v2','v3']].apply(pd.to_numeric)
X = df1[['v1', 'v2', 'v3']]
X = sm.add_constant(X, prepend=True)
#===============================================================================
# Se calcula el score mediante el modelo de regresión lineal múltiple y se adjunta
# al dataframe
prediccion = modelo.predict(exog = X)
prediccion=np.round(prediccion).astype(int)
df1['score']=prediccion
df1 = df1[['jamon','score','v1','v2','v3']]
#===============================================================================
# Se guardan los valores de las predicciones en el archivo jamones_por_calificar.csv
df1.to_csv('jamones_por_calificar.csv',index=False)