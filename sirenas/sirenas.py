# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:27:10 2021

@author: tequi
"""

# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Modelado
# ==============================================================================

from sklearn.neural_network import MLPClassifier

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
# ==============================================================================
# Cargamos el dataset
df = pd.read_csv("sirenas_endemicas_y_sirenas_migrantes_historico.csv")
#convertimos a tipo de dato numérico las columnas v1, v2, v3 y v4
df[['v1','v2','v3','v4']] = df[['v1','v2','v3','v4']].apply(pd.to_numeric)
#extraemos las columnas de características
X = df.iloc[0:99, [0,1,2,3]].values
#extraemos etiquetas, a las sirenas migrantes se les asigna el valor de 0 y a las endémicas el valor de 1
y = df.iloc[0:99, 4].values
y = np.where(y == 'sirena_migrante', 0, 1)
# Correlación entre columnas numéricas
# ==============================================================================

def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = df.select_dtypes(include=['float64', 'int64','int']).corr(method='pearson')
print(tidy_corr_matrix(corr_matrix).head(10))
# Observamos que la máxima correlación se encuentra entre v3 y v4
#==============================================================================
# Graficamos v3 vs v4
fig, ax = plt.subplots(1, 1)
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 2],
        y = X[y == i, 3], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('v3 vs v4')
ax.legend();
# Observamos que las características discriminantes son las características v3 y v4
#===============================================================================
# Leemos el dataset de sirenas por clasificar
df1 = pd.read_csv("sirenas_endemicas_y_sirenas_migrantes.csv")
df1[['v1','v2','v3','v4']] = df[['v1','v2','v3','v4']].apply(pd.to_numeric)
#===============================================================================
# Establecemos umbrales para clasificación con v3
v3up=df[df['especie']=='sirena_endemica']
v3up=v3up['v3'].mean()
v3down=df[df['especie']=='sirena_migrante']
v3down=v3down['v3'].mean()
uv3=(v3up+v3down)/2
# Establecemos umbrales para clasificación con v4
v4up=df[df['especie']=='sirena_endemica']
v4up=v4up['v4'].mean()
v4down=df[df['especie']=='sirena_migrante']
v4down=v4down['v4'].mean()
uv4=(v4up+v4down)/2
#===============================================================================
# Agregamos columnas de clasificación al dataframe usando como clasificadores los umbrales
# en v3 y v4
def etiquetasv3(caracteristica):
    if caracteristica >= uv3:
        etiqueta='sirena_endemica'
    else:
        etiqueta='sirena_migrante'
    return etiqueta
df1['especie_v3']=df1['v3'].apply(etiquetasv3)
def etiquetasv4(caracteristica):
    if caracteristica >= uv4:
        etiqueta='sirena_endemica'
    else:
        etiqueta='sirena_migrante'
    return etiqueta
df1['especie_v4']=df1['v4'].apply(etiquetasv4)
# Se comprueba con un modelo de perceptron multicapa con 10 neuronas en la capa oculta
# que los clasificadores anteriores funcionen correctamente
# Modelo de perceptron multicapa
# ==============================================================================
modelo = MLPClassifier(
                hidden_layer_sizes=(10),
                learning_rate_init=0.01,
                solver = 'lbfgs',
                max_iter = 1000,
                random_state = 123
            )
modelo.fit(X=X, y=y)
pred=modelo.predict(df1[['v1','v2','v3','v4']])
# Creamos un array de strings para posteriormente llenarlos con las predicciones
pred2=[]
for i in range(len(pred)):
    pred2.append(['a'])
for i in range(0,len(pred)):
    if pred[i]==0:
        pred2[i]='sirena_migrante'
    else:
        pred2[i]='sirena_endemica'
df1['especie']=pred2
#===============================================================================
# Se guardan los valores de las predicciones en dos archivos, uno llamado prueba.csv
# (donde estan las predicciones con los tres clasificadores), y otro llamado
# sirenas_endemicas_y_sirenas_migrantes.csv, donde se encuentra la clasificación
df1 = df1[['v1','v2','v3','v4','especie','especie_v3','especie_v4']]
df1.to_csv('prueba.csv',index=False)
df1=df1.drop(['especie_v3','especie_v4'],axis=1)
df1.to_csv('sirenas_endemicas_y_sirenas_migrantes.csv',index=False)