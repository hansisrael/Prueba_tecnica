# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 10:54:24 2021

@author: tequi
"""
# Tratamiento de datos
# ==============================================================================
import pandas as pd
# ==============================================================================
# Cargamos el dataset
veneno = pd.read_csv('veneno.csv')
sustancias = pd.read_csv('sustancias_diversas.csv')
#===============================================================================
# Extraemos los umbrales para las 10 características
veneno = veneno.drop(['caracteristica'],axis=1)
veneno=veneno.loc[0]
veneno=pd.DataFrame(veneno)
veneno=veneno[0].tolist()
# ==============================================================================
# Normalizamos las columnas con el valor del umbral. Los valores por debajo del umbral
# no se consideran peligrosos, por los que se les asigna un valor de cero
def normalizar(columna):
    normalizado=columna/veneno[j]
    if normalizado<1:
        normalizado=0
    return normalizado
j=0
for i in ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']:
    sustancias[i]=sustancias[i].apply(normalizar)
    j+=1
# ==============================================================================
# Con base en los datos normalizados, se da una puntuación a cada sustancia. A
# mayor concentración de una sustancia, mayor puntuación
def score(datos):
    puntuacion=0
    for i in ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']:
        puntuacion+=datos[i]
    return puntuacion
sustancias['score_venenosidad']=sustancias.apply(score,axis=1)
# ==============================================================================
# Se ordenan las sustancias con base en el score obtenido. A mayor puntuación,
# tiene una prioridad más alta para ser cerrado y se seleccionan las 50
# sustancias con el orden más alto
sustancias=sustancias.sort_values('score_venenosidad',ascending=False)
sustancias=sustancias[0:50]
sustancias=sustancias[['id']]
sustancias.to_csv('urgente_orden_de_cierre.csv',index=False)
