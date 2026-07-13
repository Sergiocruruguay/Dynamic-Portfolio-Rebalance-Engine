import numpy as np
import pandas as pd

def calcular_retornos_y_covarianza(df_precios):
    """
    Calcula los retornos diarios logarítmicos/porcentuales y las matrices estadísticas.
    """
    retornos_diarios = df_precios.pct_change().dropna()
    retornos_esperados = retornos_diarios.mean() * 252
    matriz_covarianza = retornos_diarios.cov() * 252
    return retornos_diarios, retornos_esperados, matriz_covarianza
