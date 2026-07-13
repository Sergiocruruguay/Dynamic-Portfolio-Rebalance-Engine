import numpy as np

def generar_datos_frontera(ret_esperados, mat_cov, num_portafolios=800):
    """
    Genera cientos de combinaciones de carteras para mapear la curva completa.
    """
    num_activos = len(ret_esperados)
    resultados = np.zeros((3, num_portafolios))
    
    for i in range(num_portafolios):
        pesos = np.random.random(num_activos)
        pesos /= np.sum(pesos)
        
        p_retorno = np.sum(pesos * ret_esperados)
        p_volatilidad = np.sqrt(np.dot(pesos.T, np.dot(mat_cov, pesos)))
        p_sharpe = p_retorno / p_volatilidad if p_volatilidad != 0 else 0
        
        resultados[0, i] = p_volatilidad
        resultados[1, i] = p_retorno
        resultados[2, i] = p_sharpe
        
    return resultados

def calcular_impacto_rebalanceo(pesos_actuales, pesos_objetivo, capital_total, tasa_comision=0.002):
    """
    Calcula los costos de transacción simulados al mover el capital
    desde la foto actual hacia el portafolio óptimo ideal.
    """
    # Dinero absoluto actualmente en cada activo
    capital_actual_activos = capital_total * np.array(pesos_actuales)
    # Dinero absoluto que debería haber
    capital_objetivo_activos = capital_total * np.array(pesos_objetivo)
    
    # El volumen total negociado (compras y ventas) es la suma de las diferencias absolutas
    volumen_negociado = np.sum(np.abs(capital_actual_activos - capital_objetivo_activos))
    
    # Costo estimado (por ejemplo, 0.2% de comisión por operación)
    costo_transaccion = volumen_negociado * tasa_comision
    
    return volumen_negociado, costo_transaccion
