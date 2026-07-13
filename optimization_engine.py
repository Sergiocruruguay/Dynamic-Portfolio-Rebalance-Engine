import numpy as np
import scipy.optimize as sco

def estadísticas_portafolio(pesos, ret_esperados, mat_cov):
    pesos = np.array(pesos)
    retorno = np.sum(pesos * ret_esperados)
    volatilidad = np.sqrt(np.dot(pesos.T, np.dot(mat_cov, pesos)))
    sharpe = retorno / volatilidad if volatilidad != 0 else 0
    return {"retorno": retorno, "volatilidad": volatilidad, "sharpe": sharpe}

def optimizar_max_sharpe(ret_esperados, mat_cov):
    num_activos = len(ret_esperados)
    args = (ret_esperados, mat_cov)
    
    # Función objetivo a minimizar (el Sharpe negativo)
    def min_func_sharpe(pesos):
        return -estadísticas_portafolio(pesos, ret_esperados, mat_cov)["sharpe"]
        
    # Restricciones: la suma de los pesos debe ser igual a 1
    restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Límites: cada peso individual entre 0.0 y 1.0
    limites = tuple((0, 1) for _ in range(num_activos))
    # Inicialización equitativa
    inicial = num_activos * [1.0 / num_activos]
    
    optimizacion = sco.minimize(min_func_sharpe, inicial, method='SLSQP', bounds=limites, constraints=restricciones)
    
    pesos_opt = optimizacion.x
    metricas = estadísticas_portafolio(pesos_opt, ret_esperados, mat_cov)
    
    return {"pesos": pesos_opt, "retorno": metricas["retorno"], "volatilidad": metricas["volatilidad"], "sharpe": metricas["sharpe"]}
