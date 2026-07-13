import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Importamos las funciones del nuevo motor independiente
from engine_rebalance import generar_datos_frontera, calcular_impacto_rebalanceo
from data_engine import descargar_datos_portafolio
from math_engine import calcular_retornos_y_covarianza
from optimization_engine import optimizar_max_sharpe

st.set_page_config(page_title='Dynamic Portfolio & Rebalance Engine', page_icon='🔄', layout='wide')
st.title('🔄 Dynamic Portfolio & Rebalance Engine')
st.subheader('Simulador de Desviación de Capital, Curvas de Eficiencia y Costos de Transacción')
st.markdown('---')

# --- SIDEBAR DE ENTRADA ---
st.sidebar.header('⚙️ Configuración de la Cartera Actual')
tickers_input = st.sidebar.text_input('Tickers del Portafolio:', value='AAPL, MSFT, META, GOOG')
capital_total = st.sidebar.number_input('Capital Total Invertido (USD):', value=10000.0, step=1000.0)

st.sidebar.markdown('---')
st.sidebar.subheader('📈 Tus Pesos Actuales (Deben sumar 100%)')

lista_tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

pesos_actuales = []
for ticker in lista_tickers:
    peso = st.sidebar.slider(f'Peso actual para {ticker} (%)', min_value=0.0, max_value=100.0, value=25.0, step=5.0)
    pesos_actuales.append(peso / 100.0)

boton_analizar = st.sidebar.button('⚙️ Simular Diagnóstico y Rebalanceo')

if boton_analizar:
    sum_pesos = sum(pesos_actuales)
    if not np.isclose(sum_pesos, 1.0):
        st.error(f'⚠️ Error en la configuración: La suma de tus pesos actuales es del {sum_pesos*100:.1f}%. Debe ser exactamente igual al 100%.')
    else:
        with st.spinner('Analizando desviaciones en la Frontera Eficiente y calculando fricción fiscal...'):
            try:
                # Reutilizamos tus robustos motores de datos anteriores
                df_precios = descargar_datos_portafolio(lista_tickers, periodo='5y')
                ret_diarios, ret_esperados, mat_cov = calcular_retornos_y_covarianza(df_precios)
                
                # Encontramos el punto teórico óptimo actual
                resultado_opt = optimizar_max_sharpe(ret_esperados, mat_cov)
                pesos_optimos = resultado_opt["pesos"]
                
                # Calculamos las métricas de la cartera actual del usuario
                ret_actual = np.sum(pesos_actuales * ret_esperados)
                vol_actual = np.sqrt(np.dot(np.array(pesos_actuales).T, np.dot(mat_cov, pesos_actuales)))
                sharpe_actual = ret_actual / vol_actual if vol_actual != 0 else 0
                
                # --- VISUALIZACIÓN DE LA FRONTERA EFICIENTE DINÁMICA ---
                st.subheader('📈 Mapeo Espacial de Eficiencia y Posicionamiento Colectivo')
                
                # Nube de puntos
                datos_curva = generar_datos_frontera(ret_esperados, mat_cov)
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=datos_curva[0, :], y=datos_curva[1, :] * 100,
                    mode='markers',
                    marker=dict(size=5, color=datos_curva[2, :], colorscale='Viridis', showscale=True, colorbar=dict(title="Ratio de Sharpe")),
                    name='Alternativas de Mercado'
                ))
                
                # Punto óptimo (Max Sharpe)
                fig.add_trace(go.Scatter(
                    x=[resultado_opt['volatilidad']], y=[resultado_opt['retorno'] * 100],
                    mode='markers', marker=dict(color='emerald', size=14, symbol='star', line=dict(color='black', width=1.5)),
                    name='Objetivo Eficiente (Max Sharpe)'
                ))
                
                # Punto donde está parado el usuario actualmente (Parpadeante simulado)
                fig.add_trace(go.Scatter(
                    x=[vol_actual], y=[ret_actual * 100],
                    mode='markers', marker=dict(color='red', size=15, symbol='circle', line=dict(color='white', width=2)),
                    name='Tu Cartera Actual (Desviada)'
                ))
                
                fig.update_layout(xaxis_title="Riesgo (Volatilidad Anual)", yaxis_title="Retorno Anual Esperado (%)", template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- ANÁLISIS DE COSTOS DE REBALANCEO ---
                st.markdown('---')
                st.subheader('📊 Diagnóstico de Rebalanceo y Fricción de Costos')
                
                vol_negociado, costo_trans = calcular_impacto_rebalanceo(pesos_actuales, pesos_optimos, capital_total)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Pérdida de Eficiencia (Sharpe)", f"-{resultado_opt['sharpe'] - sharpe_actual:.2f}", help="Diferencia de eficiencia entre el óptimo y tu cartera actual")
                c2.metric("Volumen de Capital a Mover", f"USD {vol_negociado:,.2f}")
                c3.metric("Costo de Transacción Est. (0.2%)", f"USD {costo_trans:,.2f}")
                
                # Sugerencia de periodicidad
                st.info("💡 **Frecuencia Recomendada de Monitoreo:** Debido a que el costo estimado de rebalanceo es bajo respecto al capital total, se sugiere un ajuste de tipo **Trimestral**. Esto permite que los activos respiren sin erosionar las ganancias por exceso de comisiones de corretaje.")
                
            except Exception as e:
                st.error(f"Error en la simulación dinámica: {e}")
