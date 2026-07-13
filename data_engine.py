import yfinance as yf
import pandas as pd

def descargar_datos_portafolio(tickers, periodo='5y'):
    """
    Descarga los precios históricos de cierre ajustados desde Yahoo Finance.
    """
    datos = yf.download(tickers, period=periodo)
    if isinstance(datos.columns, pd.MultiIndex):
        # Si devuelve un MultiIndex (versiones nuevas de yfinance), extraemos Adj Close o Close
        if 'Adj Close' in datos.columns.levels[0]:
            df_precios = datos['Adj Close']
        else:
            df_precios = datos['Close']
    else:
        df_precios = datos['Adj Close'] if 'Adj Close' in datos.columns else datos['Close']
        
    # Eliminar filas con valores nulos para evitar errores matemáticos
    df_precios = df_precios.dropna()
    return df_precios
