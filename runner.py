import os
import time
from tickers import tickers

for ticker in tickers:
    print(f"🔁 Ejecutando combinador para {ticker}")
    os.system(f"python combinador.py {ticker}")
    time.sleep(2)