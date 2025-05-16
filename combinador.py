import yfinance as yf
import pandas as pd
import ta
import hashlib
import os
import sys
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

if len(sys.argv) < 2:
    print("❌ Debes indicar un ticker. Ejemplo: python combinador.py SPY")
    sys.exit(1)

TICKER = sys.argv[1]
RESULTS_FILE = f"data/resultados_{TICKER.replace('=','')}.csv"
n_iterations = 100

df = yf.download(TICKER, start="2015-01-01", end="2024-12-31", auto_adjust=False)
if df.empty:
    print(f"❌ No se pudo descargar datos para {TICKER}")
    sys.exit(1)

close = df["Close"].astype(float)
high = df["High"].astype(float)
low = df["Low"].astype(float)
open_ = df["Open"].astype(float)
volume = df["Volume"].astype(float)

df_ind = pd.DataFrame(index=df.index)
df_ind["sma"] = ta.trend.SMAIndicator(close=close, window=14).sma_indicator()
df_ind["ema"] = ta.trend.EMAIndicator(close=close, window=14).ema_indicator()
macd = ta.trend.MACD(close=close)
df_ind["macd"] = macd.macd()
df_ind["macd_signal"] = macd.macd_signal()
df_ind["macd_diff"] = macd.macd_diff()
df_ind["adx"] = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()
df_ind["cci"] = ta.trend.CCIIndicator(high=high, low=low, close=close).cci()
df_ind["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()
stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
df_ind["stoch"] = stoch.stoch()
df_ind["stoch_signal"] = stoch.stoch_signal()
df_ind["williams"] = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r()
df_ind["ao"] = ta.momentum.AwesomeOscillatorIndicator(high=high, low=low).awesome_oscillator()
bb = ta.volatility.BollingerBands(close=close)
df_ind["bb_bbm"] = bb.bollinger_mavg()
df_ind["bb_bbh"] = bb.bollinger_hband()
df_ind["bb_bbl"] = bb.bollinger_lband()
df_ind["atr"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()
df_ind["obv"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
df_ind["adi"] = ta.volume.AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()
df_ind["mfi"] = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()

df_ind["future_close"] = close.shift(-3)
df_ind["target"] = (df_ind["future_close"] > close).astype(int)
df_ind.dropna(inplace=True)

X = df_ind.drop(columns=["future_close", "target"])
y = df_ind["target"]
all_indicators = list(X.columns)

def hash_combination(indicators):
    combo_str = ",".join(sorted(indicators))
    return hashlib.md5(combo_str.encode()).hexdigest()

if os.path.exists(RESULTS_FILE):
    df_results = pd.read_csv(RESULTS_FILE)
    tried_hashes = set(df_results['hash'])
else:
    df_results = pd.DataFrame(columns=["hash", "indicadores", "accuracy", "precision", "recall", "f1_score"])
    tried_hashes = set()

new_results = []
attempts = 0
max_attempts = n_iterations * 10

while len(new_results) < n_iterations and attempts < max_attempts:
    attempts += 1
    k = random.randint(1, len(all_indicators))
    indicators = random.sample(all_indicators, k)
    combo_hash = hash_combination(indicators)

    if combo_hash in tried_hashes:
        continue

    try:
        X_subset = X[indicators]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        new_results.append({
            "hash": combo_hash,
            "indicadores": ",".join(indicators),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        tried_hashes.add(combo_hash)
    except Exception:
        continue

df_results = pd.concat([df_results, pd.DataFrame(new_results)], ignore_index=True)
df_results.to_csv(RESULTS_FILE, index=False)