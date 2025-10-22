# fetch_weekly_top5_binance.py
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE = "https://api.binance.com/api/v3/klines"  # Spot REST
SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
}
YEARS = 3
INTERVAL = "1d"
LIMIT = 1000
WEEK_FREQ = "W-MON"
TZ = "Asia/Tokyo"

HEADERS = {"User-Agent": "binance-weekly-fetcher/1.0"}

def fetch_klines(symbol: str, start_ms: int, end_ms: int):
    """
    Binance Klines: [ openTime, open, high, low, close, volume, closeTime,
      quoteAssetVolume, numberOfTrades, takerBuyBase, takerBuyQuote, ignore ]
    """
    out = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT, "startTime": cur, "endTime": end_ms}
        r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"{symbol} HTTP {r.status_code}: {r.text}")
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        # 次の開始時刻を最後のopenTimeの+1msへ
        cur = batch[-1][0] + 1
        # レート制限ゆるめ
        time.sleep(0.2)
    return out

def to_df(klines):
    cols = ["openTime","open","high","low","close","volume","closeTime","quoteVolume","nTrades","tbb","tbq","ignore"]
    df = pd.DataFrame(klines, columns=cols)
    df["open"]  = pd.to_numeric(df["open"], errors="coerce")
    df["high"]  = pd.to_numeric(df["high"], errors="coerce")
    df["low"]   = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")  # USDT換算の出来高を採用
    idx = pd.to_datetime(df["openTime"], unit="ms", utc=True).dt.tz_convert(TZ)
    df.index = idx
    return df[["open","high","low","close","volume"]].sort_index()

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    ohlc = df["close"].resample(WEEK_FREQ).agg(open="first", high="max", low="min", close="last")
    # open/high/lowは日次から再構築でもOK（closeベース簡便法）。厳密にやるなら:
    # ohlc = df.resample(WEEK_FREQ).agg({"open":"first","high":"max","low":"min","close":"last"})
    vol  = df["volume"].resample(WEEK_FREQ).sum(min_count=1)
    out = pd.concat([ohlc, vol], axis=1)
    out.index.name = "date"
    return out.dropna(subset=["open","high","low","close"])

def main():
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=365*YEARS)
    start_ms = int(since.timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)
    print(f"Fetching ~{YEARS}y daily klines from Binance and resampling to weekly...")

    for sym, symbol in SYMBOLS.items():
        try:
            print(f"  - {sym} ({symbol}) ... ", end="", flush=True)
            kl = fetch_klines(symbol, start_ms, end_ms)
            if not kl:
                print("no data"); continue
            df = to_df(kl)
            wk = to_weekly(df)
            out = f"{sym}_weekly.csv"
            wk.to_csv(out, float_format="%.10g")
            print(f"saved {out} ({len(wk)} rows)")
        except Exception as e:
            print(f"\n    ERROR for {sym}: {e}")

if __name__ == "__main__":
    main()
