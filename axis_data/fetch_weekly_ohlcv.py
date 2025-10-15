#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指定期間（--start ～ --end）で日足(1d)OHLCVを取引所(CCXT)から取得し、
週次(weekly)にリサンプルしてCSV出力。取れない銘柄はCoinGeckoでフォールバック。
- 週の起点は --weekstart（W-MON 等）で統一
- デフォルトの end は「実行時点の現在(UTC)」
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ccxt
import requests


# CoinGecko のコインID（必要な分だけ）
COINGECKO_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "SOL": "solana",
    "LINK": "chainlink",
    "ADA": "cardano",
    "TRX": "tron",
    "XLM": "stellar",
    "HYPE": "hyperliquid",  # Hyperliquid (上場薄 → CG フォールバック想定)
}

PREFERRED_QUOTES = ["USDT", "USD", "USDC"]  # 優先する見積通貨

# ===== 共通ユーティリティ =====

def to_utc_datetime(s: str) -> datetime:
    """YYYY-MM-DD を UTC の datetime に変換（00:00:00）"""
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)

def ensure_utc_index(df: pd.DataFrame, ts_col_ms: Optional[str] = None) -> pd.DataFrame:
    """ミリ秒タイムスタンプ列 or DatetimeIndex を UTC に統一"""
    if ts_col_ms:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df[ts_col_ms], unit="ms", utc=True)
        df = df.drop(columns=[ts_col_ms]).set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndexが見つかりません")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def clip_range(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    """start<=index<=end で日付範囲をクリップ"""
    return df.loc[(df.index >= start) & (df.index <= end)]

def resample_to_weekly(df_daily: pd.DataFrame, weekstart: str = "MON") -> pd.DataFrame:
    """
    日次OHLCV → 週次OHLCV（週起点はW-MON等）
    """
    if df_daily is None or df_daily.empty:
        raise ValueError("resample_to_weekly: 入力の日次データが空です")

    df = df_daily.sort_index()  # ← ここがポイント（df_dailyを使う）
    rule = f"W-{weekstart.upper()}"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    weekly = df.resample(rule, label="left", closed="left").agg(agg)
    # O/H/L/C のどれかが NaN の週を落とす（端の不完全週対策）
    weekly = weekly.dropna(subset=["open","high","low","close"])
    weekly = weekly.reset_index()  # timestamp列へ
    weekly.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return weekly


# ===== CCXT: 取引所から取得 =====

def find_market_symbol(ex: ccxt.Exchange, base: str) -> Optional[str]:
    """base/USDT, base/USD, base/USDC の順で最適ペアを探す"""
    base = base.upper()
    ex.load_markets()
    best = None
    best_rank = 999
    for s in ex.symbols:
        try:
            b, q = s.split("/")
        except ValueError:
            continue
        b = b.upper()
        q = q.upper().split(":")[0]  # 'USDT:USDT' → 'USDT'
        if b != base:
            continue
        if q in PREFERRED_QUOTES:
            rank = PREFERRED_QUOTES.index(q)
            if rank < best_rank:
                best = s
                best_rank = rank
    return best

def fetch_ohlcv_ccxt_paged(
    ex: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int, limit: int = 1000
) -> List[List[float]]:
    """
    CCXT の fetch_ohlcv をページングしながら end_ms に届くまで収集。
    timeframe: '1d' を推奨。取引所の rateLimit を尊重して sleep。
    """
    out: List[List[float]] = []
    cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol=symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        out.extend([row for row in batch if row[0] <= end_ms])
        last_ts = batch[-1][0]
        if last_ts >= end_ms or len(batch) < limit:
            break
        cursor = last_ts + 1  # 次ページへ
        time.sleep(getattr(ex, "rateLimit", 200) / 1000.0)
    return out

def fetch_daily_from_exchanges(
    exchanges: List[str], base: str, start: datetime, end: datetime, debug: bool = False
) -> Optional[Tuple[pd.DataFrame, str]]:
    """
    取引所リストを上から順に試し、最初に日足データを返せたものを採用。
    """
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    for ex_name in exchanges:
        try:
            ex = getattr(ccxt, ex_name)()
            ex.enableRateLimit = True
            symbol = find_market_symbol(ex, base)
            if not symbol:
                if debug: print(f"[debug] {ex_name}: {base} の取引ペアが見つからない")
                continue
            rows = fetch_ohlcv_ccxt_paged(ex, symbol, "1d", since_ms, end_ms, limit=1000)
            if not rows:
                if debug: print(f"[debug] {ex_name}: {base} 1d 取得0件")
                continue
            dfd = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
            dfd = ensure_utc_index(dfd, ts_col_ms="ts")
            dfd = dfd[["open","high","low","close","volume"]]
            dfd = clip_range(dfd, start, end)
            if dfd.empty:
                continue
            return dfd, ex_name
        except Exception as e:
            if debug: print(f"[debug] {ex_name}: {base} 取得失敗: {e}")
            continue
    return None

# ===== CoinGecko フォールバック（日次→週次） =====

def fetch_daily_from_coingecko_range(coin_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    CoinGecko: /ohlc?days=N と /market_chart?days=N を併用し、日次OHLCVを構成。
    N は start から end までに十分な日数にマージン(+10日)を足して計算。
    """
    base = "https://api.coingecko.com/api/v3"
    span_days = max(1, ceil((end - start).days) + 10)
    # 1) OHLC
    r1 = requests.get(f"{base}/coins/{coin_id}/ohlc",
                      params={"vs_currency": "usd", "days": str(span_days)}, timeout=30)
    r1.raise_for_status()
    ohlc = r1.json()
    if not ohlc:
        raise RuntimeError("coingecko: ohlc 空")
    ohlc_df = pd.DataFrame(ohlc, columns=["ts","open","high","low","close"])
    ohlc_df["timestamp"] = pd.to_datetime(ohlc_df["ts"], unit="ms", utc=True)
    ohlc_df = ohlc_df.drop(columns=["ts"]).set_index("timestamp")

    # 2) volume
    r2 = requests.get(f"{base}/coins/{coin_id}/market_chart",
                      params={"vs_currency": "usd", "days": str(span_days), "interval":"daily"}, timeout=30)
    r2.raise_for_status()
    vols = r2.json().get("total_volumes", [])
    vol_df = pd.DataFrame(vols, columns=["timestamp","volume"]).set_index("timestamp")
    vol_df.index = pd.to_datetime(vol_df.index, unit="ms", utc=True)

    df = ohlc_df.join(vol_df, how="left")
    df["volume"] = df["volume"].fillna(0)
    df = clip_range(df, start, end)
    if df.empty:
        raise RuntimeError("coingecko: 範囲内データなし")
    return df[["open","high","low","close","volume"]]

# ===== メイン処理 =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="data", help="CSV出力先")
    parser.add_argument("--start", required=True, help="開始日 YYYY-MM-DD（例: 2024-12-02）")
    parser.add_argument("--end", default=None, help="終了日 YYYY-MM-DD（省略時は現在UTC）")
    parser.add_argument("--weekstart", default="MON",
                        choices=["MON","TUE","WED","THU","FRI","SAT","SUN"],
                        help="週の起点（W-XXX）")
    parser.add_argument("--exchanges", default="binance,okx,bybit,kraken,coinbase,bitfinex",
                        help="優先順にカンマ区切り")
    parser.add_argument("--symbols", default="BTC,ETH,BNB,XRP,SOL,LINK,ADA,TRX,XLM,HYPE",
                        help="取得する銘柄（カンマ区切り）")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    start_dt = to_utc_datetime(args.start)
    end_dt = to_utc_datetime(args.end) if args.end else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise ValueError("end は start より後にしてください")

    exchanges = [e.strip() for e in args.exchanges.split(",") if e.strip()]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    all_rows: List[pd.DataFrame] = []
    for sym in symbols:
        print(f"=== {sym} ===")
        # 1) 取引所から
        got = fetch_daily_from_exchanges(exchanges, sym, start_dt, end_dt, debug=args.debug)
        if got:
            dfd, used_ex = got
            weekly = resample_to_weekly(dfd, weekstart=args.weekstart)
            weekly.insert(0, "symbol", sym)
            out_path = os.path.join(args.outdir, f"{sym}_weekly.csv")
            weekly.to_csv(out_path, index=False)
            print(f"[ok] {sym}: {len(weekly)} rows ({used_ex} 1d→weekly) -> {out_path}")
            all_rows.append(weekly)
            continue

        # 2) CoinGecko フォールバック
        cg_id = COINGECKO_IDS.get(sym)
        if not cg_id:
            print(f"[skip] {sym}: 取引所取得失敗 & CoinGecko ID 未設定")
            continue

        try:
            dfd = fetch_daily_from_coingecko_range(cg_id, start_dt, end_dt)
            weekly = resample_to_weekly(dfd, weekstart=args.weekstart)
            weekly.insert(0, "symbol", sym)
            out_path = os.path.join(args.outdir, f"{sym}_weekly.csv")
            weekly.to_csv(out_path, index=False)
            print(f"[ok] {sym}: {len(weekly)} rows (coingecko 1d→weekly) -> {out_path}")
            all_rows.append(weekly)
        except Exception as e:
            print(f"[error] {sym}: CoinGecko 失敗: {e}")

    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True).sort_values(["symbol","timestamp"])
        all_df.to_csv(os.path.join(args.outdir, "all_symbols_weekly.csv"), index=False)
        print(f"[ok] all_symbols_weekly.csv -> {len(all_df)} rows")
    else:
        print("[warn] 出力ゼロでした")

if __name__ == "__main__":
    main()
