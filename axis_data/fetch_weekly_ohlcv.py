#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CoinCap v3 Free → 日次履歴を取得 → 週足CSVにキャッシュ → 3モデル(AXIS_IVW/MCW_Cap40/SOL_Hold)で週足BT
- 認証: 環境変数 COINCAP_API_KEY（Bearer 必須）
- 週足: W-SUN（その週の最終観測値）
- リバランス: 各四半期の最初の週
- ボラ窓: 90日 ≒ 13週
- 取引コスト: 0
- 差分更新ユーティリティ同梱（--update --lookback-days）

実行例:
  export COINCAP_API_KEY="YOUR_TOKEN"
  pip install pandas numpy matplotlib requests
  python backtest_weekly_coincap.py            # 初回フル取得→週足CSV→BT
  python backtest_weekly_coincap.py --update   # 直近120日だけ差分取得→CSV更新→BT
  python backtest_weekly_coincap.py --update --lookback-days 200
"""

from __future__ import annotations
import os
import time
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import argparse

# =========================
# ユーザー設定
# =========================
# 期間（週足出力はこの範囲でクリップ）
START_DATE = "2020-11-01"
END_DATE   = "2025-10-31"

# 週足CSV（<SYM>_Price, <SYM>_Market_Cap）を保存/再利用
CSV_PATH = Path("crypto_data_5y.csv")

# バックテスト設定
INITIAL_CAPITAL   = 1000.0
WEEK_RULE         = "W-SUN"     # 週末=日曜
VOL_WINDOW_WEEKS  = 13          # 90日 ≒ 13週
CAP_LIMIT         = 0.40        # MCWの上限40%
TOP_N             = 5
BENCH_SYMBOL      = "SOL"       # ベンチマーク銘柄（必須）

# 取得する銘柄（CoinCapの slug）
ASSETS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binance-coin",
    "XRP": "xrp",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "TRX": "tron",
    "DOT": "polkadot",
}

# CoinCap v3 認証（Bearer 必須）
COINCAP_API_KEY = os.getenv("COINCAP_API_KEY", "").strip()
if not COINCAP_API_KEY:
    raise RuntimeError("COINCAP_API_KEY が未設定です。CoinCap v3 は全エンドポイントで Bearer が必須です。")

# 安定運用パラメータ
REQUEST_PAUSE_SEC = 0.25          # 取得間隔
CHUNK_BY_YEAR     = True          # 年単位に分割して取得
RETRY_TIMES       = 3             # リトライ回数
RETRY_SLEEP_SEC   = 1.0

# ---（オプション）時価総額を“固定供給×価格”で補完したい場合（研究用近似） ---
USE_FIXED_SUPPLY_FALLBACK = True  # Trueなら market cap 欠損を固定供給で埋める
FIXED_SUPPLY = {
    "BTC": 19_500_000,
    "ETH": 120_000_000,
    "SOL": 580_000_000,
    "BNB": 147_000_000,
    "XRP": 53_000_000_000,
    "ADA": 35_000_000_000,
    "DOGE": 144_000_000_000,
    "AVAX": 450_000_000,
    "TRX": 88_000_000_000,
    "DOT": 1_300_000_000,
}
# -----------------------------------------------------------------------

# =========================
# 取得ユーティリティ（CoinCap v3）
# =========================
def coincap_headers_v3() -> Dict[str, str]:
    return {"Authorization": f"Bearer {COINCAP_API_KEY}"}

def to_unix_ms(date_str: str) -> int:
    return int(pd.to_datetime(date_str, utc=True).timestamp() * 1000)

def fetch_coincap_history_v3_once(slug: str, start_date: str, end_date: str, interval: str = "d1") -> pd.DataFrame:
    """
    1チャンク分を取得（v3）。market cap カラムが無い場合にも対応。
    戻り: index=Date, cols=[priceUsd, marketCapUsd]
    """
    url = f"https://rest.coincap.io/v3/assets/{slug}/history"
    params = {
        "interval": interval,
        "start": str(to_unix_ms(start_date)),
        "end":   str(to_unix_ms(end_date) + 86_399_000),
        "fields": "time,priceUsd,marketCapUsd,marketCap,supply,price"  # 取れるなら明示要求
    }
    for attempt in range(1, RETRY_TIMES + 1):
        try:
            r = requests.get(url, params=params, headers=coincap_headers_v3(), timeout=30)
            if r.status_code == 404:
                # ←← ここが追加：このチャンクはデータ無しとしてスキップ
                print(f"  -> {slug}: {start_date}~{end_date} はデータ404（スキップ）")
                return pd.DataFrame()
            r.raise_for_status()
            
            data = r.json().get("data", [])
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # --- Date 整形 ---
            if "time" in df.columns:
                df["Date"] = pd.to_datetime(df["time"], unit="ms", errors="coerce") \
                                .dt.tz_localize(None).dt.normalize()
            elif "date" in df.columns:
                df["Date"] = pd.to_datetime(df["date"], errors="coerce") \
                                .dt.tz_localize(None).dt.normalize()
            else:
                return pd.DataFrame()

            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

            # --- 数値化 ---
            for c in df.columns:
                if c.lower() in ("priceusd", "marketcapusd", "marketcap", "supply", "price"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # --- price 列 ---
            price_col = None
            for cand in ["priceUsd", "price", "close"]:
                if cand in df.columns:
                    price_col = cand
                    break
            if price_col is None:
                return pd.DataFrame()

            # --- market cap 列（優先順位: marketCapUsd > marketCap > price*supply > NaN） ---
            if "marketCapUsd" in df.columns:
                mcap = df["marketCapUsd"]
            elif "marketCap" in df.columns:
                mcap = df["marketCap"]
            elif "supply" in df.columns:
                mcap = df[price_col] * df["supply"]
            else:
                mcap = pd.Series(np.nan, index=df.index, name="marketCapUsd")

            out = pd.DataFrame({
                "priceUsd": df[price_col],
                "marketCapUsd": mcap
            }, index=df.index).groupby(level=0).last()

            return out
        except Exception:
            if attempt == RETRY_TIMES:
                raise
            time.sleep(RETRY_SLEEP_SEC)

def split_year_chunks(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    start = pd.to_datetime(start_date).normalize()
    end   = pd.to_datetime(end_date).normalize()
    cur_start = start
    chunks = []
    while cur_start <= end:
        cur_end = min(pd.Timestamp(year=cur_start.year, month=12, day=31), end)
        chunks.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
        cur_start = (cur_end + pd.Timedelta(days=1)).normalize()
    return chunks

def fetch_coincap_history_v3(slug: str, start_date: str, end_date: str, interval: str = "d1") -> pd.DataFrame:
    if CHUNK_BY_YEAR:
        chunks = split_year_chunks(start_date, end_date)
    else:
        chunks = [(start_date, end_date)]
    frames = []
    for (s, e) in chunks:
        df = fetch_coincap_history_v3_once(slug, s, e, interval=interval)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(REQUEST_PAUSE_SEC)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0).sort_index()
    out = out.groupby(level=0).last()
    return out.loc[start_date:end_date]

def fetch_universe_daily(assets_map: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    各銘柄の priceUsd / marketCapUsd を取得して日次で外部結合、欠損はFFILL
    """
    frames = []
    for i, (sym, slug) in enumerate(assets_map.items(), 1):
        print(f"[Fetch] ({i}/{len(assets_map)}) {sym} <- {slug}")
        df = fetch_coincap_history_v3(slug, start_date, end_date, interval="d1")
        if df.empty:
            print(f"  -> {sym}: データなし（スキップ）")
            continue
        # 固定供給で補完（必要時のみ）
        if USE_FIXED_SUPPLY_FALLBACK:
            if "marketCapUsd" in df.columns:
                nan_mask = df["marketCapUsd"].isna()
            else:
                nan_mask = pd.Series(True, index=df.index)
            if nan_mask.any():
                supply = FIXED_SUPPLY.get(sym)
                if supply is not None and "priceUsd" in df.columns:
                    df.loc[nan_mask, "marketCapUsd"] = df.loc[nan_mask, "priceUsd"] * float(supply)

        df = df.rename(columns={"priceUsd": f"{sym}_Price", "marketCapUsd": f"{sym}_Market_Cap"})
        frames.append(df)

    if not frames:
        raise ValueError("CoinCapから有効なデータを取得できませんでした。APIキー/期間/slugを確認してください。")
    daily = pd.concat(frames, axis=1).sort_index().ffill()
    daily.index.name = "Date"
    return daily.loc[start_date:end_date]

def resample_daily_to_weekly(daily: pd.DataFrame, rule: str = WEEK_RULE) -> pd.DataFrame:
    """
    日次→週足（その週の最終値＝終値/最後のmarket cap）に変換
    """
    weekly = daily.resample(rule).last().dropna(how="all")
    weekly.index.name = "Date"
    return weekly

def save_weekly_csv_from_coincap(csv_path: Path, assets_map: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    CoinCap v3 から日次取得 → 週足化 → CSV保存
    """
    print("[Fetch] CoinCap v3（日次）→ 週足へリサンプリング → CSV 保存")
    daily = fetch_universe_daily(assets_map, start_date, end_date)
    weekly = resample_daily_to_weekly(daily, WEEK_RULE)
    weekly.to_csv(csv_path, index_label="Date")
    print(f"[Save] 週足CSV: {csv_path.resolve()}")
    return weekly

# =========================
# 差分更新ユーティリティ
# =========================
def merge_weekly_csv(base: pd.DataFrame, incr: pd.DataFrame) -> pd.DataFrame:
    """週足DataFrame同士をマージしてDate重複を最後で上書き"""
    out = pd.concat([base, incr], axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out

def update_weekly_csv(csv_path: Path, assets_map: Dict[str, str], lookback_days: int = 120):
    """
    既存の週足CSVに、直近lookback_daysだけ再取得してマージ。
    """
    if not csv_path.exists():
        return save_weekly_csv_from_coincap(csv_path, assets_map, START_DATE, END_DATE)

    base = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date").sort_index()

    since = (pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    start = max(pd.to_datetime(START_DATE), pd.to_datetime(since)).strftime("%Y-%m-%d")
    recent_daily = fetch_universe_daily(assets_map, start, END_DATE)
    recent_weekly = resample_daily_to_weekly(recent_daily, WEEK_RULE)

    updated = merge_weekly_csv(base, recent_weekly)
    updated = updated.loc[START_DATE:END_DATE]
    updated.to_csv(csv_path, index_label="Date")
    print(f"[Update] 週足CSVを差分更新: {csv_path.resolve()}  行数={len(updated)}")
    return updated

# =========================
# バックテスト（週足）
# =========================
def next_index_at_or_after(target_ts: pd.Timestamp, index_values: np.ndarray):
    loc = index_values.searchsorted(target_ts)
    if loc < len(index_values):
        return index_values[loc]
    return None

def cap_and_redistribute(weights: pd.Series, cap: float = 0.40, tol: float = 1e-12, max_iter: int = 100) -> pd.Series:
    w = weights.copy().astype(float)
    if w.sum() <= 0:
        positives = w[w > 0]
        return positives / positives.sum() if len(positives) > 0 else w
    w /= w.sum()
    for _ in range(max_iter):
        over = w > cap + tol
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = w < cap - tol
        if not under.any():
            w += excess / len(w)
            w[w < 0] = 0
            if w.sum() > 0:
                w /= w.sum()
            continue
        headroom = cap - w[under]
        alloc = headroom / headroom.sum() * excess
        w[under] += alloc
    w[w < 0] = 0
    if w.sum() > 0:
        w /= w.sum()
    return w

def compute_inverse_vol_weights(returns_df: pd.DataFrame, candidates: List[str], asof_date: pd.Timestamp, window_weeks: int = 13) -> pd.Series:
    if asof_date not in returns_df.index:
        return pd.Series(dtype=float)
    end_loc = returns_df.index.get_loc(asof_date)
    if isinstance(end_loc, slice):
        end_loc = end_loc.start
    start_loc = max(0, end_loc - window_weeks)
    window_slice = returns_df.iloc[start_loc:end_loc]
    vols = window_slice[candidates].std(skipna=True)
    vols = vols.replace([np.inf, -np.inf], np.nan).dropna()
    if vols.empty:
        return pd.Series(dtype=float)
    inv_vol = 1.0 / vols
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if inv_vol.empty:
        return pd.Series(dtype=float)
    return inv_vol / inv_vol.sum()

def calculate_drawdown(values: pd.Series) -> pd.Series:
    peak = values.cummax()
    return values / peak - 1.0

def run_backtest_weekly(weekly: pd.DataFrame):
    # 必須チェック
    price_cols = [c for c in weekly.columns if c.endswith("_Price")]
    mc_cols    = [c for c in weekly.columns if c.endswith("_Market_Cap")]
    if not price_cols or not mc_cols:
        raise ValueError("CSVには <SYM>_Price / <SYM>_Market_Cap 列が必要です。")
    if f"{BENCH_SYMBOL}_Price" not in weekly.columns or f"{BENCH_SYMBOL}_Market_Cap" not in weekly.columns:
        raise ValueError(f"ベンチマーク用に {BENCH_SYMBOL}_Price / {BENCH_SYMBOL}_Market_Cap が必要です。")

    # シンボル集合
    def sym_from(col, suffix): return col[: -len(suffix)]
    assets_price = [sym_from(c, "_Price") for c in price_cols]
    assets_mc    = [sym_from(c, "_Market_Cap") for c in mc_cols]
    universe     = sorted(set(assets_price).intersection(assets_mc))

    prices  = weekly[[f"{s}_Price" for s in universe]].rename(columns={f"{s}_Price": s for s in universe})
    mktcaps = weekly[[f"{s}_Market_Cap" for s in universe]].rename(columns={f"{s}_Market_Cap": s for s in universe})

    # 週次リターン
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 四半期の最初の週
    quarter_starts = pd.date_range(start=START_DATE, end=END_DATE, freq="QS-JAN")
    index_vals = prices.index.values
    rebalance_weeks = []
    for d in quarter_starts:
        nxt = next_index_at_or_after(d, index_vals)
        if nxt is not None:
            rebalance_weeks.append(pd.Timestamp(nxt))
    rebalance_weeks = sorted(set(rebalance_weeks))
    if not rebalance_weeks:
        raise ValueError("リバランス週が見つかりません。")

    # 曲線＆ポジション
    axis_val = pd.Series(index=prices.index, dtype=float)
    mcw_val  = pd.Series(index=prices.index, dtype=float)
    sol_val  = pd.Series(index=prices.index, dtype=float)

    axis_pos = pd.Series(0.0, index=universe, dtype=float)
    mcw_pos  = pd.Series(0.0, index=universe, dtype=float)

    first_week = prices.index[0]
    axis_val.iloc[0] = INITIAL_CAPITAL
    mcw_val.iloc[0]  = INITIAL_CAPITAL

    sol_units = INITIAL_CAPITAL / prices.at[first_week, BENCH_SYMBOL]
    sol_val.iloc[0] = sol_units * prices.at[first_week, BENCH_SYMBOL]

    if first_week not in rebalance_weeks:
        rebalance_weeks = [first_week] + rebalance_weeks

    # 週次ループ
    for i, t in enumerate(prices.index):
        if t in rebalance_weeks:
            mc_today = mktcaps.loc[t].replace([np.inf, -np.inf], np.nan).dropna()
            top = mc_today.sort_values(ascending=False).head(TOP_N).index.tolist()

            # AXIS: Inverse Volatility Weight
            ivw = compute_inverse_vol_weights(rets, top, t, window_weeks=VOL_WINDOW_WEEKS)
            if ivw.empty:
                avail = [s for s in top if s in rets.columns]
                ivw = pd.Series(1.0 / len(avail), index=avail) if len(avail) > 0 else pd.Series(dtype=float)
            ivw = ivw.reindex(mktcaps.columns).fillna(0.0)
            if ivw.sum() > 0:
                ivw = ivw / ivw.sum()

            # MCW_Cap40
            raw_mc_w = (mc_today[top] / mc_today[top].sum()).reindex(mktcaps.columns).fillna(0.0)
            mcw = cap_and_redistribute(raw_mc_w, cap=CAP_LIMIT)

            # 現在価値に合わせてリバランス（キャッシュ無しでシンプルに金額配分）
            axis_port_val = axis_val.loc[t] if not pd.isna(axis_val.loc[t]) else axis_val.iloc[i-1]
            mcw_port_val  = mcw_val.loc[t]  if not pd.isna(mcw_val.loc[t])  else mcw_val.iloc[i-1]
            axis_pos = ivw * axis_port_val
            mcw_pos  = mcw * mcw_port_val

        if i > 0:
            r = rets.loc[t]
            axis_pos = axis_pos * (1.0 + r.reindex(mktcaps.columns).fillna(0.0))
            mcw_pos  = mcw_pos  * (1.0 + r.reindex(mktcaps.columns).fillna(0.0))

            axis_val.loc[t] = axis_pos.sum()
            mcw_val.loc[t]  = mcw_pos.sum()
            sol_val.loc[t]  = sol_units * prices.at[t, BENCH_SYMBOL]

    axis_val = axis_val.ffill()
    mcw_val  = mcw_val.ffill()
    sol_val  = sol_val.ffill()

    results = pd.DataFrame({
        "AXIS_IVW": axis_val,
        "MCW_Cap40": mcw_val,
        "SOL_Hold": sol_val
    }, index=prices.index)

    # 週足のポートフォリオ価値CSV
    results.to_csv("portfolio_values.csv", index_label="Date")
    print("[Save] portfolio_values.csv（週次）")

    # メトリクス（週次ベース）
    def weekly_return_stats(curve: pd.Series):
        r = curve.pct_change().dropna()
        if r.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        cum_ret = curve.iloc[-1] / curve.iloc[0] - 1.0
        mean_w  = r.mean()
        std_w   = r.std(ddof=0)
        ann_ret = (1.0 + mean_w) ** 52 - 1.0
        ann_vol = std_w * sqrt(52)
        sharpe  = (mean_w / std_w) * sqrt(52) if std_w > 0 else np.nan
        dd = calculate_drawdown(curve)
        max_dd = dd.min() if not dd.empty else np.nan
        return cum_ret, ann_ret, ann_vol, sharpe, max_dd

    rows = {}
    for col in results.columns:
        rows[col] = dict(zip(
            ["Cumulative Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"],
            weekly_return_stats(results[col])
        ))
    metrics_df = pd.DataFrame(rows).T[
        ["Cumulative Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]
    ]
    print("\n[Metrics - Weekly basis]")
    print(metrics_df.to_string(float_format=lambda x: f"{x:,.4f}"))

    # 図
    plt.figure(figsize=(10,6))
    log_vals = np.log(results)
    for c in results.columns:
        plt.plot(log_vals.index, log_vals[c], label=c)
    plt.title("Log Cumulative Performance (Weekly)")
    plt.xlabel("Week"); plt.ylabel("Log Portfolio Value"); plt.legend(); plt.tight_layout()
    plt.savefig("chart_performance_log.png", dpi=150); plt.close()
    print("[Save] chart_performance_log.png")

    plt.figure(figsize=(10,6))
    for c in results.columns:
        dd = calculate_drawdown(results[c])
        plt.plot(dd.index, dd, label=c)
    plt.title("Drawdown (Underwater) - Weekly")
    plt.xlabel("Week"); plt.ylabel("Drawdown"); plt.legend(); plt.tight_layout()
    plt.savefig("chart_drawdown.png", dpi=150); plt.close()
    print("[Save] chart_drawdown.png")

# =========================
# メイン
# =========================
def main():
    parser = argparse.ArgumentParser(description="CoinCap v3 Weekly Backtest with CSV Cache")
    parser.add_argument("--update", action="store_true", help="既存週足CSVを差分更新してからBTを実行")
    parser.add_argument("--lookback-days", type=int, default=120, help="差分取得の過去日数（デフォルト120）")
    args = parser.parse_args()

    if args.update:
        print(f"[Info] 差分更新モード: 過去{args.lookback_days}日分のみ取得してマージ")
        weekly = update_weekly_csv(CSV_PATH, ASSETS, lookback_days=args.lookback_days)
    else:
        if not CSV_PATH.exists():
            print(f"[Info] 週足CSVが無いので作成します -> {CSV_PATH}")
            weekly = save_weekly_csv_from_coincap(CSV_PATH, ASSETS, START_DATE, END_DATE)
        else:
            print(f"[Info] 既存の週足CSVを使用: {CSV_PATH.resolve()}")
            weekly = pd.read_csv(CSV_PATH, parse_dates=["Date"]).set_index("Date").sort_index()

    weekly = weekly.loc[START_DATE:END_DATE]
    run_backtest_weekly(weekly)

if __name__ == "__main__":
    main()
