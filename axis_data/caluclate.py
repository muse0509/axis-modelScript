# -*- coding: utf-8 -*-
"""
Backtest: AXIS (IVW) vs MCW_Cap40 vs SOL_Hold
- Data: crypto_data_5y.csv
- Base Index: 100
- Rebalance: quarterly ('QS-JAN'), 実データ上の最初の営業日にスナップ
- Outputs:
    portfolio_values_usd.csv
    portfolio_index_100.csv
    allocations_log.csv
    chart_performance_log_base100.png
    chart_drawdown.png
    chart_relative_spread_axis_minus_mcw.png
    chart_rolling_sharpe_126d.png
    chart_drawdown_duration.png
    chart_hhi_rebalance.png
    chart_turnover_bar.png
    table_updown_capture.csv
    table_avg_turnover.csv
    metrics_table.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# パラメータ
# -----------------------------
CSV_PATH = "crypto_data_5y.csv"

INITIAL_USD = 1000.0
BASE_INDEX = 100.0
TOP_K = 5
IVW_ROLLING_DAYS = 90
CAP_LIMIT = 0.40      # 40%
RB_FREQ = "QS-JAN"    # Quarter start (JAN anchored)
TRADING_DAYS = 252

# -----------------------------
# ユーティリティ
# -----------------------------
def extract_price_mcap(df):
    price_cols = [c for c in df.columns if c.endswith("_Price")]
    mcap_cols  = [c for c in df.columns if c.endswith("_Market_Cap")]

    assets_from_price = [c.replace("_Price", "") for c in price_cols]
    assets_from_mcap  = [c.replace("_Market_Cap", "") for c in mcap_cols]
    assets = sorted(list(set(assets_from_price).intersection(set(assets_from_mcap))))

    price_df = pd.DataFrame(index=df.index, columns=assets, dtype=float)
    mcap_df  = pd.DataFrame(index=df.index, columns=assets, dtype=float)

    for a in assets:
        price_df[a] = pd.to_numeric(df[f"{a}_Price"], errors="coerce")
        mcap_df[a]  = pd.to_numeric(df[f"{a}_Market_Cap"], errors="coerce")

    return price_df, mcap_df, assets

def first_date_on_or_after(index, target_date):
    loc = index.searchsorted(target_date)
    if loc >= len(index):
        return None
    return index[loc]

def generate_rebalance_dates(all_dates):
    start = all_dates.min().normalize()
    end   = all_dates.max().normalize()

    ideal = pd.date_range(start=start, end=end, freq=RB_FREQ)
    rb_dates = [all_dates[0]]  # 初回は必ずデータ初日

    for d in ideal:
        snap = first_date_on_or_after(all_dates, d)
        if snap is not None and (len(rb_dates) == 0 or snap != rb_dates[-1]):
            rb_dates.append(snap)

    rb_dates = sorted(list(set(rb_dates)))
    return pd.DatetimeIndex(rb_dates)

def cap40_redistribute(raw_weights, cap=0.40, tol=1e-12):
    w = raw_weights.copy().astype(float)
    active = pd.Series(True, index=w.index)
    for _ in range(100):
        over = (w > cap) & active
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        active[over] = False

        under = active & (w < cap - tol)
        if under.any() and excess > tol:
            redistribute_total = w[under].sum()
            if redistribute_total > tol:
                w[under] += excess * (w[under] / redistribute_total)
            else:
                k = under.sum()
                w[under] += excess / max(k, 1)
        else:
            break

    s = w.sum()
    if s > 0:
        w = w / s
    return w

def rebalance_axis_ivw(date, returns, mcaps, top_k=5, lookback_days=90, min_valid=3):
    m = mcaps.loc[date].dropna().sort_values(ascending=False)
    top = m.index[:top_k].tolist()
    note = "ivw_weights"

    end_loc = returns.index.get_loc(date)
    start_loc = max(0, end_loc - lookback_days)
    hist = returns.iloc[start_loc:end_loc][top]

    vol = hist.std(ddof=0)

    # 有効なボラ(>0 かつ 非NaN)だけで IVW、min_valid 未満なら equal
    valid = vol[(~vol.isna()) & (vol > 0)]
    if (len(hist) < 10) or (len(valid) < min_valid):
        w = pd.Series(1.0/len(top), index=top)
        note = "fallback_equal_weights_due_to_invalid_vol"
    else:
        inv = 1.0 / valid
        w_valid = inv / inv.sum()
        w = pd.Series(0.0, index=top, dtype=float)
        w.loc[w_valid.index] = w_valid.values
        if w.sum() > 0:
            w = w / w.sum()

    log_df = pd.DataFrame({
        "Asset": top,
        "Vol90d": [vol.get(a, np.nan) for a in top],
        "Mcap": [m.get(a, np.nan) for a in top],
        "Weight": [w.get(a, 0.0) for a in top],
        "Note": [note]*len(top)
    })
    return w, log_df

def rebalance_mcw_cap40(date, mcaps, top_k=5, cap=0.40):
    m = mcaps.loc[date].dropna().sort_values(ascending=False)
    top = m.index[:top_k]
    raw = (m[top] / m[top].sum()).astype(float)
    w = cap40_redistribute(raw, cap=cap)

    log_df = pd.DataFrame({
        "Asset": top,
        "Mcap": m[top].values,
        "RawWeight": raw.values,
        "Weight": w.values,
        "Note": ["mcw_cap_40"]*len(top)
    })
    return w, log_df

def drawdown_series(curve_series):
    peak = curve_series.cummax()
    dd = curve_series / peak - 1.0
    return dd

def drawdown_duration(curve_series):
    dd = drawdown_series(curve_series)
    dur = []
    c = 0
    for v in dd:
        if v < 0:
            c += 1
        else:
            c = 0
        dur.append(c)
    return pd.Series(dur, index=curve_series.index)

def rolling_sharpe(returns, window=126):
    r = returns.copy().replace([np.inf, -np.inf], np.nan).dropna()
    roll_mean = r.rolling(window).mean() * TRADING_DAYS
    roll_vol  = r.rolling(window).std(ddof=0) * np.sqrt(TRADING_DAYS)
    out = roll_mean / roll_vol
    return out.replace([np.inf, -np.inf], np.nan)

# === 自前メトリクス ===
def ann_volatility_from_daily(daily_returns):
    r = pd.Series(daily_returns).replace([np.inf, -np.inf], np.nan).dropna()
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))

def sharpe_from_daily(daily_returns, rf=0.0):
    r = pd.Series(daily_returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return np.nan
    mean_daily_excess = (r - rf/TRADING_DAYS).mean()
    vol_daily = r.std(ddof=0)
    if vol_daily == 0 or np.isnan(vol_daily):
        return np.nan
    return float((mean_daily_excess / vol_daily) * np.sqrt(TRADING_DAYS))

def max_drawdown_from_daily(daily_returns):
    r = pd.Series(daily_returns).fillna(0.0)
    curve = (1.0 + r).cumprod()
    dd = curve / curve.cummax() - 1.0
    return float(dd.min())

def annual_return_from_curve(curve):
    total_r = curve.iloc[-1]/curve.iloc[0] - 1.0
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((1.0 + total_r)**(1.0/years) - 1.0)

# -----------------------------
# メイン
# -----------------------------
def main():
    # --- 1) データ読み込み ---
    raw = pd.read_csv(CSV_PATH)
    if "Date" not in raw.columns:
        raise ValueError("CSVにDate列が必要です。")

    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").drop_duplicates("Date")
    raw = raw.set_index("Date")

    price_df, mcap_df, assets = extract_price_mcap(raw)
    if price_df.empty or mcap_df.empty:
        raise ValueError("価格/時価総額の列が見つかりません。")

    rets = price_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    all_dates = price_df.index
    rb_dates  = generate_rebalance_dates(all_dates)

    # --- 2) バックテスト器 ---
    models = ["AXIS_IVW", "MCW_Cap40", "SOL_Hold"]
    weights = {m: pd.Series(dtype=float) for m in models}

    pv_usd = pd.DataFrame(index=all_dates, columns=models, dtype=float)
    pv_usd.iloc[0] = INITIAL_USD  # USDベース初期化（3モデルとも同額）

    alloc_logs = []

    if "SOL" not in price_df.columns or pd.isna(price_df["SOL"].iloc[0]):
        raise ValueError("SOLの価格列が必要です。CSVに 'SOL_Price' が含まれているか確認してください。")
    sol_qty = INITIAL_USD / price_df["SOL"].iloc[0]

    prev_date = all_dates[0]
    for i, date in enumerate(all_dates):
        # リバランス
        if date in rb_dates:
            w_axis, log_axis = rebalance_axis_ivw(date, rets, mcap_df, top_k=TOP_K, lookback_days=IVW_ROLLING_DAYS)
            weights["AXIS_IVW"] = w_axis
            log_axis.insert(0, "Model", "AXIS_IVW")
            log_axis.insert(0, "Date", date)
            alloc_logs.append(log_axis)
            print(f"\n[Rebalance] {date.date()} AXIS_IVW")
            print(log_axis[["Asset", "Vol90d", "Mcap", "Weight", "Note"]].to_string(index=False, max_colwidth=60))

            w_mcw, log_mcw = rebalance_mcw_cap40(date, mcap_df, top_k=TOP_K, cap=CAP_LIMIT)
            weights["MCW_Cap40"] = w_mcw
            log_mcw.insert(0, "Model", "MCW_Cap40")
            log_mcw.insert(0, "Date", date)
            alloc_logs.append(log_mcw)
            print(f"\n[Rebalance] {date.date()} MCW_Cap40")
            log_mcw_fmt = log_mcw.copy()
            log_mcw_fmt["Mcap"] = log_mcw_fmt["Mcap"].map(lambda x: f"{x:,.0f}")
            print(log_mcw_fmt[["Asset", "Mcap", "RawWeight", "Weight", "Note"]].to_string(index=False, max_colwidth=60))

        if i == 0:
            # 初日はP&L更新なし（初期化のみ）
            continue

        # --- 日次P&L更新 ---
        # AXIS
        w = weights["AXIS_IVW"].reindex(assets).fillna(0.0)
        daily_ret_axis = (w * rets.loc[date]).sum()
        pv_usd.loc[date, "AXIS_IVW"] = pv_usd.loc[prev_date, "AXIS_IVW"] * (1.0 + daily_ret_axis)

        # MCW
        w = weights["MCW_Cap40"].reindex(assets).fillna(0.0)
        daily_ret_mcw = (w * rets.loc[date]).sum()
        pv_usd.loc[date, "MCW_Cap40"] = pv_usd.loc[prev_date, "MCW_Cap40"] * (1.0 + daily_ret_mcw)

        # SOL Hold
        pv_usd.loc[date, "SOL_Hold"] = sol_qty * price_df.loc[date, "SOL"]

        prev_date = date

    # --- 3) 出力（曲線・指数化） ---
    pv_usd = pv_usd.ffill().dropna(how="any")
    pv_usd.to_csv("portfolio_values_usd.csv", index=True)

    # ✅ 列ごとの初期値で割ってベース100化（ブロードキャスト）
    pv_idx = pv_usd.divide(pv_usd.iloc[0], axis=1) * BASE_INDEX
    pv_idx.to_csv("portfolio_index_100.csv", index=True)

    # アロケーションログ
    alloc_df = pd.concat(alloc_logs, ignore_index=True)
    cols_order = ["Date", "Model", "Asset", "Weight", "Vol90d", "Mcap", "RawWeight", "Note"]
    for c in cols_order:
        if c not in alloc_df.columns:
            alloc_df[c] = np.nan
    alloc_df = alloc_df[cols_order]
    alloc_df.sort_values(["Date", "Model", "Asset"], inplace=True)
    alloc_df.to_csv("allocations_log.csv", index=False)
    print("\nAllocation log saved -> allocations_log.csv")

    # --- 4) メトリクス ---
    rets_models = pv_usd.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    perf = {}
    for m in ["AXIS_IVW","MCW_Cap40","SOL_Hold"]:
        sr = sharpe_from_daily(rets_models[m].values, rf=0.0)
        md = max_drawdown_from_daily(rets_models[m].values)
        ann_vol = ann_volatility_from_daily(rets_models[m].values)
        cum_ret = pv_usd[m].iloc[-1]/pv_usd[m].iloc[0] - 1.0
        ann_ret = annual_return_from_curve(pv_usd[m])
        perf[m] = {
            "Cumulative Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Vol": ann_vol,
            "Sharpe": sr,
            "Max Drawdown": md
        }

    perf_df = pd.DataFrame(perf).T
    perf_df.to_csv("metrics_table.csv")
    print("\n=== Performance Metrics ===")
    pretty = perf_df.copy()
    for c in ["Cumulative Return","Annualized Return","Annualized Vol","Max Drawdown"]:
        pretty[c] = (pretty[c]*100).map(lambda v: f"{v:,.4f}")
    pretty["Sharpe"] = pretty["Sharpe"].map(lambda v: f"{v:,.4f}")
    print(pretty)

    # --- 5) グラフ類 ---
    # 5-1) ログ累積（基準100）
    plt.figure(figsize=(11,5))
    np.log(pv_idx).plot(ax=plt.gca())
    plt.title("Cumulative Performance (Log Scale, Base=100)")
    plt.ylabel("log(Index)")
    plt.xlabel("Date")
    plt.legend(pv_idx.columns, loc="best")
    plt.tight_layout()
    plt.savefig("chart_performance_log_base100.png", dpi=200)
    plt.close()

    # 5-2) ドローダウン
    dd_df = pd.DataFrame({m: drawdown_series(pv_idx[m]) for m in pv_idx.columns}, index=pv_idx.index)
    dd_df.plot(figsize=(11,5))
    plt.title("Drawdown (Underwater, Base=100)")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("chart_drawdown.png", dpi=200)
    plt.close()

    # 5-3) 相対スプレッド（AXIS - MCW）
    spread = pv_idx["AXIS_IVW"] - pv_idx["MCW_Cap40"]
    spread.plot(figsize=(11,4))
    plt.title("AXIS_IVW minus MCW_Cap40 (Index Points, Base=100)")
    plt.ylabel("Spread")
    plt.axhline(0, linewidth=1)
    plt.tight_layout()
    plt.savefig("chart_relative_spread_axis_minus_mcw.png", dpi=200)
    plt.close()

    # 5-4) ローリングSharpe（126日）
    rs = pd.DataFrame({
        "AXIS_IVW": rolling_sharpe(rets_models["AXIS_IVW"], window=126),
        "MCW_Cap40": rolling_sharpe(rets_models["MCW_Cap40"], window=126),
        "SOL_Hold":  rolling_sharpe(rets_models["SOL_Hold"],  window=126),
    })
    rs.plot(figsize=(11,4))
    plt.title("Rolling Sharpe (126 trading days)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("chart_rolling_sharpe_126d.png", dpi=200)
    plt.close()

    # 5-5) ドローダウン期間（Days Under Water）
    dur_df = pd.DataFrame({m: drawdown_duration(pv_idx[m]) for m in pv_idx.columns}, index=pv_idx.index)
    dur_df.plot(figsize=(11,4))
    plt.title("Drawdown Duration (Days Under Water)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("chart_drawdown_duration.png", dpi=200)
    plt.close()

    # 5-6) HHI（リバランス日の集中度）
    al = alloc_df.copy()
    al["Date"] = pd.to_datetime(al["Date"])
    def to_hhi(df):
        w = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
        return float((w**2).sum())

    hhi = (al.groupby(["Date","Model"])
             .apply(to_hhi)
             .unstack("Model")
             .sort_index())
    hhi.plot(figsize=(11,4))
    plt.title("Concentration (HHI) at Rebalance Dates")
    plt.ylabel("Sum(weights^2)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("chart_hhi_rebalance.png", dpi=200)
    plt.close()

    # 5-7) Turnover（|Δw|/2）
    def turnover_series(model):
        w = (al[al["Model"]==model]
             .pivot_table(index="Date", columns="Asset", values="Weight", aggfunc="last")
             .fillna(0.0)).sort_index()
        tw = (w.diff().abs().sum(axis=1) / 2.0).dropna()
        return tw

    to_axis = turnover_series("AXIS_IVW")
    to_mcw  = turnover_series("MCW_Cap40")
    turn_df = pd.DataFrame({"AXIS_IVW": to_axis, "MCW_Cap40": to_mcw})
    turn_df.plot(kind="bar", figsize=(12,4))
    plt.title("Turnover by Rebalance (lower is better)")
    plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig("chart_turnover_bar.png", dpi=200)
    plt.close()

    pd.DataFrame({
        "AXIS_IVW_avg_turnover": [to_axis.mean() if len(to_axis) else np.nan],
        "MCW_Cap40_avg_turnover": [to_mcw.mean() if len(to_mcw) else np.nan]
    }).to_csv("table_avg_turnover.csv", index=False)

    # 5-8) Up/Down Capture（BTCを市場代理に）
    if "BTC" in rets.columns:
        mkt = rets["BTC"]
        up = mkt > 0
        down = mkt < 0

        def capture(series, mkt):
            up_cap = (series[up].mean() / mkt[up].mean()) if up.any() and mkt[up].mean() != 0 else np.nan
            dn_cap = (series[down].mean() / mkt[down].mean()) if down.any() and mkt[down].mean() != 0 else np.nan
            return up_cap, dn_cap

        caps = {}
        for col in ["AXIS_IVW","MCW_Cap40","SOL_Hold"]:
            caps[col] = capture(rets_models[col], mkt)

        cap_df = pd.DataFrame(caps, index=["UpCapture","DownCapture"]).T
        cap_df.to_csv("table_updown_capture.csv")
        print("\n=== Up/Down Capture vs BTC ===")
        print(cap_df)
    else:
        print("\n(BTCが無いので Up/Down Capture はスキップ)")

    # --- 完了 ---
    print("\nOutputs:")
    print(" - portfolio_values_usd.csv")
    print(" - portfolio_index_100.csv")
    print(" - chart_performance_log_base100.png")
    print(" - chart_drawdown.png")
    print(" - chart_relative_spread_axis_minus_mcw.png")
    print(" - chart_rolling_sharpe_126d.png")
    print(" - chart_drawdown_duration.png")
    print(" - chart_hhi_rebalance.png")
    print(" - chart_turnover_bar.png")
    print(" - table_updown_capture.csv")
    print(" - table_avg_turnover.csv")
    print(" - allocations_log.csv")
    print(" - metrics_table.csv")

if __name__ == "__main__":
    main()
