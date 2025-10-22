#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from datetime import timezone
from typing import Dict, Optional
import pytz

def parse_map(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    m = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--map の形式エラー: {pair}（例: 2797=ETH）")
        k, v = pair.split("=", 1)
        m[k.strip()] = v.strip()
    return m

def to_utc_series(ts_series: pd.Series, tz_in: str) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts_series, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(pytz.timezone(tz_in))
    dt = dt.dt.tz_convert(timezone.utc)
    return pd.DatetimeIndex(dt)

def resample_to_weekly(df: pd.DataFrame, weekstart: str) -> pd.DataFrame:
    rule = f"W-{weekstart.upper()}"
    agg = {
        "open": "first",
        "high": "max",
        "low":  "min",
        "close":"last",
        "volume":"sum",
    }
    out = (
        df.sort_index()
          .resample(rule, label="left", closed="left")
          .agg(agg)
          .dropna(subset=["open","high","low","close"])
          .reset_index()
    )
    return out

def main():
    parser = argparse.ArgumentParser(description="日次CSV→週足OHLCVへ整形")
    parser.add_argument("--input", required=True, help="入力CSVパス")
    parser.add_argument("--output", required=True, help="出力CSVパス")
    parser.add_argument("--map", default=None, help="name→symbol の対応（例: '2797=ETH,1234=BTC'）")
    parser.add_argument("--symbol", default=None, help="全行同一シンボルにする場合（例: ETH）")
    parser.add_argument("--tz-in", default="Asia/Tokyo", help="入力timestampのタイムゾーン（既定: Asia/Tokyo）")
    parser.add_argument("--weekstart", default="MON", choices=["MON","TUE","WED","THU","FRI","SAT","SUN"], help="週の起点（既定: MON）")
    parser.add_argument("--start", default=None, help="集計開始日（例: 2024-12-02）")
    parser.add_argument("--end", default=None, help="集計終了日（例: 2025-10-15）")
    parser.add_argument("--sep", default=None, help="入力CSVの区切り文字。未指定なら自動判定（例: ';'）")
    args = parser.parse_args()

    name_map = parse_map(args.map)

    # 区切り文字を自動判定（または --sep を使用）
    df = pd.read_csv(
        args.input,
        sep=(args.sep if args.sep else None),
        engine="python"
    )

    # 大文字小文字ゆらぎ吸収
    cols_lower = {c.lower(): c for c in df.columns}
    df = df.rename(columns=cols_lower)

    required = {"open","high","low","close","volume","timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"入力CSVに必要列が不足しています: {missing}")

    # symbol 列
    if args.symbol:
        df["symbol"] = args.symbol
    else:
        if "name" not in df.columns:
            raise ValueError("入力CSVに 'name' 列がありません。--symbol で固定指定も可能です。")
        df["symbol"] = df["name"].astype(str).map(name_map).fillna(df["name"].astype(str))

    # timestamp → UTC index
    ts_utc = to_utc_series(df["timestamp"], tz_in=args.tz_in)
    df = df.assign(timestamp=ts_utc).set_index("timestamp")

    # 数値化
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 期間クリップ
    if args.start:
        df = df.loc[df.index >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df.loc[df.index <= pd.Timestamp(args.end, tz="UTC")]

    if df.empty:
        raise ValueError("指定期間内にデータがありません。start/endやtz-inをご確認ください。")

    # symbol ごとに週足集計
    weekly_all = []
    for sym, g in df.groupby("symbol", sort=False):
        wk = resample_to_weekly(g[["open","high","low","close","volume"]], weekstart=args.weekstart)
        wk.insert(0, "symbol", sym)
        weekly_all.append(wk)

    out = pd.concat(weekly_all, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out[["symbol","timestamp","open","high","low","close","volume"]]

    out.to_csv(args.output, index=False)
    print(f"[ok] {args.output} -> {len(out)} rows")

if __name__ == "__main__":
    main()
