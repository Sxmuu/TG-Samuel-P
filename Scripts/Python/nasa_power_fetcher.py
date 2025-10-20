#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nasa_power_fetcher.py
---------------------
General-purpose script to fetch climate data from NASA POWER API for one or many points,
and export DAILY data (as returned by the API), plus MONTHLY and ANNUAL aggregations.
Designed for agriculture/insurance/data-science workflows.

Features
- Single point via CLI flags (--lat/--lon) or multiple points via CSV (--points-csv).
- Choose temporal resolution to query from POWER: daily OR monthly ("as-is" from the API).
- Always produces three outputs (if requested): daily.csv (from API), monthly.csv, annual.csv.
  * If you queried monthly from POWER, then "daily.csv" will be empty (header only) unless you pass --also-daily.
- Flexible parameter list: defaults suited for agroclimate use.
- Robust HTTP with retries + backoff. Minimal rate limiting.
- Clean, tidy CSV output with columns: source, lat, lon, date/year, month, variable columns.
- Aggregation logic: SUM variables vs MEAN variables are handled sensibly (see VAR_AGG_POLICY).

Usage examples
--------------
1) Single point, DAILY data 2007-01-01 to 2023-12-31, default variables:
   python nasa_power_fetcher.py --lat 4.44 --lon -75.24 --start 20070101 --end 20231231 --temporal daily --outdir ./salidas

2) Multiple points from CSV (cols: name,lat,lon), MONTHLY direct from POWER (no daily):
   python nasa_power_fetcher.py --points-csv ./municipios.csv --temporal monthly --start 200701 --end 202312 --outdir ./salidas

3) Monthly from POWER + also compute daily (slower; does two calls):
   python nasa_power_fetcher.py --lat 4.44 --lon -75.24 --temporal monthly --also-daily --start 200701 --end 202312 --outdir ./salidas

CSV input format for --points-csv
---------------------------------
- Must include columns: lat, lon. Optional: name, id, dpto, municipio, codigo_dane, etc.
- Any extra columns are preserved in the outputs (handy for joins later).

Notes
-----
- NASA POWER limits: avoid very large, dense requests. Consider batching years if needed.
- If you need bounding boxes or grids, this script can be extended to use "regional" endpoints.
- The script only uses "temporal/*/point" endpoints for simplicity and portability.

Author: Shared for collaborative academic use.
License: MIT
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://power.larc.nasa.gov/api/temporal/{temporal}/point"

# Default variables (agroclimate-friendly)
DEFAULT_PARAMETERS = [
    "T2M",               # 2m Air Temperature (°C)
    "PRECTOTCORR",       # Precipitation Corrected (mm/day for daily; mm/month for monthly)
    "RH2M",              # Relative Humidity at 2m (%)
    "WS2M",              # Wind Speed at 2m (m/s),
    "PS",                 # Shortwave Downwelling Radiation (MJ/m^2/day)
]

# Aggregation policy when building monthly/annual from DAILY
# SUM: precipitation, radiation-like fluxes
# MEAN: temperature, humidity, wind speed
VAR_AGG_POLICY = {
    "SUM": {"PRECTOTCORR"},
    "MEAN": {"T2M", "RH2M", "WS2M", "PS"},
}
# Any variable not listed falls back to MEAN.
# You can override via CLI: --sum-vars PRECTOTCORR --mean-vars T2M,RH2M,WS2M

@dataclass
class Point:
    lat: float
    lon: float
    meta: Dict[str, object]

def _build_session(total_retries: int = 5, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def _parse_parameters(param_str_or_list: Optional[str | List[str]]) -> str:
    if param_str_or_list is None:
        return ",".join(DEFAULT_PARAMETERS)
    if isinstance(param_str_or_list, list):
        return ",".join(param_str_or_list)
    # string: allow comma-separated or space-separated
    parts = [p.strip().upper() for p in param_str_or_list.replace(" ", ",").split(",") if p.strip()]
    return ",".join(parts)

def _infer_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a proper datetime index/columns exist when DAILY data present."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    elif {"year", "month"}.issubset(df.columns):
        # Nothing else to do here
        pass
    return df

def read_points_from_csv(csv_path: str) -> List[Point]:
    df = pd.read_csv(csv_path)
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV must contain 'lat' and 'lon' columns.")
    points: List[Point] = []
    for _, row in df.iterrows():
        meta = {k: row[k] for k in df.columns if k not in {"lat", "lon"}}
        points.append(Point(lat=float(row["lat"]), lon=float(row["lon"]), meta=meta))
    return points

def fetch_power_point(
    session: requests.Session,
    temporal: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
    parameters: str,
    community: str = "AG",
    fmt: str = "JSON",
    rate_sleep: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch NASA POWER data for one point and returns a tidy DataFrame.
    - temporal: "daily" or "monthly"
    - start/end formats:
        daily   -> YYYYMMDD (e.g., 20070101)
        monthly -> YYYYMM   (e.g., 200701)
    """
    url = API_BASE.format(temporal=temporal.lower().strip())
    params = {
        "community": community,
        "parameters": parameters,
        "start": start,
        "end": end,
        "format": fmt,
        "latitude": lat,
        "longitude": lon,
    }
    r = session.get(url, params=params, timeout=60)
    # Rate limiting courtesy
    time.sleep(rate_sleep)
    r.raise_for_status()
    js = r.json()
    # Expected path: properties.parameter -> dict[var] -> {date_str: value}
    props = js.get("properties", {})
    pblock = props.get("parameter", {})
    if not isinstance(pblock, dict) or not pblock:
        # return empty DF with metadata
        return pd.DataFrame([{"source": "NASA_POWER", "lat": lat, "lon": lon}] )

    # Build tidy rows
    # For daily: keys like "YYYYMMDD"; for monthly: "YYYYMM"
    rows = []
    for var, series in pblock.items():
        if not isinstance(series, dict):
            continue
        for k, v in series.items():
            if temporal == "daily":
                yyyy, mm, dd = k[0:4], k[4:6], k[6:8]
                date = f"{yyyy}-{mm}-{dd}"
                rows.append({"date": date, "var": var, "value": v})
            else:
                yyyy, mm = k[0:4], k[4:6]
                rows.append({"year": int(yyyy), "month": int(mm), "var": var, "value": v})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame([{"source": "NASA_POWER", "lat": lat, "lon": lon}] )

    # Pivot to wide (variables as columns)
    if temporal == "daily":
        df_w = df.pivot_table(index=["date"], columns="var", values="value").reset_index()
        df_w["source"] = "NASA_POWER"
        df_w["lat"] = lat
        df_w["lon"] = lon
        # reorder: source, lat, lon, date, vars...
        cols = ["source", "lat", "lon", "date"] + [c for c in df_w.columns if c not in {"source","lat","lon","date"}]
        df_w = df_w[cols]
    else:
        df_w = df.pivot_table(index=["year","month"], columns="var", values="value").reset_index()
        df_w["source"] = "NASA_POWER"
        df_w["lat"] = lat
        df_w["lon"] = lon
        cols = ["source", "lat", "lon", "year", "month"] + [c for c in df_w.columns if c not in {"source","lat","lon","year","month"}]
        df_w = df_w[cols]

    return df_w

def aggregate_daily_to_monthly_and_annual(df_daily: pd.DataFrame, sum_vars: Iterable[str], mean_vars: Iterable[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily dataframe to monthly and annual, using SUM or MEAN per variable."""
    if df_daily.empty or "date" not in df_daily.columns:
        # Return empty shells
        monthly = pd.DataFrame(columns=["source","lat","lon","year","month"])
        annual  = pd.DataFrame(columns=["source","lat","lon","year"])
        return monthly, annual

    df = df_daily.copy()
    df = _infer_date_columns(df)

    base_cols = ["source","lat","lon"]
    var_cols = [c for c in df.columns if c not in set(base_cols + ["date","year","month"])]
    # Compute monthly
    groups = base_cols + ["year","month"]
    mon_parts = []
    for var in var_cols:
        if var in sum_vars:
            mon = df.groupby(groups, dropna=False)[var].sum(min_count=1)
        else:
            mon = df.groupby(groups, dropna=False)[var].mean()
        mon_parts.append(mon)
    if mon_parts:
        monthly = pd.concat(mon_parts, axis=1).reset_index()
    else:
        monthly = pd.DataFrame(columns=groups)

    # Compute annual from monthly (safer than summing daily directly, but equivalent for SUM vars)
    ann_groups = base_cols + ["year"]
    ann_parts = []
    for var in var_cols:
        if var in sum_vars:
            ann = monthly.groupby(ann_groups, dropna=False)[var].sum(min_count=1)
        else:
            ann = monthly.groupby(ann_groups, dropna=False)[var].mean()
        ann_parts.append(ann)
    if ann_parts:
        annual = pd.concat(ann_parts, axis=1).reset_index()
    else:
        annual = pd.DataFrame(columns=ann_groups)

    return monthly, annual

def save_csv(df: pd.DataFrame, path: str) -> None:
    # Always write with utf-8 and index=False
    df.to_csv(path, index=False, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch NASA POWER data (daily/monthly) for one or many points and export daily, monthly, annual CSVs."
    )
    g_point = parser.add_mutually_exclusive_group(required=True)
    g_point.add_argument("--lat", type=float, help="Latitude for single point")
    g_point.add_argument("--points-csv", type=str, help="CSV with columns: lat, lon, [any metadata cols]")

    parser.add_argument("--lon", type=float, help="Longitude for single point (required if --lat)")
    parser.add_argument("--temporal", type=str, choices=["daily","monthly"], default="daily", help="Temporal resolution to query from POWER")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYYMMDD for daily, YYYYMM for monthly)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYYMMDD for daily, YYYYMM for monthly)")
    parser.add_argument("--parameters", type=str, default=",".join(DEFAULT_PARAMETERS),
                        help=f"Comma/space separated list of variables. Default: {','.join(DEFAULT_PARAMETERS)}")
    parser.add_argument("--outdir", type=str, default="./salidas", help="Output directory")
    parser.add_argument("--community", type=str, default="AG", help="NASA POWER 'community' parameter")
    parser.add_argument("--also-daily", action="store_true",
                        help="If temporal=monthly, also fetch daily (adds an extra API call) to compute monthly/annual on our side")
    parser.add_argument("--sum-vars", type=str, default="PRECTOTCORR", help="Variables to SUM when aggregating daily")
    parser.add_argument("--mean-vars", type=str, default="T2M,RH2M,WS2M,PS", help="Variables to MEAN when aggregating daily")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between requests (rate limiting)")
    parser.add_argument("--retries", type=int, default=5, help="HTTP total retries")
    args = parser.parse_args()

    # Build points list
    if args.lat is not None:
        if args.lon is None:
            parser.error("--lon is required when using --lat")
        points = [Point(lat=args.lat, lon=args.lon, meta={})]
    else:
        points = read_points_from_csv(args.points_csv)

    # Parse variables and policies
    params = _parse_parameters(args.parameters)
    sum_vars = {v.strip().upper() for v in args.sum_vars.replace(" ", ",").split(",") if v.strip()}
    mean_vars = {v.strip().upper() for v in args.mean_vars.replace(" ", ",").split(",") if v.strip()}
    # Ensure no overlap
    mean_vars = mean_vars - sum_vars

    session = _build_session(total_retries=args.retries)

    out_daily_all: List[pd.DataFrame] = []
    out_month_all: List[pd.DataFrame] = []
    out_annual_all: List[pd.DataFrame] = []

    for i, p in enumerate(points, start=1):
        try:
            print(f"[{i}/{len(points)}] Fetching {args.temporal} for lat={p.lat}, lon={p.lon} ...", file=sys.stderr)

            # fetch primary (daily or monthly) from POWER
            df_primary = fetch_power_point(
                session=session,
                temporal=args.temporal,
                lat=p.lat,
                lon=p.lon,
                start=args.start,
                end=args.end,
                parameters=params,
                community=args.community,
                rate_sleep=args.sleep,
            )

            # Attach metadata columns from CSV if present
            if p.meta:
                for k, v in p.meta.items():
                    df_primary[k] = v

            if args.temporal == "daily":
                daily = df_primary.copy()
                # Aggregate to monthly & annual
                monthly, annual = aggregate_daily_to_monthly_and_annual(daily, sum_vars, mean_vars)
            else:
                # monthly direct from POWER
                monthly = df_primary.copy()

                if args.also-daily:
                    # Also fetch daily to compute aggregations ourselves
                    daily = fetch_power_point(
                        session=session,
                        temporal="daily",
                        lat=p.lat,
                        lon=p.lon,
                        start=args.start + ("01" if len(args.start) == 6 else ""),
                        end=args.end + ("28" if len(args.end) == 6 else ""),  # safe-ish end day; POWER ignores overflow
                        parameters=params,
                        community=args.community,
                        rate_sleep=args.sleep,
                    )
                    if p.meta:
                        for k, v in p.meta.items():
                            daily[k] = v
                    # Recompute monthly & annual from daily (can help if your policy differs)
                    monthly_from_daily, annual = aggregate_daily_to_monthly_and_annual(daily, sum_vars, mean_vars)
                    # Prefer POWER monthly unless user wants otherwise; here we keep POWER monthly
                else:
                    # No daily data available; create empty shells for daily & annual (derive annual from monthly)
                    daily = pd.DataFrame(columns=["source","lat","lon","date"])
                    # Annual from monthly by policy: sum vs mean
                    if monthly.empty:
                        annual = pd.DataFrame(columns=["source","lat","lon","year"])
                    else:
                        base_cols = ["source","lat","lon","year"]
                        var_cols = [c for c in monthly.columns if c not in {"source","lat","lon","year","month"} | set(p.meta.keys())]
                        parts = []
                        for var in var_cols:
                            if var.upper() in sum_vars:
                                agg = monthly.groupby(["source","lat","lon","year"], dropna=False)[var].sum(min_count=1)
                            else:
                                agg = monthly.groupby(["source","lat","lon","year"], dropna=False)[var].mean()
                            parts.append(agg)
                        annual = pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(columns=base_cols)

            # Append
            out_daily_all.append(daily)
            out_month_all.append(monthly)
            out_annual_all.append(annual)

        except Exception as ex:
            print(f"[WARN] Failed for point lat={p.lat}, lon={p.lon}: {ex}", file=sys.stderr)
            continue

    # Concatenate and save
    daily_all = pd.concat(out_daily_all, ignore_index=True) if out_daily_all else pd.DataFrame()
    monthly_all = pd.concat(out_month_all, ignore_index=True) if out_month_all else pd.DataFrame()
    annual_all = pd.concat(out_annual_all, ignore_index=True) if out_annual_all else pd.DataFrame()

    # Create outdir
    import os
    os.makedirs(args.outdir, exist_ok=True)

    # Ensure column order pleasant
    def order_cols(df: pd.DataFrame, level: str) -> pd.DataFrame:
        if df.empty:
            return df
        base = ["source","lat","lon"]
        if level == "daily":
            lead = base + ["date"]
        elif level == "monthly":
            lead = base + ["year","month"]
        else:
            lead = base + ["year"]
        # Keep meta columns next (if any), then variables
        meta_cols = [c for c in df.columns if c not in set(lead) and c not in set(DEFAULT_PARAMETERS)]
        var_cols = [c for c in df.columns if c not in set(lead + meta_cols)]
        return df[lead + meta_cols + var_cols]

    daily_all = order_cols(daily_all, "daily")
    monthly_all = order_cols(monthly_all, "monthly")
    annual_all = order_cols(annual_all, "annual")

    # Write CSVs
    daily_path = os.path.join(args.outdir, "nasa_power_daily.csv")
    monthly_path = os.path.join(args.outdir, "nasa_power_monthly.csv")
    annual_path = os.path.join(args.outdir, "nasa_power_annual.csv")

    save_csv(daily_all, daily_path)
    save_csv(monthly_all, monthly_path)
    save_csv(annual_all, annual_path)

    print(f"[OK] Saved:\n  - {daily_path}\n  - {monthly_path}\n  - {annual_path}")

if __name__ == "__main__":
    main()
