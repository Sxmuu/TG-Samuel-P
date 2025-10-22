#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de variables meteorológicas diarias para 2026 — versión defendible (solo ERA5)
------------------------------------------------------------------------------------------
Método:
  1. Descarga la climatología mensual 1991–2020 del reanálisis ERA5 (Copernicus C3S).
  2. Interpola los valores medios mensuales en las coordenadas de las estaciones de
     monitoreo de aire (Bogotá).
  3. Calcula variables derivadas: humedad relativa (RH2M) y viento a 2 m (WS2M).
  4. Extiende los valores mensuales a resolución diaria para el año 2026.
  5. Exporta el dataset completo (365 días × 15 estaciones).

Variables finales:
  PRECTOTCORR ≈ tp (precipitación total)
  PS ≈ msl (presión a nivel del mar)
  RH2M (humedad relativa)
  T2M (temperatura 2 m)
  WS2M (velocidad del viento a 2 m)

Requiere:
    pip install cdsapi xarray netCDF4 numpy pandas
"""

# ========================================
# 1️⃣ IMPORTACIÓN DE LIBRERÍAS Y CONFIG.
# ========================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cdsapi
except ImportError:
    raise ImportError("Instala 'cdsapi' con: pip install cdsapi")

# Directorio de salida
OUTDIR = Path("./out_climate_2026").resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)

# Coordenadas embebidas de estaciones de monitoreo (Bogotá)
STATIONS = [
    (1, "STA_01", 4.65847, -74.08396),
    (2, "STA_02", 4.57781, -74.16628),
    (3, "STA_03", 4.73719, -74.06948),
    (4, "STA_04", 4.67825, -74.14382),
    (5, "STA_05", 4.78375, -74.04414),
    (6, "STA_06", 4.60850, -74.11494),
    (7, "STA_07", 4.62505, -74.16133),
    (8, "STA_08", 4.69070, -74.08249),
    (9, "STA_09", 4.62549, -74.06698),
    (10, "STA_10", 4.66800, -74.14850),
    (11, "STA_11", 4.63177, -74.11749),
    (12, "STA_12", 4.57256, -74.08381),
    (13, "STA_13", 4.76125, -74.09346),
    (14, "STA_14", 4.57623, -74.13096),
    (15, "STA_15", 4.53206, -74.11714)
]

Z0_DEFAULT = 0.1  # rugosidad superficial
INTERP_METHOD = "linear"

# ==========================================
# 2️⃣ FUNCIONES AUXILIARES
# ==========================================

def compute_rh2m_from_t_and_td(t2m_K, d2m_K):
    """Calcula humedad relativa (%) a partir de T y Td en Kelvin."""
    T = t2m_K - 273.15
    Td = d2m_K - 273.15
    es_T = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    es_Td = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    rh = 100.0 * (es_Td / es_T)
    return np.clip(rh, 0.0, 100.0)

def wind10_to_wind2(ws10, z0=Z0_DEFAULT):
    """Convierte viento a 10 m a viento a 2 m (perfil logarítmico neutro)."""
    z0 = max(float(z0), 1e-4)
    factor = np.log(2.0 / z0) / np.log(10.0 / z0)
    return ws10 * factor

def _is_valid_netcdf(path: Path) -> bool:
    """Chequea magic bytes de NetCDF3/NetCDF4(HDF5)."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(b'CDF') or head == b'\x89HDF\r\n\x1a\n'
    except Exception:
        return False

def download_era5_monthly_climatology(outdir: Path, max_tries: int = 3) -> Path:
    """
    Descarga ERA5 mensual 1991–2020 y valida que sea NetCDF real.
    Si llega corrupto/HTML, reintenta automáticamente.
    """
    c = cdsapi.Client(timeout=600, retry_max=10, sleep_max=60)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "ERA5_climatology_1991_2020.nc"

    req = {
        "variable": [
            "2m_temperature", "2m_dewpoint_temperature",
            "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "total_precipitation"
        ],
        "product_type": "monthly_averaged_reanalysis",
        "month": [f"{m:02d}" for m in range(1, 13)],
        "year": [str(y) for y in range(1991, 2021)],
        "time": "00:00",
        "area": [13.5, -79.5, -4.5, -66.5],
        "format": "netcdf",
    }

    attempt = 1
    while attempt <= max_tries:
        if outpath.exists():
            # Si ya existe, validar
            if _is_valid_netcdf(outpath):
                print(f"[CDS] Climatología ERA5 ya existente: {outpath.name}")
                return outpath
            else:
                print("[CDS] Archivo existente está corrupto/ilegible. Borrando para reintentar…")
                outpath.unlink(missing_ok=True)

        print("[CDS] Descargando climatología mensual ERA5 1991–2020 … (intento", attempt, "de", max_tries, ")")
        c.retrieve("reanalysis-era5-single-levels-monthly-means", req, str(outpath))

        if _is_valid_netcdf(outpath):
            return outpath

        print("⚠️ Descarga inválida (no NetCDF). Reintentando…")
        outpath.unlink(missing_ok=True)
        attempt += 1

    raise RuntimeError("No se logró obtener un NetCDF válido de ERA5 tras varios intentos.")

def open_era5_dataset_safely(nc_path: Path) -> xr.Dataset:
    """
    Abre el NetCDF probando motores en orden:
    1) h5netcdf (NetCDF4 moderno)
    2) netcdf4
    3) scipy (NetCDF3)
    y evita OneDrive copiando a carpeta temporal local.
    """
    from shutil import copy2
    import tempfile

    nc_path = nc_path.resolve()
    print(f"📂 Verificando archivo: {nc_path}")
    if not nc_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {nc_path}")

    # Copia temporal fuera de OneDrive
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    copy2(nc_path, tmp_path)

    last_err = None
    for engine in ("h5netcdf", "netcdf4", "scipy"):
        try:
            ds = xr.open_dataset(tmp_path, engine=engine)
            print(f"✅ Archivo ERA5 abierto con engine='{engine}' desde copia temporal.")
            # Limpieza opcional del temporal se hace al final, tras usar ds.
            return ds
        except Exception as e:
            last_err = e

    # Si fallaron todos los engines, mostrar error y primeras bytes para diagnóstico
    try:
        with open(tmp_path, "rb") as f:
            head = f.read(200)
        print("🔎 Primeros bytes del archivo:", head[:80])
    except Exception:
        pass
    raise OSError(f"No se pudo abrir el NetCDF con ningún engine. Último error: {last_err}")

# ==========================================
# 3️⃣ PIPELINE PRINCIPAL
# ==========================================

def main():
    print(f"📍 Estaciones embebidas: {len(STATIONS)}")

    # 1) Descargar/validar NetCDF ERA5
    # RECOMENDADO: sacar OUTDIR de OneDrive para evitar bloqueos, por ej.:
    # OUTDIR = Path("C:/ERA5_local").resolve()
    era5_path = download_era5_monthly_climatology(OUTDIR)

    # 2) Abrir robustamente
    ds_era5 = open_era5_dataset_safely(era5_path)

    # 3) Climatología mensual y resto del pipeline
    ds_era5_clim = ds_era5.groupby("time.month").mean("time")
    print("📊 Generando serie diaria completa 2026 (basada en ERA5 climatology)…")

    df_era5_list = []
    for sid, name, lat, lon in STATIONS:
        sub = ds_era5_clim.interp(latitude=lat, longitude=lon, method=INTERP_METHOD)
        for m in range(1, 13):
            vals = {v: float(sub[v].sel(month=m)) for v in ["t2m", "d2m", "tp", "msl", "u10", "v10"]}
            rh = compute_rh2m_from_t_and_td(vals["t2m"], vals["d2m"])
            ws10 = np.hypot(vals["u10"], vals["v10"])
            ws2 = wind10_to_wind2(ws10)
            ndays = pd.Timestamp(f"2026-{m:02d}-01").days_in_month
            for d in range(1, ndays + 1):
                df_era5_list.append({
                    "site": sid,
                    "station": name,
                    "date": pd.Timestamp(f"2026-{m:02d}-{d:02d}"),
                    "t2m": vals["t2m"],
                    "rh2m": rh,
                    "tp": vals["tp"],
                    "msl": vals["msl"],
                    "ws2m": ws2,
                    "source": "ERA5_climatology"
                })

    df_all = pd.DataFrame(df_era5_list).sort_values(["site", "date"])
    out_csv = OUTDIR / "climate_features_2026_stations.csv"
    df_all.to_csv(out_csv, index=False)
    print(f"✅ Dataset final guardado en: {out_csv}")

# ==========================================
# 4️⃣ EJECUCIÓN
# ==========================================

if __name__ == "__main__":
    main()
