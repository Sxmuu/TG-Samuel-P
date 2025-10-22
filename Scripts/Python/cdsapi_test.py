import cdsapi

# Inicializa el cliente (usa tu .cdsapirc automáticamente)
c = cdsapi.Client()

# Descarga una hora de ERA5 (dato pequeño para probar)
c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "variable": "2m_temperature",
        "year": "2020",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "format": "netcdf",
    },
    "test_era5.nc"
)
