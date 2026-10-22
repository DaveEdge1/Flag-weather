"""ERA5 6-hourly fields from the Copernicus CDS (needs ~/.cdsapirc or CDSAPI_URL/CDSAPI_KEY).

Pressure levels: u, v, geopotential at 250 and 500 hPa.
Single levels:   IVT components (vertical integrals of water-vapour flux), total column water vapour.
Files are cached by month in data/era5/.
"""
import glob

import numpy as np
import pandas as pd
import xarray as xr

from ..config import DATA, ERA5_AREA

DIR = DATA / 'era5'
DIR.mkdir(exist_ok=True)
_COMMON = {'product_type': ['reanalysis'], 'time': ['00:00', '06:00', '12:00', '18:00'], 'area': ERA5_AREA,
           'data_format': 'netcdf', 'download_format': 'unarchived'}


def _client():
    import cdsapi
    return cdsapi.Client(quiet=True, progress=False)


def fetch(start, end, overwrite=False):
    """Download all days in [start, end], one file pair per month (partial months are re-fetched)."""
    c = _client()
    days = pd.date_range(start, end, freq='D')
    for (y, m), d in days.groupby([days.year, days.month]).items():
        tag = f'{y}-{m:02d}_{d[0].day:02d}-{d[-1].day:02d}'
        req = dict(_COMMON, year=[str(y)], month=[f'{m:02d}'], day=[f'{x.day:02d}' for x in d])
        pl, sl = DIR / f'pl_{tag}.nc', DIR / f'sl_{tag}.nc'
        if overwrite or not pl.exists():
            c.retrieve('reanalysis-era5-pressure-levels',
                       dict(req, variable=['u_component_of_wind', 'v_component_of_wind', 'geopotential'],
                            pressure_level=['250', '500']), str(pl))
        if overwrite or not sl.exists():
            c.retrieve('reanalysis-era5-single-levels',
                       dict(req, variable=['vertical_integral_of_eastward_water_vapour_flux',
                                           'vertical_integral_of_northward_water_vapour_flux',
                                           'total_column_water_vapour']), str(sl))
    lsm = DIR / 'lsm.nc'
    if not lsm.exists():
        c.retrieve('reanalysis-era5-single-levels',
                   dict(_COMMON, variable=['land_sea_mask'], year=['2026'], month=['01'], day=['01'], time=['00:00']),
                   str(lsm))


def load(start, end):
    """Merged dataset with ivt, ivt_u, ivt_v, tcwv, ws250, z500 for [start, end]."""
    pl = xr.concat([xr.open_dataset(f) for f in sorted(glob.glob(str(DIR / 'pl_*.nc')))], 'valid_time')
    sl = xr.concat([xr.open_dataset(f) for f in sorted(glob.glob(str(DIR / 'sl_*.nc')))], 'valid_time')
    pl = pl.sortby('valid_time').drop_duplicates('valid_time').sel(valid_time=slice(start, end))
    sl = sl.sortby('valid_time').drop_duplicates('valid_time').sel(valid_time=slice(start, end))
    return xr.Dataset({
        'ivt': np.hypot(sl.viwve, sl.viwvn), 'ivt_u': sl.viwve, 'ivt_v': sl.viwvn, 'tcwv': sl.tcwv,
        'ws250': np.hypot(pl.u.sel(pressure_level=250), pl.v.sel(pressure_level=250)).drop_vars('pressure_level'),
        'z500': (pl.z.sel(pressure_level=500) / 9.80665).drop_vars('pressure_level'),
    })


def land_sea_mask():
    return xr.open_dataset(DIR / 'lsm.nc').lsm.isel(valid_time=0)
