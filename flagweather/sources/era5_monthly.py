"""ERA5 monthly means (global, 1.5 deg): 500-hPa geopotential height, mean sea-level pressure and 250-hPa
zonal wind, 1950-present.

Cached by decade in data/era5_monthly/ (git-ignored); the current decade is re-fetched to pick up new months.
"""
import glob

import pandas as pd
import xarray as xr

from ..config import DATA

DIR = DATA / 'era5_monthly'
DIR.mkdir(exist_ok=True)
GRID = [1.5, 1.5]
_BASE = {'product_type': ['monthly_averaged_reanalysis'], 'time': ['00:00'], 'grid': GRID,
         'data_format': 'netcdf', 'download_format': 'unarchived', 'month': [f'{m:02d}' for m in range(1, 13)]}


def fetch(first_year=1950, refresh_current=True):
    import cdsapi
    c = cdsapi.Client(quiet=True, progress=False)
    now = pd.Timestamp.now()
    for d0 in range(first_year - first_year % 10, now.year + 1, 10):
        years = [str(y) for y in range(max(d0, first_year), min(d0 + 9, now.year) + 1)]
        current = int(years[-1]) == now.year
        for name, ds, extra in [
            ('z500', 'reanalysis-era5-pressure-levels-monthly-means',
             dict(variable=['geopotential'], pressure_level=['500'])),
            ('msl', 'reanalysis-era5-single-levels-monthly-means', dict(variable=['mean_sea_level_pressure'])),
            ('u250', 'reanalysis-era5-pressure-levels-monthly-means',
             dict(variable=['u_component_of_wind'], pressure_level=['250'])),
        ]:
            f = DIR / f'{name}_{years[0]}_{years[-1]}.nc'
            if f.exists() and not (current and refresh_current):
                continue
            for old in DIR.glob(f'{name}_{years[0]}_*.nc'):
                old.unlink()
            c.retrieve(ds, dict(_BASE, year=years, **extra), str(f))
    lsm = DIR / 'lsm.nc'
    if not lsm.exists():
        c.retrieve('reanalysis-era5-single-levels-monthly-means',
                   dict(_BASE, variable=['land_sea_mask'], year=['2020'], month=['01']), str(lsm))


def load():
    """Dataset with z500 (m) and msl (hPa), monthly time index, longitudes 0-360."""
    def cat(name):
        fs = sorted(glob.glob(str(DIR / f'{name}_*.nc')))
        return xr.concat([xr.open_dataset(f) for f in fs], 'valid_time').sortby('valid_time')
    z = cat('z500')
    m = cat('msl')
    ds = xr.Dataset({'z500': (z['z'].squeeze(drop=True) / 9.80665),
                     'msl': m['msl'] / 100.0})
    if glob.glob(str(DIR / 'u250_*.nc')):
        ds['u250'] = cat('u250')['u'].squeeze(drop=True)
    ds = ds.rename({'valid_time': 'time'})
    ds['time'] = pd.DatetimeIndex(ds.time.values).to_period('M').to_timestamp()
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby('longitude').sortby('latitude')
    return ds


def land_sea_mask():
    m = xr.open_dataset(DIR / 'lsm.nc').lsm.squeeze(drop=True)
    return m.assign_coords(longitude=m.longitude % 360).sortby('longitude').sortby('latitude')
