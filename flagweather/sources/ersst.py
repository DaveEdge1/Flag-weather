"""NOAA ERSSTv5 monthly SST (2 deg) over the Pacific, via PSL OPeNDAP, cached in data/ (git-ignored)."""
import time

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from ..config import DATA

URL = 'https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc'
FILE = DATA / 'ersst_pacific.nc'
LAT = (-30, 70)
LON = (100, 300)   # degrees east


def fetch(first_year=1950, blk=120):
    ds = netCDF4.Dataset(URL)
    lat, lon = ds['lat'][:], ds['lon'][:]
    ii = np.where((lat >= LAT[0]) & (lat <= LAT[1]))[0]
    jj = np.where((lon >= LON[0]) & (lon <= LON[1]))[0]
    tv = ds['time']
    times = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, 1) for t in
                              netCDF4.num2date(tv[:], tv.units, getattr(tv, 'calendar', 'standard'))])
    k0 = int(np.argmax(times.year >= first_year))
    parts = []
    for k in range(k0, len(times), blk):
        for attempt in range(4):
            try:
                a = ds['sst'][k:k + blk, ii.min():ii.max() + 1, jj.min():jj.max() + 1]
                parts.append(np.ma.filled(a.astype(float), np.nan))
                break
            except RuntimeError:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f'ERSST block {k} failed')
    sst = xr.DataArray(np.concatenate(parts), name='sst',
                       coords={'time': times[k0:], 'latitude': lat[ii.min():ii.max() + 1].astype(float),
                               'longitude': lon[jj.min():jj.max() + 1].astype(float)},
                       dims=['time', 'latitude', 'longitude']).sortby('latitude')
    sst.to_netcdf(FILE)
    return sst


def load(max_age_days=7):
    if not FILE.exists() or (time.time() - FILE.stat().st_mtime) > max_age_days * 86400:
        return fetch()
    return xr.open_dataarray(FILE)
