"""North Pacific jet variability from ERA5 monthly 250-hPa zonal wind.

Two published EOF definitions are reproduced on monthly means:
  - Athanasiadis, Wallace & Wettstein (2010, J. Atmos. Sci.): Dec-Mar, Pacific sector 120E-105W, EOFs computed
    for 1957-2002; P1 ~ PNA, P2 = the 'previously unnoted' pattern tied to West Coast precipitation. The poleward
    limit of their sector is not stated in the text used here; 0-80N is assumed (they note that the 0-20N band
    must be included for P2 to emerge).
  - Winters, Keyser & Bosart (2019, Wea. Forecasting) North Pacific Jet (NPJ) phase diagram: Sep-May, 10-80N,
    100E-120W, 1979-2014; EOF1 = jet extension (+) / retraction (-), EOF2 = poleward (+) / equatorward (-) shift.
    They used daily data; monthly means are an approximation.
EOFs use anomalies from the reference-period monthly means, sqrt(cos(lat)) weighting, and PCs standardized over
the reference period. All months are projected onto the reference EOFs.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

DEFS = {
    'A10': dict(name='Athanasiadis et al. (2010)', months=(12, 1, 2, 3), lat=(0, 80), lon=(120, 255),
                ref=(1957, 2002)),
    'W19': dict(name='Winters et al. (2019) NPJ', months=(9, 10, 11, 12, 1, 2, 3, 4, 5), lat=(10, 80),
                lon=(100, 240), ref=(1979, 2014)),
}


@dataclass
class JetEOF:
    key: str
    patterns: xr.DataArray     # (mode, latitude, longitude): regression of u250 anomaly (m/s) on standardized PC
    pcs: pd.DataFrame          # monthly PCs for all months (standardized over the reference period)
    var_frac: np.ndarray


def compute(u250, key, n=2):
    d = DEFS[key]
    u = u250.sel(latitude=slice(*d['lat']), longitude=slice(*d['lon']))
    t = pd.DatetimeIndex(u.time.values)
    clim = u.sel(time=slice(f"{d['ref'][0]}", f"{d['ref'][1]}")).groupby('time.month').mean('time')
    a = (u.groupby('time.month') - clim).drop_vars('month')
    w = np.sqrt(np.cos(np.deg2rad(a.latitude)))
    aw = (a * w).values.reshape(len(t), -1)
    inref = (t.year >= d['ref'][0]) & (t.year <= d['ref'][1]) & np.isin(t.month, d['months'])
    X = aw[inref]
    _, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    frac = s ** 2 / np.sum(s ** 2)
    raw = aw @ vt[:n].T                                   # project every month
    mu, sd = raw[inref].mean(0), raw[inref].std(0)
    pcs = (raw - mu) / sd
    # regression patterns (unweighted u250 anomaly per 1 s.d. of PC), reference months only
    A = a.values.reshape(len(t), -1)[inref]
    pat = (pcs[inref].T @ (A - A.mean(0))) / inref.sum()
    pat = xr.DataArray(pat.reshape(n, *a.shape[1:]), dims=['mode', 'latitude', 'longitude'],
                       coords={'mode': np.arange(1, n + 1), 'latitude': a.latitude, 'longitude': a.longitude})
    return JetEOF(key, pat, pd.DataFrame(pcs, index=t, columns=[f'PC{i + 1}' for i in range(n)]), frac[:n])


def orient(e, mode, lat, lon, positive=True):
    """Flip the sign of a mode so that its pattern is positive (or negative) at (lat, lon)."""
    v = float(e.patterns.sel(mode=mode).sel(latitude=lat, longitude=lon, method='nearest'))
    if (v < 0) == positive:
        e.patterns.loc[dict(mode=mode)] *= -1
        e.pcs[f'PC{mode}'] *= -1
    return e


def standard(u250, pna=None):
    """Both EOF sets with the sign conventions used on the site:
    W19 EOF1 + = jet extension; W19 EOF2 + = poleward shift; A10 P2 + = stronger westerlies near 25N, 145W
    (southward-displaced subtropical jet); A10 P1 aligned to correlate positively with the PNA index."""
    e = {k: compute(u250, k) for k in DEFS}
    orient(e['W19'], 1, 35, 195, True)
    orient(e['W19'], 2, 47, 195, True)
    orient(e['A10'], 2, 25, 215, True)
    if pna is not None and e['A10'].pcs.PC1.corr(pna.reindex(e['A10'].pcs.index)) < 0:
        e['A10'].pcs['PC1'] *= -1
        e['A10'].patterns.loc[dict(mode=1)] *= -1
    return e
