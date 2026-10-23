"""Seasonal Pacific - Southwest North America diagnostics: ENSO, PDO, PNA, Aleutian Low, North Pacific High.

Anomalies are relative to 1991-2020 monthly means. Winter = November-March, labelled by the year in
which it starts (winter 2026 = Nov 2026 - Mar 2027).
"""
import urllib.request

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from . import jet
from .sources import acis, era5_monthly, ersst

CLIM = (1991, 2020)
FIRST = 1950
NPI_BOX = dict(lat=(30, 65), lon=(160, 220))        # North Pacific Index (Trenberth & Hurrell 1994)
AL_BOX = dict(lat=(30, 65), lon=(150, 230))         # search box for the Aleutian Low centre
NPH_BOX = dict(lat=(15, 50), lon=(180, 240))        # search box for the North Pacific High centre
BOXES = {  # 500-hPa height anomaly summary boxes
    'Gulf of Alaska': dict(lat=(45, 60), lon=(200, 225)),
    'Southwest US': dict(lat=(31, 40), lon=(245, 255)),
}
NINO34 = dict(lat=(-5, 5), lon=(190, 240))
DOMAIN = dict(lat=(-10, 70), lon=(120, 300))
ENSO_WORDS = [(2.0, 'very strong'), (1.5, 'strong'), (1.0, 'moderate'), (0.5, 'weak')]


def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Flag-weather'})
    return urllib.request.urlopen(req, timeout=60).read().decode()


# ---- indices from published files --------------------------------------------------------------
def oni():
    """CPC ONI (3-month running Niño3.4 anomaly), indexed by the centre month."""
    centre = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8, 'ASO': 9,
              'SON': 10, 'OND': 11, 'NDJ': 12}
    rows = [r.split() for r in _get('https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt').splitlines()[1:]]
    s = {pd.Timestamp(int(r[1]), centre[r[0]], 1): float(r[3])
         for r in rows if len(r) == 4 and r[0] in centre}
    return pd.Series(s, name='ONI').sort_index()


def pdo():
    recs = {}
    for p in map(str.split, _get('https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat').splitlines()[2:]):
        if len(p) == 13:
            for m, v in enumerate(p[1:]):
                if float(v) < 99:
                    recs[pd.Timestamp(int(p[0]), m + 1, 1)] = float(v)
    return pd.Series(recs, name='PDO').sort_index()


def pna():
    recs = {}
    for r in map(str.split, _get('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/'
                                 'norm.pna.monthly.b5001.current.ascii').splitlines()):
        if len(r) == 3:
            recs[pd.Timestamp(int(r[0]), int(r[1]), 1)] = float(r[2])
    return pd.Series(recs, name='PNA').sort_index()


def enso_word(v):
    for thr, w in ENSO_WORDS:
        if abs(v) >= thr:
            return f'{w} {"El Niño" if v > 0 else "La Niña"}'
    return 'ENSO-neutral'


# ---- gridded helpers ---------------------------------------------------------------------------
def _box(da, b):
    return da.sel(latitude=slice(*b['lat']), longitude=slice(*b['lon']))


def _wmean(da):
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(['latitude', 'longitude'])


def anomaly(da):
    clim = da.sel(time=slice(f'{CLIM[0]}', f'{CLIM[1]}')).groupby('time.month').mean('time')
    return (da.groupby('time.month') - clim).drop_vars('month')


def winter_mean(da, months=(11, 12, 1, 2, 3)):
    """Mean over Nov-Mar, labelled by the year the winter starts; only complete winters."""
    t = pd.DatetimeIndex(da.time.values)
    keep = np.isin(t.month, months)
    sel = da.isel(time=keep)
    ts = t[keep]
    wy = xr.DataArray(ts.year - (ts.month < 7), dims='time', coords={'time': sel.time}, name='winter')
    g = sel.groupby(wy)
    return g.mean('time').where(g.count('time') == len(months))


class Pacific:
    def load(self):
        self.atm = era5_monthly.load()
        self.lsm = era5_monthly.land_sea_mask()
        self.sst = ersst.load()
        self.atm_a = xr.Dataset({k: anomaly(self.atm[k]) for k in ('z500', 'msl')})
        self.sst_a = anomaly(self.sst)
        self.oni, self.pdo, self.pna = oni(), pdo(), pna()
        self.nino34 = _wmean(_box(self.sst_a, NINO34)).to_series().rename('Niño3.4')
        msl = self.atm.msl
        self.npi = _wmean(_box(msl, NPI_BOX)).to_series().rename('NPI')
        self.npi_a = _wmean(_box(self.atm_a.msl, NPI_BOX)).to_series().rename('NPI anomaly')
        self.al = self._centre(msl, AL_BOX, np.argmin)
        self.nph = self._centre(msl, NPH_BOX, np.argmax)
        for c in (self.al, self.nph):
            clim = c.p[(c.index.year >= CLIM[0]) & (c.index.year <= CLIM[1])].groupby(
                c.index[(c.index.year >= CLIM[0]) & (c.index.year <= CLIM[1])].month).mean()
            c['p_anom'] = c.p - c.index.month.map(clim).values
        pr = acis.precip_record()
        mon = pr.resample('MS').agg(lambda x: x.sum(min_count=int(len(x) * 0.9)))
        self.flg_winter = self._winter_series(mon).rename('Flagstaff Nov–Mar precip (in)')
        self.jet = jet.standard(self.atm.u250, self.pna) if 'u250' in self.atm else None
        self.flg_monthly = acis.monthly('2019-01', pd.Timestamp.now().strftime('%Y-%m'))
        self.latest_atm = pd.Timestamp(self.atm.time.values[-1])
        self.latest_sst = pd.Timestamp(self.sst.time.values[-1])
        return self

    @staticmethod
    def _centre(msl, b, fn):
        sub = _box(msl, b)
        v = sub.values.reshape(sub.shape[0], -1)
        k = fn(v, axis=1)
        lat2, lon2 = np.meshgrid(sub.latitude.values, sub.longitude.values, indexing='ij')
        return pd.DataFrame({'p': v[np.arange(len(k)), k], 'lat': lat2.ravel()[k], 'lon': lon2.ravel()[k]},
                            index=pd.DatetimeIndex(sub.time.values))

    @staticmethod
    def _winter_series(monthly):
        out = {}
        for y in range(FIRST, pd.Timestamp.now().year):
            w = monthly.loc[f'{y}-11':f'{y + 1}-03']
            if len(w) == 5 and w.notna().all():
                out[y] = w.sum()
        return pd.Series(out)

    # ---- periods ------------------------------------------------------------------------------
    def period(self, n, end=None):
        """Last n complete months common to SST and atmosphere."""
        end = end or min(self.latest_atm, self.latest_sst)
        return pd.date_range(end=end, periods=n, freq='MS')

    def mean_anom(self, months):
        return (self.sst_a.sel(time=months).mean('time'), self.atm_a.z500.sel(time=months).mean('time'),
                self.atm_a.msl.sel(time=months).mean('time'), self.atm.msl.sel(time=months).mean('time'))

    def strong_el_nino_years(self, threshold=1.5):
        o = self.oni[self.oni.index.month == 12]          # NDJ ONI, centred on December
        yrs = [t.year for t, v in o.items() if v >= threshold and FIRST <= t.year < pd.Timestamp.now().year]
        return yrs

    def analog_composite(self, months_of_year, years):
        """Mean anomaly over the given calendar months of each analog year (month >= 7 -> that year,
        else the following year)."""
        sel = [pd.Timestamp(y + (m < 7), m, 1) for y in years for m in months_of_year]
        sel_sst = [t for t in sel if t in pd.DatetimeIndex(self.sst_a.time.values)]
        sel_atm = [t for t in sel if t in pd.DatetimeIndex(self.atm_a.time.values)]
        return (self.sst_a.sel(time=sel_sst).mean('time'), self.atm_a.z500.sel(time=sel_atm).mean('time'),
                self.atm_a.msl.sel(time=sel_atm).mean('time'))

    @staticmethod
    def pattern_corr(a, b, box=DOMAIN):
        a, b = _box(a, box), _box(b, box)
        w = np.cos(np.deg2rad(a.latitude)) * xr.ones_like(a)
        m = np.isfinite(a) & np.isfinite(b)
        aw, bw, ww = a.values[m.values], b.values[m.values], w.values[m.values]
        am, bm = np.average(aw, weights=ww), np.average(bw, weights=ww)
        return float(np.sum(ww * (aw - am) * (bw - bm)) /
                     np.sqrt(np.sum(ww * (aw - am) ** 2) * np.sum(ww * (bw - bm) ** 2)))

    def box_means(self, field):
        return {k: float(_wmean(_box(field, b))) for k, b in BOXES.items()}

    # ---- winter relationships with Flagstaff precipitation ----------------------------------------
    def winter_fields(self):
        return (winter_mean(self.sst_a), winter_mean(self.atm_a.z500), winter_mean(self.atm_a.msl))

    def correlation_maps(self):
        p = self.flg_winter
        out = []
        for f in self.winter_fields():
            yrs = sorted(set(p.index) & set(int(y) for y in f.winter.values if np.isfinite(f.sel(winter=y)).any()))
            fy = f.sel(winter=yrs)
            x = xr.DataArray(p.loc[yrs].values, dims='winter', coords={'winter': yrs})
            out.append(xr.corr(fy, x, dim='winter'))
        return out, len(yrs)

    def index_table(self):
        """Correlation of Flagstaff Nov-Mar precipitation with Nov-Mar means of each index."""
        def wm(s):
            s = s.copy()
            s.index = pd.DatetimeIndex(s.index)
            out = {}
            for y in range(FIRST, pd.Timestamp.now().year):
                w = s.loc[f'{y}-11':f'{y + 1}-03']
                if len(w) == 5:
                    out[y] = w.mean()
            return pd.Series(out)
        idx = {
            'ONI (Nov–Jan)': self.oni[self.oni.index.month == 12].set_axis(
                [t.year for t in self.oni[self.oni.index.month == 12].index]),
            'PDO': wm(self.pdo),
            'North Pacific High central pressure': wm(self.nph.p),
            'North Pacific High centre latitude': wm(self.nph.lat),
        }
        rows = []
        for k, s in idx.items():
            j = pd.concat([s.rename('x'), self.flg_winter.rename('y')], axis=1).dropna()
            r, pval = stats.pearsonr(j.x, j.y)
            rows.append(dict(index=k, n=len(j), r=r, p=pval, years=f'{j.index.min()}–{j.index.max()}'))
        return pd.DataFrame(rows).set_index('index')

    def percentile(self, s, t, same_month=True):
        """Percentile of s[t] among values for the same calendar month since FIRST."""
        ref = s[s.index.year >= FIRST]
        if same_month:
            ref = ref[ref.index.month == pd.Timestamp(t).month]
        return float(stats.percentileofscore(ref.dropna(), s[t]))


# ---- North Pacific jet and the Southwest trough -------------------------------------------------
SW_TROUGH_BOX = dict(lat=(31, 41), lon=(231, 246))   # 31-41N, 129-114W: strongest 500-hPa correlation centre


def nov_mar(s, first=FIRST):
    """Nov-Mar mean of a monthly series, labelled by start year; complete winters only."""
    s = s.copy()
    s.index = pd.DatetimeIndex(s.index)
    out = {}
    for y in range(first, pd.Timestamp.now().year):
        w = s.loc[f'{y}-11':f'{y + 1}-03']
        if len(w) == 5 and w.notna().all():
            out[y] = w.mean()
    return pd.Series(out)


def jet_winter_table(p):
    """Nov-Mar means of jet indices, the Southwest trough index (full-sample and leave-one-out), ONI, PNA and
    Flagstaff Nov-Mar precipitation."""
    oni_dec = p.oni[p.oni.index.month == 12]
    oni_dec.index = oni_dec.index.year
    d = pd.DataFrame({
        'precip': p.flg_winter, 'ONI': oni_dec, 'PNA': nov_mar(p.pna),
        'A10 P1': nov_mar(p.jet['A10'].pcs.PC1), 'A10 P2': nov_mar(p.jet['A10'].pcs.PC2),
        'SW trough': nov_mar(_wmean(_box(p.atm_a.z500, SW_TROUGH_BOX)).to_series()),
    }).loc[FIRST:].dropna()
    d['SW trough (cross-validated)'] = _sw_trough_loo(p, d)
    return d


def _sw_trough_loo(p, d):
    """Leave-one-out: for each winter, choose the 500-hPa grid point most negatively correlated with Flagstaff
    precipitation in the other winters, then take that winter's mean over a 10 x 15 degree box around it."""
    zw = winter_mean(p.atm_a.z500).sel(winter=d.index.values)
    zd = zw.sel(latitude=slice(*DOMAIN['lat']), longitude=slice(*DOMAIN['lon']))
    Z, P = zd.values.reshape(len(d), -1), d.precip.values
    out = []
    for i in range(len(d)):
        m = np.ones(len(d), bool)
        m[i] = False
        Zc, Pc = Z[m] - Z[m].mean(0), P[m] - P[m].mean()
        r = (Zc * Pc[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (Pc ** 2).sum())
        la, lo = np.unravel_index(np.nanargmin(r), zd.shape[1:])
        lat0, lon0 = float(zd.latitude[la]), float(zd.longitude[lo])
        b = dict(lat=(lat0 - 5, lat0 + 5), lon=(lon0 - 7.5, lon0 + 7.5))
        out.append(float(_wmean(_box(zw.isel(winter=i), b))))
    return out


def jet_skill(d):
    rows = []
    X0 = np.c_[np.ones(len(d)), d.ONI]

    def res(v):
        return v - X0 @ np.linalg.lstsq(X0, v, rcond=None)[0]
    for k in ['ONI', 'A10 P2', 'SW trough (cross-validated)']:
        r, pv = stats.pearsonr(d[k], d.precip)
        row = dict(index=k, r=r, p=pv, r_ONI=d[k].corr(d.ONI))
        if k != 'ONI':
            row['partial'], row['p_partial'] = stats.pearsonr(res(d[k].values), res(d.precip.values))
        rows.append(row)
    return pd.DataFrame(rows).set_index('index')


def r2(d, cols):
    X = np.c_[np.ones(len(d)), d[cols]]
    b = np.linalg.lstsq(X, d.precip, rcond=None)[0]
    return float(1 - ((d.precip - X @ b) ** 2).sum() / ((d.precip - d.precip.mean()) ** 2).sum())
