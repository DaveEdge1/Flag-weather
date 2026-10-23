"""Sea-surface temperature off Baja California and its relation to tropical cyclones and Flagstaff autumn rain.

Baja box: 18-32N, 122-108W (Pacific side of the peninsula and the Gulf of California), ERSSTv5 monthly,
anomalies relative to 1991-2020.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .pacific import _box, _wmean, anomaly, oni
from .sources import acis, ersst, nhc

BOX = dict(lat=(18, 32), lon=(238, 252))
TROPICAL = {'TD', 'TS', 'HU'}


class Baja:
    def __init__(self, year=2026, storms=('Polo', 'Rachel')):
        self.year, self.storms = year, storms

    def load(self):
        self.sst = ersst.load()
        self.anom = anomaly(self.sst)
        self.clim = self.sst.sel(time=slice('1991', '2020')).groupby('time.month').mean('time')
        self.box_a = _wmean(_box(self.anom, BOX)).to_series()
        self.box_t = _wmean(_box(self.sst, BOX)).to_series()
        self.bt = nhc.best_track(self.year)
        self.oni = oni()
        pr = acis.precip_record()
        self.daily = pr
        mon = pr.resample('MS').agg(lambda x: x.sum(min_count=int(len(x) * 0.9)))
        self.sepoct = pd.Series({y: mon.loc[f'{y}-09':f'{y}-10'].sum() for y in range(1950, self.year)
                                 if mon.loc[f'{y}-09':f'{y}-10'].notna().all()})
        self.latest = pd.Timestamp(self.sst.time.values[-1])
        return self

    # ---- box index ---------------------------------------------------------------------------
    def month_table(self, months):
        rows = []
        for t in months:
            ref = self.box_a[(self.box_a.index.month == t.month) & (self.box_a.index.year >= 1950)]
            rows.append(dict(month=t, sst=self.box_t[t], anom=self.box_a[t],
                             rank=int((ref > self.box_a[t]).sum() + 1), n=len(ref),
                             prev_max=ref[ref.index.year < t.year].max(),
                             prev_year=int(ref[ref.index.year < t.year].idxmax().year)))
        out = pd.DataFrame(rows).set_index('month')
        out['prev_year'] = out.prev_year.astype(int)
        return out

    def season_series(self, months=(7, 8, 9)):
        a = self.box_a[self.box_a.index.month.isin(months)]
        g = a.groupby(a.index.year)
        return g.mean()[g.count() == len(months)]

    # ---- storms ------------------------------------------------------------------------------
    def track_table(self):
        """Best-track positions every 12 h north of 18N while tropical, with SST of that month (or the latest
        available month) at the nearest ERSST cell and its 1991-2020 normal."""
        b = self.bt[self.bt['name'].isin(self.storms) & (self.bt.time.dt.hour % 12 == 0) & (self.bt.lat >= 18)]
        rows = []
        for _, r in b.sort_values('time').iterrows():
            mon = min(pd.Timestamp(r.time.year, r.time.month, 1), self.latest)
            s = float(self.sst.sel(time=mon).sel(latitude=r.lat, longitude=r.lon % 360, method='nearest'))
            c = float(self.clim.sel(month=mon.month).sel(latitude=r.lat, longitude=r.lon % 360, method='nearest'))
            rows.append(dict(storm=r['name'], time=r.time, lat=r.lat, lon=r.lon, type=r.type, vmax=r.vmax_kt,
                             sst=s, normal=c, sst_month=mon))
        return pd.DataFrame(rows)

    def tracks(self):
        b = self.bt[self.bt['name'].isin(self.storms)].sort_values('time')
        return {n: g for n, g in b.groupby('name')}

    # ---- history -----------------------------------------------------------------------------
    def history(self):
        sep = self.box_a[self.box_a.index.month == 9]
        sep.index = sep.index.year
        o = self.oni[self.oni.index.month == 8]   # JAS ONI
        o.index = o.index.year
        return pd.DataFrame({'precip': self.sepoct, 'baja_sep': sep, 'oni_jas': o}).loc[1950:self.year - 1].dropna()

    def history_stats(self):
        d = self.history()
        r, p = stats.pearsonr(d.baja_sep, d.precip)
        X = np.c_[np.ones(len(d)), d.oni_jas]

        def res(v):
            return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        rp, pp = stats.pearsonr(res(d.baja_sep.values), res(d.precip.values))
        top = d[d.baja_sep >= d.baja_sep.quantile(0.8)]
        return dict(n=len(d), r=r, p=p, r_oni=d.oni_jas.corr(d.precip), r_baja_oni=d.baja_sep.corr(d.oni_jas),
                    r_partial=rp, p_partial=pp, n_top=len(top), top_above=int((top.precip > d.precip.median()).sum()),
                    top_median=top.precip.median(), median=d.precip.median(),
                    driest_top=top.precip.idxmin(), driest_top_val=top.precip.min(),
                    driest_top_rank=int((d.baja_sep > d.baja_sep[top.precip.idxmin()]).sum() + 1))

    def current_autumn_precip(self):
        """Airport precipitation from Sep 1 of this year through the last reported day."""
        s = self.daily.loc[f'{self.year}-09-01':].dropna()
        total = s.sum()
        rank = int((self.sepoct > total).sum() + 1)
        return total, s.index[-1], rank, len(self.sepoct)

    def isotherm_lat(self, month, lons=(-120, -118, -116), t=26.0):
        """Northernmost latitude (18-35N) at which SST >= t along each longitude, for `month` and its normal."""
        out = {}
        for lo in lons:
            row = {}
            for lab, f in (('obs', self.sst.sel(time=month)), ('normal', self.clim.sel(month=month.month))):
                col = f.sel(longitude=lo % 360, method='nearest').sel(latitude=slice(18, 35))
                warm = col.latitude.values[col.values >= t]
                row[lab] = float(warm.max()) if len(warm) else float('nan')
            out[lo] = row
        return pd.DataFrame(out).T
