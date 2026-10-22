"""Assemble the data, objective tags and summary numbers for one precipitation event.

An event is a UTC window [start, end]. Rain totals are summed over the window for every
network gauge; ERA5 is used if it covers the window (page "final"), otherwise the page is
"provisional".
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import AR_IVT, DATA, FLAGSTAFF, SNOTEL, TC_RADIUS_KM
from .sources import acis, era5, gauges, nhc, snotel

CACHE = DATA / 'events'
CACHE.mkdir(exist_ok=True)
NAMED = {'TD', 'TS', 'HU', 'DB', 'LO', 'EX', 'SD', 'SS'}  # best-track types kept (incl. remnants)


def _km(lat, lon):
    la1, lo1, la2, lo2 = map(np.deg2rad, (FLAGSTAFF['lat'], FLAGSTAFF['lon'], lat, lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@dataclass
class Event:
    start: str           # UTC, e.g. '2026-10-09T12'
    end: str             # UTC, inclusive
    title: str = ''
    map_times: list = field(default_factory=list)   # optional override (UTC)

    def __post_init__(self):
        self.start, self.end = pd.Timestamp(self.start), pd.Timestamp(self.end)
        self.slug = self.start.strftime('%Y-%m-%d')

    # ---- data ---------------------------------------------------------------
    def load(self):
        self.net = gauges.network()
        cache = CACHE / f'{self.slug}_hourly.csv'
        complete = self.end < pd.Timestamp.now('UTC').tz_localize(None) - pd.Timedelta('1D')
        if complete and cache.exists():
            self.hourly = pd.read_csv(cache, index_col=0, parse_dates=True)
        else:
            d0, d1 = (self.start - pd.Timedelta('1D')).date(), (self.end + pd.Timedelta('1D')).date()
            self.hourly = gauges.hourly(self.net.index, d0, d1)
            if complete:
                self.hourly.to_csv(cache)
        self.hourly = self.hourly[[c for c in self.hourly.columns if c in self.net.index]]
        win = self.hourly.loc[self.start:self.end]
        self.totals = win.sum()

        # local (MST) calendar days covered by the window
        l0 = (self.start - pd.Timedelta('7h')).normalize()
        l1 = (self.end - pd.Timedelta('7h')).normalize()
        self.days = pd.date_range(l0, l1, freq='D')
        self.airport = acis.daily(l0.date(), l1.date())
        sd = snotel.daily(list(SNOTEL), (l0 - pd.Timedelta('1D')).date(), l1.date())
        self.snotel = sd.pivot_table(index=['triplet', 'date'], columns='element', values='value')

        self.best_track = nhc.best_track(self.start.year)
        self.final = era5.covers(self.start, self.end)
        if self.final:
            self.era5 = era5.load(self.start - pd.Timedelta('1D'), self.end + pd.Timedelta('1D'))
            p = self.era5.sel(latitude=FLAGSTAFF['lat'], longitude=FLAGSTAFF['lon'], method='nearest')
            pt = p[['ivt', 'tcwv', 'ws250', 'z500']].to_dataframe()[['ivt', 'tcwv', 'ws250', 'z500']]
            pt['ivt_from_deg'] = (np.degrees(np.arctan2(-p.ivt_u, -p.ivt_v)) % 360).values
            pt.attrs['grid_point'] = (float(p.latitude), float(p.longitude))
            self.point = pt
        return self

    # ---- summary numbers ----------------------------------------------------
    def network_stats(self):
        t = self.totals
        med = self.hourly.loc[self.start:self.end].median(axis=1)
        return dict(n=len(t), median=t.median(), p10=t.quantile(0.1), p90=t.quantile(0.9),
                    min=t.min(), max=t.max(), max_site=self.net.loc[t.idxmax(), 'site'],
                    min_site=self.net.loc[t.idxmin(), 'site'],
                    peak_hour=med.idxmax(), peak_hour_median=med.max(), peak_hour_max=self.hourly.loc[self.start:self.end].max(axis=1).max())

    def airport_stats(self):
        a = self.airport
        return dict(precip=a.pcpn.sum(), snow=a.snow.sum(), tmax=a.maxt.max(), tmin=a.mint.min())

    def snotel_stats(self):
        rows = []
        for trip, name in SNOTEL.items():
            s = self.snotel.loc[trip]
            before = s.loc[:self.days[0] - pd.Timedelta('1D')]
            during = s.loc[self.days[0]:self.days[-1]]
            dp = during.PREC.max() - (before.PREC.iloc[-1] if len(before) else np.nan)
            rows.append(dict(site=name, precip=dp, swe_end=during.WTEQ.iloc[-1], snwd_max=during.SNWD.max(),
                             tmin=during.TMIN.min(), tmax=during.TMAX.max()))
        return pd.DataFrame(rows).set_index('site')

    def era5_stats(self):
        if not self.final:
            return None
        w = self.point.loc[self.start:self.end]
        i = w.ivt.idxmax()
        return dict(ivt_max=w.ivt.max(), ivt_time=i, ivt_dir=w.loc[i, 'ivt_from_deg'],
                    ar_hours=int((w.ivt >= AR_IVT).sum() * 6), ws250_max=w.ws250.max(), z500_min=w.z500.min(),
                    tcwv_max=w.tcwv.max())

    def tropical_cyclones(self):
        """Closest approach of each storm within TC_RADIUS_KM during [start - 1 day, end]."""
        bt = self.best_track
        bt = bt[(bt.time >= self.start - pd.Timedelta('1D')) & (bt.time <= self.end) & bt.type.isin(NAMED)].copy()
        if bt.empty:
            return pd.DataFrame()
        bt['km'] = _km(bt.lat, bt.lon)
        bt = bt[bt.km <= TC_RADIUS_KM]
        if bt.empty:
            return pd.DataFrame()
        near = bt.loc[bt.groupby('name').km.idxmin()]
        peak = self.best_track.groupby('name').vmax_kt.max()
        near = near.assign(peak_kt=near['name'].map(peak))
        return near[['name', 'time', 'lat', 'lon', 'km', 'type', 'vmax_kt', 'peak_kt']].sort_values('km')

    def tags(self):
        tags = []
        e = self.era5_stats()
        if e is not None:
            tags.append(f'IVT ≥ {AR_IVT}' if e['ivt_max'] >= AR_IVT else f'IVT < {AR_IVT}')
        if not self.tropical_cyclones().empty:
            tags.append('Tropical cyclone nearby')
        if self.airport_stats()['snow'] > 0 or self.snotel_stats().swe_end.max() > 0:
            tags.append('Snow')
        tags.append('Final' if self.final else 'Provisional')
        return tags

    def default_map_times(self):
        if self.map_times:
            return [pd.Timestamp(t) for t in self.map_times]
        c = self.network_stats()['peak_hour'].round('6h')
        step = pd.Timedelta('24h') if self.end - self.start > pd.Timedelta('36h') else pd.Timedelta('12h')
        times = [c - step, c, c + step]
        lo, hi = self.era5.valid_time.values[[0, -1]]
        return [min(max(t, pd.Timestamp(lo)), pd.Timestamp(hi)) for t in times]
