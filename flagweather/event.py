"""Assemble the data, objective tags and summary numbers for one precipitation event.

An event is a UTC window [start, end]. Rain totals are summed over the window for every
network gauge; ERA5 is used if it covers the window (page "final"), otherwise the page is
"provisional".
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import AR_IVT, DATA, FLAGSTAFF, SNOTEL, TC_RADIUS_KM
from .sources import acis, era5, gauges, nhc, snotel

CACHE = DATA / 'events'
CACHE.mkdir(exist_ok=True)
NAMED = {'TD', 'TS', 'HU', 'DB', 'LO', 'EX', 'SD', 'SS'}  # best-track types kept (incl. remnants)
LOW_RADIUS_KM = 1500   # search radius for a closed 500-hPa low
JET_RADIUS_KM = 1000   # search radius for the strongest 250-hPa wind
TROUGH_WEST = -135     # western limit (deg) of the 500-hPa trough-axis search along Flagstaff's latitude


def _km(lat, lon, lat0=FLAGSTAFF['lat'], lon0=FLAGSTAFF['lon']):
    la1, lo1, la2, lo2 = map(np.deg2rad, (lat0, lon0, lat, lon))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _bearing(lat, lon):
    """Bearing (deg) from Flagstaff to (lat, lon)."""
    la1, lo1, la2, lo2 = map(np.deg2rad, (FLAGSTAFF['lat'], FLAGSTAFF['lon'], lat, lon))
    x = np.sin(lo2 - lo1) * np.cos(la2)
    y = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return float(np.degrees(np.arctan2(x, y)) % 360)


def _from_deg(u, v):
    return float(np.degrees(np.arctan2(-u, -v)) % 360)


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
        self.totals = self.hourly.loc[self.start:self.end].sum()

        # local (MST) calendar days covered by the window
        l0 = (self.start - pd.Timedelta('7h')).normalize()
        l1 = (self.end - pd.Timedelta('7h')).normalize()
        self.days = pd.date_range(l0, l1, freq='D')
        self.airport = acis.daily(l0.date(), l1.date()).join(acis.daily_normals(l0.date(), l1.date()))
        self.wytd = acis.water_year_to_date(l1.date())
        self.precip_record = acis.precip_record()
        sd = snotel.daily(list(SNOTEL), (l0 - pd.Timedelta('1D')).date(), l1.date())
        self.snotel = sd.pivot_table(index=['triplet', 'date'], columns='element', values='value')
        self.snotel_median = sd.pivot_table(index=['triplet', 'date'], columns='element', values='median')

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

    # ---- rain ---------------------------------------------------------------
    def network_stats(self):
        t = self.totals
        h = self.hourly.loc[self.start:self.end]
        med = h.median(axis=1)
        return dict(n=len(t), median=t.median(), p10=t.quantile(0.1), p90=t.quantile(0.9),
                    min=t.min(), max=t.max(), max_site=self.net.loc[t.idxmax(), 'site'],
                    min_site=self.net.loc[t.idxmin(), 'site'],
                    peak_hour=med.idxmax(), peak_hour_median=med.max(), peak_hour_max=h.max(axis=1).max())

    def rain_core(self):
        """Hours spanning the central 80% of the network-mean rain (start of first hour, end of last hour)."""
        m = self.hourly.loc[self.start:self.end].mean(axis=1)
        c = m.cumsum() / m.sum()
        return c[c >= 0.1].index[0], c[c >= 0.9].index[0] + pd.Timedelta('1h')

    # ---- climate context ----------------------------------------------------
    def airport_stats(self):
        a = self.airport
        n = len(self.days)
        rank, nyears, first = acis.rank_in_month(a.pcpn.sum(), n, self.days[-1].month, self.precip_record)
        missing = int(a.pcpn.isna().sum())
        return dict(precip=a.pcpn.sum(), precip_normal=a.pcpn_n.sum(), snow=a.snow.sum(), snow_normal=a.snow_n.sum(),
                    tmax=a.maxt.max(), tmin=a.mint.min(),
                    tmax_dep=(a.maxt - a.maxt_n).mean(), tmin_dep=(a.mint - a.mint_n).mean(),
                    ndays=n, missing=missing, rank=rank, nyears=nyears, first_year=first,
                    wytd=self.wytd[0], wytd_normal=self.wytd[1])

    def snotel_stats(self):
        rows = []
        for trip, name in SNOTEL.items():
            s, med = self.snotel.loc[trip], self.snotel_median.loc[trip]
            before = s.loc[:self.days[0] - pd.Timedelta('1D')]
            during = s.loc[self.days[0]:self.days[-1]]
            dp = during.PREC.max() - (before.PREC.iloc[-1] if len(before) else np.nan)
            last = during.index[-1]
            wy, wy_med = during.PREC.iloc[-1], med.PREC.get(last, np.nan) if 'PREC' in med else np.nan
            rows.append(dict(site=name, precip=dp, wy=wy, wy_median=wy_med,
                             wy_pct=100 * wy / wy_med if wy_med and wy_med > 0 else np.nan,
                             swe_end=during.WTEQ.iloc[-1], snwd_max=during.SNWD.max(),
                             tmin=during.TMIN.min(), tmax=during.TMAX.max()))
        return pd.DataFrame(rows).set_index('site')

    # ---- atmosphere ---------------------------------------------------------
    def era5_stats(self):
        if not self.final:
            return None
        w = self.point.loc[self.start:self.end]
        i = w.ivt.idxmax()
        return dict(ivt_max=w.ivt.max(), ivt_time=i, ivt_dir=w.loc[i, 'ivt_from_deg'],
                    ar_hours=int((w.ivt >= AR_IVT).sum() * 6), ws250_max=w.ws250.max(), z500_min=w.z500.min(),
                    tcwv_max=w.tcwv.max())

    def atmos_setting(self):
        """Objective descriptors of the circulation, centred on the time of peak IVT at Flagstaff."""
        if not self.final:
            return None
        e = self.era5_stats()
        t0, t1 = self.rain_core()
        rain = self.hourly.loc[self.start:self.end].mean(axis=1)
        ivt_h = self.point.ivt.resample('1h').interpolate().reindex(rain.index)
        frac_ar = float(rain[ivt_h >= AR_IVT].sum() / rain.sum()) if rain.sum() > 0 else np.nan

        tp = e['ivt_time']
        s = self.era5.sel(valid_time=tp)
        lat2, lon2 = np.meshgrid(s.latitude.values, s.longitude.values, indexing='ij')
        dist = _km(lat2, lon2)
        f = s.sel(latitude=FLAGSTAFF['lat'], longitude=FLAGSTAFF['lon'], method='nearest')
        out = dict(core_start=t0, core_end=t1, frac_ar=frac_ar, ivt_time=tp,
                   flow500_from=_from_deg(float(f.u500), float(f.v500)),
                   flow500_speed=float(np.hypot(f.u500, f.v500)))

        # 500-hPa trough axis: lowest height along Flagstaff's latitude between TROUGH_WEST and Flagstaff;
        # none if the lowest value is at the western edge of that segment (no minimum inside it)
        row = s.z500.sel(latitude=FLAGSTAFF['lat'], method='nearest').sel(longitude=slice(TROUGH_WEST, FLAGSTAFF['lon']))
        k = int(row.argmin())
        out.update(z500_flg=float(f.z500), trough_at_flg=k == len(row) - 1, trough_lon=None)
        if 0 < k < len(row) - 1:
            out.update(trough_lon=float(row.longitude[k]), trough_z=float(row[k]),
                       trough_km=float(_km(FLAGSTAFF['lat'], float(row.longitude[k]))))
        # closed 500-hPa low: local minimum (5-degree neighbourhood, away from the domain edge) within LOW_RADIUS_KM
        zz = s.z500.values
        is_min = zz == ndimage.minimum_filter(zz, size=21, mode='nearest')
        is_min[:10, :] = is_min[-10:, :] = is_min[:, :10] = is_min[:, -10:] = False
        cand = np.argwhere(is_min & (dist <= LOW_RADIUS_KM))
        out['closed_low'] = None
        if len(cand):
            i = tuple(min(cand, key=lambda c: dist[tuple(c)]))
            out['closed_low'] = dict(z=float(zz[i]), lat=float(lat2[i]), lon=float(lon2[i]), km=float(dist[i]),
                                     bearing=_bearing(lat2[i], lon2[i]))
        wsp = np.where(dist <= JET_RADIUS_KM, s.ws250.values, -np.inf)
        j = np.unravel_index(np.argmax(wsp), wsp.shape)
        out.update(jet_speed=float(wsp[j]), jet_km=float(dist[j]), jet_bearing=_bearing(lat2[j], lon2[j]))

        w = self.point.loc[self.start:self.end]
        out.update(ws250_flg_max=float(w.ws250.max()), ws250_flg_time=w.ws250.idxmax())

        # is the IVT >= threshold region over Flagstaff continuous with a tropical cyclone at the IVT peak?
        out['tc_connected'] = []
        if e['ivt_max'] >= AR_IVT:
            lab, _ = ndimage.label(s.ivt.values >= AR_IVT)
            fl = lab[np.unravel_index(np.argmin(dist), dist.shape)]
            bt = self.best_track[(self.best_track.time - tp).abs() <= pd.Timedelta('3h')]
            for _, r in bt[bt.type.isin(NAMED)].iterrows():
                d_tc = _km(lat2, lon2, r.lat, r.lon)
                if fl and (lab[d_tc <= 150] == fl).any():
                    out['tc_connected'].append(dict(name=r['name'], lat=r.lat, lon=r.lon, type=r.type,
                                                    km=float(_km(r.lat, r.lon)), bearing=_bearing(r.lat, r.lon)))
        return out

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
