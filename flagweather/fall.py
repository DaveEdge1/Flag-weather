"""Fall (September–November) season diagnostics for the seasonal synopsis: Flagstaff precipitation and
temperature against the record, the season's events, the cut-off-low regime and the ENSO/PDO state."""
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from .event import Event
from .pacific import BOXES, SW_TROUGH_BOX, Pacific, _box, _wmean
from .sources import acis, era5

FIRST = 1950
# Closed 500-hPa lows counted in this box (the California coast and the Southwest): lat, lon (deg E, -180..180)
CUTOFF_BOX = dict(lat=(25, 42), lon=(-132, -112))


def _ord(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


class Fall:
    def __init__(self, year=2026, events=()):
        self.year = year
        self.event_windows = events        # [(start, end, moisture-source note), ...]

    def load(self):
        y = self.year
        self.record = acis.precip_record()
        self.through = min(self.record.last_valid_index(), pd.Timestamp(f'{y}-11-30'))
        self.daily = acis.daily(f'{y}-09-01', self.through.date()).join(
            acis.daily_normals(f'{y}-09-01', self.through.date()))
        self.p = Pacific().load()
        return self

    # ---- precipitation and temperature ---------------------------------------------------------
    def season_total(self, y, through=None):
        end = (through or pd.Timestamp(f'{y}-11-30')).replace(year=y)
        s = self.record.loc[f'{y}-09-01':end]
        return s.sum() if s.notna().sum() >= 0.9 * len(s) else np.nan

    def full_falls(self, first=None):
        first = first or int(self.record.index.year.min()) + 1   # record starts mid-September 1898
        return pd.Series({y: self.season_total(y) for y in range(first, self.year)}).dropna()

    def to_date(self):
        """Season to date: total, normal, rank among the same dates since FIRST, and against full falls."""
        tot = self.season_total(self.year, self.through)
        same = pd.Series({y: self.season_total(y, self.through) for y in range(FIRST, self.year + 1)}).dropna()
        full = self.full_falls()
        d = self.daily
        return dict(total=tot, normal=d.pcpn_n.sum(), through=self.through,
                    rank_same=int((same > tot).sum() + 1), n_same=len(same),
                    full_max=full.max(), full_max_year=int(full.idxmax()), full_first=int(full.index.min()),
                    full_n=len(full), full_median=full.median(), exceeds_full=bool(tot > full.max()),
                    snow=d.snow.sum(), snow_normal=d.snow_n.sum(),
                    tdep=(d.avgt - (d.maxt_n + d.mint_n) / 2).mean(),
                    wet_days=int((d.pcpn >= 0.01).sum()), inch_days=d.pcpn[d.pcpn >= 1])

    def month_table(self):
        d = self.daily
        rows = []
        for m, g in d.groupby(d.index.month):
            t = g.index[0]
            mon = self.record[self.record.index.month == m]
            totals = mon.groupby(mon.index.year).sum(min_count=25)
            complete = len(g) == t.days_in_month
            rank = int((totals.drop(self.year, errors='ignore') > g.pcpn.sum()).sum() + 1) if complete else None
            rows.append({'Month': f'{t:%B}' + ('' if complete else f' 1–{g.index[-1].day}'),
                         'Precipitation (in)': f'{g.pcpn.sum():.2f}', 'Normal (in)': f'{g.pcpn_n.sum():.2f}',
                         'Rank (wettest)': f'{_ord(rank)} of {totals.notna().sum()}' if rank else '–',
                         'Mean temp. vs normal (°F)': f'{(g.avgt - (g.maxt_n + g.mint_n) / 2).mean():+.1f}',
                         'Snowfall (in)': f'{g.snow.sum():.1f}', 'Normal snowfall (in)': f'{g.snow_n.sum():.1f}'})
        return pd.DataFrame(rows)

    def daily_ranks(self, date):
        d = self.record.dropna()
        v = d[date]
        same_month = d[d.index.month == pd.Timestamp(date).month]
        return dict(value=v, rank_all=int((d > v).sum() + 1), rank_month=int((same_month > v).sum() + 1),
                    first=int(d.index.year.min()))

    def cumulative(self, years):
        """Cumulative Sep 1 - Nov 30 precipitation by day of season for the given years, plus the median."""
        out = {}
        for y in list(years) + [self.year]:
            s = self.record.loc[f'{y}-09-01':f'{y}-11-30'].fillna(0).cumsum()
            out[y] = s.set_axis((s.index - pd.Timestamp(f'{y}-09-01')).days)
        allc = pd.DataFrame({y: self.record.loc[f'{y}-09-01':f'{y}-11-30'].fillna(0).cumsum().set_axis(range(91))
                             for y in self.full_falls(FIRST).index})
        out['median'] = allc.median(axis=1)
        return out

    # ---- events --------------------------------------------------------------------------------
    def event_table(self):
        rows = []
        for start, end, source in self.event_windows:
            ev = Event(start, end).load()
            n, a = ev.network_stats(), ev.airport_stats()
            e = ev.era5_stats()
            d0, d1 = ev.days[0], ev.days[-1]
            dates = (f'{d0:%b} {d0.day}' if d0 == d1 else f'{d0:%b} {d0.day}–{d1.day}' if d0.month == d1.month
                     else f'{d0:%b} {d0.day} – {d1:%b} {d1.day}')
            rows.append({'Event': f'[{dates}](../events/{ev.slug}.qmd)',
                         'Network median (in)': f'{n["median"]:.2f}',
                         'Airport (in)': f'{a["precip"]:.2f}',
                         'Peak IVT at Flagstaff (kg m⁻¹ s⁻¹)': f'{e["ivt_max"]:.0f}' if e else 'not yet in ERA5',
                         'Moisture source (NWS Flagstaff)': source})
        return pd.DataFrame(rows)

    # ---- circulation ---------------------------------------------------------------------------
    def month_anoms(self, t):
        p, t = self.p, pd.Timestamp(t)
        z, m = p.atm_a.z500.sel(time=t), p.atm_a.msl.sel(time=t)
        series = {k: _wmean(_box(p.atm_a.z500, b)).to_series() for k, b in BOXES.items()}
        series['SW trough box'] = _wmean(_box(p.atm_a.z500, SW_TROUGH_BOX)).to_series()
        out = {}
        for k, s in series.items():
            same = s[(s.index.month == t.month) & (s.index.year >= FIRST)]
            out[k] = dict(z=float(s[t]), rank_high=int((same > s[t]).sum() + 1), n=len(same))
        out['GoA msl'] = float(_wmean(_box(m, BOXES['Gulf of Alaska'])))
        return out

    def cutoff_lows(self, start, end):
        """6-hourly ERA5 times with a closed 500-hPa low (local minimum over a 5-degree neighbourhood) in CUTOFF_BOX."""
        z = era5.load(start, end).z500
        lat, lon = z.latitude.values, z.longitude.values
        rows = []
        for t in z.valid_time.values:
            a = z.sel(valid_time=t).values
            mn = a == ndimage.minimum_filter(a, size=21, mode='nearest')
            mn[:10, :] = mn[-10:, :] = mn[:, :10] = mn[:, -10:] = False
            for i, k in np.argwhere(mn):
                if (CUTOFF_BOX['lat'][0] <= lat[i] <= CUTOFF_BOX['lat'][1]
                        and CUTOFF_BOX['lon'][0] <= lon[k] <= CUTOFF_BOX['lon'][1]):
                    rows.append(dict(time=pd.Timestamp(t), lat=lat[i], lon=lon[k], z=a[i, k]))
        df = pd.DataFrame(rows)
        return dict(n_times=z.valid_time.size, n_with=df.time.nunique() if len(df) else 0,
                    start=pd.Timestamp(z.valid_time.values[0]), end=pd.Timestamp(z.valid_time.values[-1]), lows=df)

    # ---- ENSO and PDO --------------------------------------------------------------------------
    def enso(self):
        p = self.p
        aso = p.oni[p.oni.index.month == 9]
        aso.index = aso.index.year
        pdo = p.pdo[p.pdo.index.month.isin([9, 10])]
        pdo = pdo.groupby(pdo.index.year).mean()
        son = self.full_falls(FIRST)
        j = pd.concat([aso.rename('oni'), son.rename('p')], axis=1).dropna()
        r, pv = stats.pearsonr(j.oni, j.p)
        en = j[j.oni >= 1.0]
        strong = [y for y, v in aso.items() if v >= 1.5 and y < self.year]
        conflict = [y for y in aso.index if y < self.year and aso[y] >= 1.0 and pdo.get(y, 0) <= -1.0]
        months = pd.DatetimeIndex([f'{self.year}-09-01', f'{self.year}-10-01'])
        sst_c, z_c, _ = p.analog_composite([9, 10], strong)
        sst_now, z_now = p.sst_a.sel(time=months).mean('time'), p.atm_a.z500.sel(time=months).mean('time')
        return dict(oni=aso[self.year], oni_rank=int((aso > aso[self.year]).sum() + 1), oni_n=len(aso),
                    oni_first=int(aso.index.min()), oni_next=aso.drop(self.year).max(),
                    oni_next_year=int(aso.drop(self.year).idxmax()),
                    nino34=float(p.nino34.iloc[-1]), nino34_t=p.nino34.index[-1],
                    pdo=float(p.pdo.iloc[-1]), pdo_t=p.pdo.index[-1], pdo_so=pdo[self.year],
                    pdo_pct=p.percentile(p.pdo, p.pdo.index[-1]),
                    pna=float(p.pna.iloc[-1]), pna_t=p.pna.index[-1],
                    r=r, p=pv, n=len(j), en_n=len(en), en_median=en.p.median(),
                    en_above=int((en.p > son.median()).sum()), median=son.median(),
                    strong={y: (aso[y], pdo[y], son.get(y, np.nan), p.flg_winter.get(y, np.nan)) for y in strong},
                    conflict={y: (aso[y], pdo[y], son.get(y, np.nan), p.flg_winter.get(y, np.nan)) for y in conflict},
                    winter_median=p.flg_winter.median(),
                    pc_sst=p.pattern_corr(sst_now, sst_c), pc_z=p.pattern_corr(z_now, z_c),
                    al=p.al.iloc[-1], al_t=p.al.index[-1])
