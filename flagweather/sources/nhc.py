"""NHC ATCF best-track positions (East Pacific). Operational; final tracks come after the season."""
import urllib.request

import pandas as pd

from ..config import DATA

BTK = 'https://ftp.nhc.noaa.gov/atcf/btk/'


def best_track(year, basin='ep', max_storms=40):
    rows = []
    for n in range(1, max_storms + 1):
        try:
            txt = urllib.request.urlopen(f'{BTK}b{basin}{n:02d}{year}.dat', timeout=30).read().decode()
        except Exception:
            continue
        for ln in txt.splitlines():
            p = [x.strip() for x in ln.split(',')]
            if len(p) < 28:
                continue
            lat = int(p[6][:-1]) / 10 * (1 if p[6][-1] == 'N' else -1)
            lon = int(p[7][:-1]) / 10 * (-1 if p[7][-1] == 'W' else 1)
            rows.append(dict(id=f'{basin}{n:02d}', name=p[27].title(), time=pd.Timestamp(p[2][:8] + 'T' + p[2][8:]),
                             lat=lat, lon=lon, vmax_kt=int(p[8]), type=p[10]))
    bt = pd.DataFrame(rows).drop_duplicates(['id', 'time'])
    bt = bt[~bt['name'].str.upper().str.contains('INVEST|GENESIS')]
    bt.to_csv(DATA / f'nhc_bt_{basin}{year}.csv', index=False)
    return bt
