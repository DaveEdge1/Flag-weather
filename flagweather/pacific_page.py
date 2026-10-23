"""Text for the Pacific page (observations computed from the data)."""
import numpy as np
import pandas as pd

from .pacific import BOXES, CLIM, FIRST, enso_word


def _rank(s, t, highest=True):
    ref = s[(s.index.year >= FIRST) & (s.index.month == pd.Timestamp(t).month)].dropna()
    r = int((ref > s[t]).sum() + 1) if highest else int((ref < s[t]).sum() + 1)
    return r, len(ref)


def _ord(n):
    return f'{n}{"th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'


def _lon(x):
    return f'{x:.0f}°E' if x <= 180 else f'{360 - x:.0f}°W'


def _ranktxt(s, t, label):
    hi, n = _rank(s, t, True)
    lo, _ = _rank(s, t, False)
    word, k = ('highest', hi) if hi <= lo else ('lowest', lo)
    return f'{word if k == 1 else _ord(k) + " " + word} {label} of {n} since {FIRST}'


def summary(p, months3):
    t = months3[-1]
    mon = f'{t:%B %Y}'
    o = p.oni.dropna()
    ot = o.index[-1]
    seas = {1: 'DJF', 2: 'JFM', 3: 'FMA', 4: 'MAM', 5: 'AMJ', 6: 'MJJ', 7: 'JJA', 8: 'JAS', 9: 'ASO', 10: 'SON',
            11: 'OND', 12: 'NDJ'}[ot.month]
    n34 = p.nino34
    lines = [
        f'- **ENSO:** the latest ONI ({seas} {ot.year}) is {o.iloc[-1]:+.2f} °C, in NOAA\'s {enso_word(o.iloc[-1])} range. '
        f'ERSSTv5 Niño3.4 for {mon} is {n34[t]:+.2f} °C relative to {CLIM[0]}–{CLIM[1]}, the '
        f'{_ranktxt(n34, t, t.strftime("%B") + " value")}.',
    ]
    pdo = p.pdo.dropna()
    lines.append(f'- **PDO:** {pdo.iloc[-1]:+.2f} in {pdo.index[-1]:%B %Y}, the '
                 f'{_ranktxt(pdo, pdo.index[-1], pdo.index[-1].strftime("%B") + " value")}.')
    pna = p.pna.dropna()
    lines.append(f'- **PNA:** {pna.iloc[-1]:+.2f} in {pna.index[-1]:%B %Y}; '
                 f'{months3[0]:%b}–{months3[-1]:%b} mean {pna.reindex(months3).mean():+.2f}.')
    al = p.al.loc[t]
    lines.append(f'- **Aleutian Low ({mon}):** lowest sea-level pressure in the North Pacific box was {al.p:.0f} hPa at '
                 f'{al.lat:.0f}°N {_lon(al.lon)} ({al.p_anom:+.1f} hPa relative to the {t:%B} mean for that box). '
                 f'North Pacific Index anomaly {p.npi_a[t]:+.1f} hPa, the {_ranktxt(p.npi_a, t, t.strftime("%B") + " value")} '
                 f'(negative means a deeper Aleutian Low).')
    h = p.nph.loc[t]
    lines.append(f'- **North Pacific High ({mon}):** highest sea-level pressure in the subtropical box was {h.p:.0f} hPa '
                 f'at {h.lat:.0f}°N {_lon(h.lon)} ({h.p_anom:+.1f} hPa relative to the {t:%B} mean).')
    return '\n'.join(lines)


def analog_text(p, months3, years, comp):
    cur = p.mean_anom(months3)[:3]
    names = ['SST', '500-hPa height', 'sea-level pressure']
    pcs = [p.pattern_corr(cur[0], comp[0]), p.pattern_corr(cur[1], comp[1]), p.pattern_corr(cur[2], comp[2])]
    bc, bk = p.box_means(cur[1]), p.box_means(comp[1])
    lab = f'{months3[0]:%b}–{months3[-1]:%b}'
    lines = [
        f'- **Analog years** (Nov–Jan ONI ≥ 1.5 °C, {FIRST}–present): ' + ', '.join(f'{y}–{str(y + 1)[2:]}' for y in years) + '.',
        f'- **Pattern correlation, {lab} {months3[-1].year} with the analog mean for {lab}** (domain 10°S–70°N, '
        f'120°E–60°W): ' + '; '.join(f'{n} {r:+.2f}' for n, r in zip(names, pcs)) + '.',
    ] + [f'- **500-hPa height anomaly, {k}:** {bc[k]:+.0f} m in {lab} {months3[-1].year}; analog mean {bk[k]:+.0f} m.'
         for k in BOXES]
    return '\n'.join(lines)


def winter_table(p, years):
    rows = []
    clim = p.flg_winter.loc[1950:]
    for y in years:
        if y in p.flg_winter.index:
            v = p.flg_winter[y]
            rows.append({'Winter': f'{y}–{str(y + 1)[2:]}', 'Nov–Jan ONI (°C)': f'{p.oni.get(pd.Timestamp(y, 12, 1)):+.2f}',
                         'Flagstaff Nov–Mar (in)': f'{v:.2f}', '% of 1950–present mean': f'{100 * v / clim.mean():.0f}'})
    return pd.DataFrame(rows).to_markdown(index=False)


def index_table(p):
    t = p.index_table()
    t['r'] = t.r.map('{:+.2f}'.format)
    t['p'] = t.p.map(lambda v: '< 0.001' if v < 0.001 else f'{v:.3f}')
    return t[['r', 'p', 'n', 'years']].rename(columns={'r': 'r with Flagstaff Nov–Mar precip'}).to_markdown()
