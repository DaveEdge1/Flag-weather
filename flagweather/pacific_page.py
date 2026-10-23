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
    return pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True)


def index_table(p):
    t = p.index_table()
    t['r'] = t.r.map('{:+.2f}'.format)
    t['p'] = t.p.map(lambda v: '< 0.001' if v < 0.001 else f'{v:.3f}')
    return t[['r', 'p', 'n', 'years']].rename(columns={'r': 'r with Flagstaff Nov–Mar precip'}).to_markdown(disable_numparse=True)


# ---- North Pacific jet -----------------------------------------------------------------------------
def npj_regime(pc1, pc2):
    """Regime by the dominant component (quadrants bounded by the diagonals); 'near the origin' inside the
    unit circle."""
    if np.hypot(pc1, pc2) < 1:
        return 'near the origin'
    if abs(pc1) >= abs(pc2):
        return 'jet extension' if pc1 > 0 else 'jet retraction'
    return 'poleward shift' if pc2 > 0 else 'equatorward shift'


def jet_current(p):
    w = p.jet['W19'].pcs.dropna()
    a = p.jet['A10'].pcs.dropna()
    lines = []
    for t, r in w.iloc[-3:].iterrows():
        season = '' if t.month in (9, 10, 11, 12, 1, 2, 3, 4, 5) else ' (outside the Sep–May season of the EOFs)'
        lines.append(f'{t:%B %Y}: PC1 {r.PC1:+.2f}, PC2 {r.PC2:+.2f}, {npj_regime(r.PC1, r.PC2)}{season}')
    djfm = a[a.index.month.isin([12, 1, 2, 3])]
    yw = djfm.index[-1].year if djfm.index[-1].month <= 3 else djfm.index[-1].year + 1
    wmean = djfm.loc[f'{yw - 1}-12':f'{yw}-03'].PC2.mean()
    out = ['- **North Pacific jet (Winters et al. EOFs), last three months:** ' + '; '.join(lines) + '.',
           f'- **Athanasiadis P2** is defined for December–March. Its most recent Dec–Mar mean ({yw - 1}–{str(yw)[2:]}) '
           f'was {wmean:+.2f}; the next value will be available after December {p.latest_atm.year}.']
    return '\n'.join(out)


def jet_table(skill):
    names = {'ONI': 'ONI (Nov–Jan)', 'PNA': 'PNA', 'A10 P1': 'Athanasiadis P1 (≈ PNA)',
             'A10 P2': 'Athanasiadis P2 (+: southward-displaced subtropical jet)',
             'NPJ extension': 'NPJ EOF1 (+: jet extension)', 'NPJ poleward shift': 'NPJ EOF2 (+: poleward shift)',
             'SW trough (cross-validated)': 'Southwest trough (cross-validated)'}
    t = pd.DataFrame({
        'r with Flagstaff precip': skill.r.map('{:+.2f}'.format),
        'p': skill.p.map(lambda v: '< 0.001' if v < 0.001 else f'{v:.3f}'),
        'r with ONI': skill.r_ONI.map('{:+.2f}'.format),
        'r with PNA': skill.r_PNA.map('{:+.2f}'.format),
        'partial r, ONI removed': skill.partial.map(lambda v: '—' if pd.isna(v) else f'{v:+.2f}'),
    })
    t.index = [names[i] for i in t.index]
    return t.to_markdown(disable_numparse=True)
