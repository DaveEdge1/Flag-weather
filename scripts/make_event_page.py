"""Create (or refresh) an event page: site/events/<YYYY-MM-DD>.qmd.

    python scripts/make_event_page.py START END "Title" [MAPTIME ...]

START/END are UTC (e.g. 2026-10-09T12). The front matter (description, tags) is computed now;
the page body recomputes figures when rendered. Final pages are frozen; provisional pages
re-execute on every build until ERA5 covers them.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flagweather import page  # noqa: E402
from flagweather.event import Event  # noqa: E402

TEMPLATE = '''---
title: "{title}"
date: {date}
description: "{description}"
categories: [{categories}]
execute:
  freeze: {freeze}
---

```{{python}}
import sys
sys.path.insert(0, '../..')
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown
from flagweather import page
from flagweather.config import SNOTEL
from flagweather.event import Event
from flagweather.figures import events as figs
from flagweather.sources import era5, snotel

try:
    ev = Event('{start}', '{end}', map_times={map_times}).load()
    ok = True
except Exception as err:   # a data source was unreachable at build time
    ok = False
    print(f'*Some data sources could not be reached when this page was built ({{type(err).__name__}}). '
          'The page will be completed by the next scheduled build.*')
```

```{{python}}
#| output: asis
if ok:
    print('## Summary\\n')
    print(page.narrative(ev))
    print('\\n## Rain across the network\\n')
```

```{{python}}
#| fig-cap: "Event rain total at each city and county gauge (circle area proportional to total, labels in inches) over shaded relief (USGS 3DEP), 200-m elevation contours, streams (USGS NHD) and highways (Census TIGER). Orange triangles: SNOTEL sites with event precipitation (in). Square: Flagstaff airport."
if ok:
    figs.gauge_map(ev.net, ev.totals, 'Event rain total (in)', snotel.stations(list(SNOTEL)), ev.snotel_stats().precip)
    plt.show()
```

```{{python}}
#| fig-cap: "Hourly and cumulative rain across the gauge network (median and 10th–90th percentile; cumulative from the start of the event window). When ERA5 is available: IVT, 250-hPa wind speed and 500-hPa height at the ERA5 grid point nearest Flagstaff. Times UTC."
if ok and ev.final:
    figs.network_timeseries(ev.hourly, ev.point.loc[ev.start - pd.Timedelta('1D'):ev.end + pd.Timedelta('1D')],
                            start=ev.start, end=ev.end)
    plt.show()
elif ok:
    figs.network_timeseries(ev.hourly, start=ev.start, end=ev.end)
    plt.show()
```

```{{python}}
#| output: asis
if ok:
    print(page.key_numbers(ev))
```

```{{python}}
#| output: asis
if ok and ev.final:
    print('## Atmospheric setting\\n')
```

```{{python}}
#| output: asis
if ok and ev.final:
    from flagweather import animation
    frames = animation.write_frames(ev, era5.land_sea_mask())
    print(animation.player(frames, ev.era5_stats()['ivt_time'], 'player-{slug}'))
    print('\\n*ERA5 every 6 hours, from 12 h before to 12 h after the event window, opening at the time of peak IVT '
          'at Flagstaff. Shading: integrated vapour transport (IVT); blue contour and arrows: IVT ≥ 250 kg m⁻¹ s⁻¹. '
          'Black: 500-hPa height (60-m interval). Orange: 250-hPa wind 40, 55 and 70 m s⁻¹. ★ Flagstaff; '
          '✕ NHC best-track position. Lower strip: network median hourly rain, with the event window shaded '
          'and the map time marked in orange. Step with the buttons, slider or ← → keys; space plays and pauses.*\\n')
```

```{{python}}
#| output: asis
if ok:
    print('## Climate context\\n')
    print(page.climate(ev))
    print('\\n## Station data\\n')
```

```{{python}}
#| output: asis
if ok:
    print('**Flagstaff airport** (local days; normals 1991–2020; M = not yet reported)\\n')
    print(page.station_table(ev))
    print('\\n\\n**SNOTEL** (water-year precipitation from Oct 1; medians 1991–2020)\\n')
    print(page.snotel_table(ev))
```
'''


def main(start, end, title, *map_times):
    ev = Event(start, end, title, list(map_times)).load()
    text = TEMPLATE.format(slug=ev.slug, title=title, date=ev.days[0].date(), start=start, end=end,
                           description=page.description(ev).replace('"', "'"),
                           categories=', '.join(ev.tags()), freeze='auto' if ev.final else 'false',
                           map_times=list(map_times))
    out = Path(__file__).resolve().parent.parent / 'site' / 'events' / f'{ev.slug}.qmd'
    out.write_text(text, encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
