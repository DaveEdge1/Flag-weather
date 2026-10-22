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

ev = Event('{start}', '{end}', map_times={map_times}).load()
Markdown(page.summary(ev))
```

## Rain across the network

```{{python}}
#| fig-cap: "Event rain total at each city and county gauge (circle area proportional to total, labels in inches). Triangles: SNOTEL sites with event precipitation (in). Square: Flagstaff airport."
figs.gauge_map(ev.net, ev.totals, 'Event rain total (in)', snotel.stations(list(SNOTEL)), ev.snotel_stats().precip)
plt.show()
```

```{{python}}
#| fig-cap: "Hourly rain across the gauge network (median and 10th–90th percentile). When ERA5 is available: IVT, 250-hPa wind speed and 500-hPa height at the ERA5 grid point nearest Flagstaff."
if ev.final:
    figs.network_timeseries(ev.hourly, ev.point.loc[ev.start - pd.Timedelta('1D'):ev.end + pd.Timedelta('1D')])
else:
    figs.network_timeseries(ev.hourly, start=ev.start, end=ev.end)
plt.show()
```

```{{python}}
#| output: asis
if ev.final:
    print('## Atmospheric setting\\n')
```

```{{python}}
#| fig-cap: "ERA5 integrated vapour transport (IVT), 500-hPa height and 250-hPa wind speed. Blue contour and arrows: IVT ≥ 250 kg m⁻¹ s⁻¹. ✕: NHC best-track positions."
if ev.final:
    figs.map_panels(ev.era5, era5.land_sea_mask(), ev.default_map_times(), ev.best_track)
    plt.show()
```

## Station data

```{{python}}
#| output: asis
print('**Flagstaff airport (daily, local days)**\\n')
print(page.station_table(ev))
print('\\n\\n**SNOTEL**\\n')
print(page.snotel_table(ev))
```
'''


def main(start, end, title, *map_times):
    ev = Event(start, end, title, list(map_times)).load()
    text = TEMPLATE.format(title=title, date=ev.days[0].date(), start=start, end=end,
                           description=page.description(ev).replace('"', "'"),
                           categories=', '.join(ev.tags()), freeze='auto' if ev.final else 'false',
                           map_times=list(map_times))
    out = Path(__file__).resolve().parent.parent / 'site' / 'events' / f'{ev.slug}.qmd'
    out.write_text(text, encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
