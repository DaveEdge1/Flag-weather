"""Animated atmospheric setting: one ERA5 map per 6 h with a network-rain strip, and an HTML frame player.

Frames are written as WebP under site/events/frames/<slug>/ and committed, so frozen pages build
without ERA5. The player (plain HTML/JS) supports play/pause, step, first/last, a slider, speed and
the arrow keys.
"""
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures.events import draw_map
from .figures.style import BLUE, INK, MUTED, ORANGE, SURF, clean

FRAMES_DIR = Path(__file__).resolve().parent.parent / 'site' / 'events' / 'frames'
PAD = pd.Timedelta('12h')   # frames start/end this long before/after the event window


def frame_times(ev):
    t = pd.DatetimeIndex(ev.era5.valid_time.values)
    return t[(t >= (ev.start - PAD).floor('6h')) & (t <= (ev.end + PAD).ceil('6h'))]


def _frame(ev, lsm, t, rain):
    fig = plt.figure(figsize=(11, 9.6), facecolor=SURF, layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[7.6, 1.35, 0.16])
    ax = fig.add_subplot(gs[0])
    pc = draw_map(ax, ev.era5, lsm, t, ev.best_track)
    sx = fig.add_subplot(gs[1])
    sx.bar(rain.index, rain.values, width=1 / 24, align='edge', color=BLUE)
    sx.axvline(t, color=ORANGE, lw=2)
    sx.axvspan(ev.start, ev.end, color='#eeeeec', zorder=0)
    sx.set_xlim(rain.index[0], rain.index[-1])
    sx.set_ylabel('Network median\nrain (in/h)', fontsize=8.5)
    loc = plt.matplotlib.dates.AutoDateLocator(maxticks=8)
    sx.xaxis.set_major_locator(loc)
    sx.xaxis.set_major_formatter(plt.matplotlib.dates.ConciseDateFormatter(loc))
    clean(sx)
    cax = fig.add_subplot(gs[2])
    cb = fig.colorbar(pc, cax=cax, orientation='horizontal')
    cb.set_label('IVT (kg m⁻¹ s⁻¹)', color=INK, fontsize=9)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    return fig


def write_frames(ev, lsm, dpi=90):
    """Render frames for an event; returns list of (relative path, UTC time)."""
    out = FRAMES_DIR / ev.slug
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.webp'):
        old.unlink()
    times = frame_times(ev)
    rain = ev.hourly.loc[times[0]:times[-1] + pd.Timedelta('6h')].median(axis=1)
    frames = []
    for i, t in enumerate(times):
        fig = _frame(ev, lsm, t, rain)
        f = out / f'{i:03d}.webp'
        fig.savefig(f, dpi=dpi, pil_kwargs={'quality': 82})
        plt.close(fig)
        frames.append((f'frames/{ev.slug}/{f.name}', t))
    return frames


def player(frames, start_time, uid):
    """Raw-HTML frame player (wrapped for Quarto output: asis)."""
    srcs = [p for p, _ in frames]
    labels = [f'{t:%b %d %H} UTC · {(t - pd.Timedelta("7h")):%b %d %H:%M} MST' for _, t in frames]
    start = int(np.argmin([abs((t - start_time).total_seconds()) for _, t in frames]))
    data = html.escape(json.dumps({'src': srcs, 'label': labels, 'start': start}))
    return f'''```{{=html}}
<div class="fw-player" id="{uid}" data-frames="{data}" tabindex="0">
  <img class="fw-img" alt="ERA5 atmospheric setting, animated" src="{srcs[start]}">
  <div class="fw-controls">
    <button type="button" data-act="first" title="First frame">⏮</button>
    <button type="button" data-act="prev" title="Previous 6 h (←)">◀ 6 h</button>
    <button type="button" data-act="play" class="fw-play" title="Play / pause (space)">Play</button>
    <button type="button" data-act="next" title="Next 6 h (→)">6 h ▶</button>
    <button type="button" data-act="last" title="Last frame">⏭</button>
    <input type="range" class="fw-slider" min="0" max="{len(srcs) - 1}" value="{start}" aria-label="Frame">
    <select class="fw-speed" aria-label="Speed">
      <option value="1000">1 fps</option><option value="500" selected>2 fps</option><option value="250">4 fps</option>
    </select>
    <span class="fw-label">{labels[start]}</span>
  </div>
</div>
```
'''
