"""Interactive stacked bar charts (Plotly): one panel per series on a shared time axis. Hovering anywhere shows
the date and every panel's value, with a spike line across all panels."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .style import BLUE, GRID, INK, MUTED, ORANGE, SURF

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def stacked_bars(panels, x_hover='%b %Y', panel_height=135, bar_width_days=25, title=None):
    """panels: list of dicts with keys
         series  pd.Series indexed by Timestamp (or int years)
         label   axis/hover label
         unit    unit text for hover ('' for none)
         fmt     number format for hover (default '+.2f')
         refs    optional list of reference levels drawn as ± dashed lines
         colors  'sign' (orange >= 0, blue < 0; default), 'median' (vs. the series median), or a fixed colour
         custom  optional pd.Series of strings shown in the hover after the label (e.g. ONI season)
    """
    n = len(panels)
    fig = go.Figure()
    gap = 0.035
    h = (1 - gap * (n - 1)) / n
    years = all(np.issubdtype(np.asarray(p['series'].index).dtype, np.integer) for p in panels)
    for i, p in enumerate(panels):
        s = p['series'].dropna()
        x = s.index if years else pd.DatetimeIndex(s.index)
        mode = p.get('colors', 'sign')
        if mode == 'sign':
            col = [ORANGE if v >= 0 else BLUE for v in s.values]
        elif mode == 'median':
            col = [ORANGE if v >= s.median() else BLUE for v in s.values]
        else:
            col = mode
        ya = 'y' if i == 0 else f'y{i + 1}'
        fmt = p.get('fmt', '+.2f')
        unit = f" {p['unit']}" if p.get('unit') else ''
        label = p['label'].replace('\n', ' ')
        cd = p.get('custom')
        if cd is not None:
            cd = cd.reindex(s.index).fillna('').values
            tmpl = f'{label} (%{{customdata}}): %{{y:{fmt}}}{unit}<extra></extra>'
        else:
            tmpl = f'{label}: %{{y:{fmt}}}{unit}<extra></extra>'
        fig.add_trace(go.Bar(x=x, y=s.values, marker_color=col, marker_line_width=0, xaxis='x', yaxis=ya,
                             customdata=cd, hovertemplate=tmpl, name=label,
                             width=None if years else bar_width_days * 86400000))
        top = 1 - i * (h + gap)
        fig.layout[f'yaxis{"" if i == 0 else i + 1}'] = dict(
            domain=[top - h, top], title=dict(text=p['label'].replace('\n', '<br>'), font=dict(size=11, color=INK)),
            zeroline=True, zerolinecolor=MUTED, zerolinewidth=1, gridcolor=GRID, tickfont=dict(size=10, color=MUTED),
            fixedrange=True)
        for r in p.get('refs', []):
            for sgn in (1, -1):
                fig.add_shape(type='line', xref='x domain', x0=0, x1=1, yref=ya, y0=sgn * r, y1=sgn * r,
                              line=dict(color=GRID, width=1, dash='dash'), layer='below')
        when = cd[-1] if cd is not None else (str(x[-1]) if years else f'{pd.Timestamp(x[-1]):%b %Y}')
        fig.add_annotation(xref='x domain', x=1, yref=f'{ya} domain', y=1.0, xanchor='right', yanchor='bottom',
                           text=f'{when}: {s.values[-1]:{fmt}}{unit}', showarrow=False, font=dict(size=10, color=INK))
    fig.update_layout(
        hovermode='x unified', hoversubplots='axis', grid=dict(rows=n, columns=1),
        height=panel_height * n + 70, margin=dict(l=90, r=20, t=40 if title else 15, b=40),
        paper_bgcolor=SURF, plot_bgcolor=SURF, showlegend=False, bargap=0.15,
        font=dict(family=FONT, color=INK), title=dict(text=title, x=0.01, font=dict(size=14)) if title else None,
        hoverlabel=dict(bgcolor='white', font=dict(size=12, family=FONT), bordercolor=GRID),
        xaxis=dict(anchor=f'y{n}' if n > 1 else 'y', showspikes=True, spikemode='across', spikesnap='cursor',
                   spikecolor=MUTED, spikethickness=1, spikedash='solid', gridcolor=GRID,
                   tickfont=dict(size=10, color=MUTED), hoverformat=None if years else x_hover),
    )
    return fig


def show(fig):
    """Display in Quarto/Jupyter with plotly.js from the CDN, no mode bar."""
    from IPython.display import HTML, display
    display(HTML(fig.to_html(full_html=False, include_plotlyjs='cdn',
                             config={'displayModeBar': False, 'responsive': True})))
