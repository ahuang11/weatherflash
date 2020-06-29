from datetime import datetime

import numpy as np
import panel as pn
import pandas as pd
import holoviews as hv
pn.extension()
hv.extension('bokeh')
hv.renderer('bokeh').theme = 'caliber'

URL_FMT = (
    'https://mesonet.agron.iastate.edu/'
    'cgi-bin/request/daily.py?'
    'network={network}&stations={station}&'
    'year1=1928&month1=1&day1=1&'
    'year2=2020&month2=12&day2=1'
)
DF_COLS_ALL = [
    'min_temp_f', 'max_temp_f', 'precip_in', 'snow_in',
    'min_feel', 'max_feel', 'max_wind_speed_kts', 'max_wind_gust_kts',
    'climo_high_f', 'climo_low_f', 'climo_precip_in', 'day'
]
DF_COLS_POSITIVE = ['Precip In', 'Snow In']
DF_COLS_RENAMES = {
    'Climo High F': 'Climo Max Temp F',
    'Climo Low F': 'Climo Min Temp F',
    'Min Feel': 'Min Feel F',
    'Max Feel': 'Max Feel F',
    'Max Wind Speed Kts': 'Max Wind Kts',
    'Max Wind Gust Kts': 'Max Gust Kts'
}
DF_COLS_TMP = ['Min Temp F', 'Max Temp F', 'Min Feel F', 'Max Feel F']
DF_COLS_PCP = ['Precip In']
DF_COLS_WND = ['Max Wind Kts', 'Max Gust Kts']

ASOS_META_PKL = 'asos_meta.pkl'

WHITE = '#e5e5e5'
GRAY = '#5B5B5B'
RED = '#d44642'
BLUE = '#87b6bc'
YELLOW = '#F6CA06'

CSS = """
    body {
        font-family: Calibri Light;
        color: #5B5B5B;
        margin-top: 1.5%;
        margin-bottom: 2%;
        margin-left: 10%;
        margin-right: 14%;
    }

    /* Tooltip container */
    .tooltip {
      position: relative;
      display: inline-block;
    }

    /* Tooltip text */
    .tooltip .tooltiptext {
      visibility: hidden;
      background-color: #555;
      color: #fff;
      text-align: center;
      padding: 3px 5px;
      border-radius: 6px;
      min-width: 125px;

      /* Position the tooltip text */
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 0%;

      /* Fade in tooltip */
      opacity: 0;
      transition: opacity 0.2s;
    }

    /* Show the tooltip text when you mouse over the tooltip container */
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
"""
pn.config.raw_css.append(CSS)

def set_toolbar_autohide(plot, element):
    bokeh_plot = plot.state
    bokeh_plot.toolbar.autohide = True

tools_kwds = dict(tools=['hover'], default_tools=[])
hv.opts.defaults(
    hv.opts.VLine(color='gray', line_dash='dashed',
                  line_width=1, **tools_kwds),
    hv.opts.Histogram(
        responsive=True, show_grid=True, axiswise=True,
        fill_color='whitesmoke', line_color=WHITE,
        fontsize={'title': 13, 'labels': 12, 'ticks': 11}, **tools_kwds
    ),
    hv.opts.Layout(fontsize={'title': 15}),
    hv.opts.Text(text_color=GRAY, text_alpha=0.85, text_font='calibri',
                 **tools_kwds),
    backend='bokeh'
)


class WeatherFlash():
    def __init__(self):
        self.df_meta = pd.read_pickle(ASOS_META_PKL)
        self.stations = list(self.df_meta['stid'])
        self.stations += [station.lower() for station in self.stations]

    def read_data(self, station):
        station = station.upper()
        _, self.name, _, _, _, self.ts, network = self.df_meta.loc[
            self.df_meta['stid'] == station].values[0]

        df = pd.read_csv(
            URL_FMT.format(station=station, network=network),
            index_col='day', usecols=DF_COLS_ALL, parse_dates=True,
            na_values='None'
        )[DF_COLS_ALL[:-1]]
        df['day_of_year'] = df.index.dayofyear
        df.columns = df.columns.str.replace('_', ' ').str.title()
        df = df.rename(columns=DF_COLS_RENAMES)
        for col in DF_COLS_POSITIVE:
            df.loc[df[col] < 0, col] = np.nan
        self.df = df.dropna(subset=[df.columns[0]])

    @staticmethod
    def order_of_mag(x):
        if x == 0:
            return 0
        else:
            return np.floor(np.log10(np.abs(x)))

    def roundn(self, x, base=None, method='up'):
        if method == 'up':
            return np.ceil(x / base) * base
        elif method == 'down':
            return np.floor(x / base) * base
        else:
            return np.round(x / base) * base

    def create_hist(self, df_sel, var):
        # keep histogram pairs consistent with the same xlim + ylim
        # since the pairs are likely to be min + max or somehow related
        # for more intuitive comparison between the pairs
        col_ind = list(df_sel.columns).index(var)
        if col_ind % 2 == 0:
            var_ref = df_sel.columns[col_ind + 1]
        else:
            var_ref = df_sel.columns[col_ind - 1]
        var_min = df_sel[[var, var_ref]].min().min()
        var_max = df_sel[[var, var_ref]].max().max()

        oom = self.order_of_mag(var_max) - 1
        scale = 10 ** oom
        if oom > 0:
            scale = np.log10(scale)
        base = scale * 5

        var_min = self.roundn(var_min, base=base, method='down')
        var_max = self.roundn(var_max, base=base)

        if var_max < 1:
            var_max = 1

        num_bins = (var_max - var_min) / base
        if num_bins <= 7:
            var_bins = np.arange(var_min, var_max, base / 3).tolist()
        elif num_bins <= 14:
            var_bins = np.arange(var_min, var_max, base / 2).tolist()
        else:
            var_bins = np.arange(var_min, var_max, base).tolist()

        if var_max == var_min:
            var_max += 0.01
        xlim = var_min - base / 3, var_max + base / 3

        var_freq, var_edge = np.histogram(
            df_sel[var].values, bins=var_bins)
        var_ref_freq, _ = np.histogram(
            df_sel[var_ref].values, bins=var_bins)
        ymax = max(var_freq.max(), var_ref_freq.max())
        ylim = (0, ymax + ymax / 5)

        var_split = var.split()
        var_field = ' '.join(var_split[:-1])
        var_fmt = '.2f' if var not in ['Precip In', 'Snow In'] else '.2f'
        var_units = var_split[-1]

        var_hist = hv.Histogram((var_edge, var_freq)).opts(
            xlim=xlim, ylim=ylim, xlabel='', ylabel='Number of Days',
        ).redim.label(x=var, Frequency=f'{var_field} Count')

        plot = var_hist
        # highlight selected date
        var_sel = df_sel.loc[self.datetime.strftime('%Y-%m-%d'), var]
        if np.isnan(var_sel):
            label = var
        else:
            var_ind = np.where(var_edge <= var_sel)[0][-1]
            if var_ind == len(var_edge) - 1:
                var_ind -= 1
            var_slice = slice(*var_edge[var_ind:var_ind + 2])
            var_hist_hlgt = var_hist[var_slice, :].opts(fill_color=RED)
            label = f'{var_field}: {var_sel:{var_fmt}} {var_units}'
            plot *= var_hist_hlgt

        if var_freq.max() == 0:
            plot *= hv.Text(xlim[1] / 2, ylim[-1] / 2, 'Data N/A', fontsize=18)

        try:
            var_climo = df_sel.iloc[0][f'Climo {var}']
            var_vline = hv.VLine(var_climo)
            plot *= var_vline
        except KeyError:
            pass

        return plot.opts(title=label)

    def parse_field_units(self, var):
        split = var.split()
        units = split.pop(-1)
        units = units.upper() if len(units) == 1 else f' {units.lower()}'
        field = ' '.join(split).lower()
        return field, units

    def generate_tooltip(self, row, stat):
        var = getattr(row, f'idx{stat}')()
        val = row[var]
        field, units = self.parse_field_units(var)
        tooltip = f'The {field} was {val:.2f}{units}'
        return tooltip

    def create_hover_text(self, color, label, tooltip):
        if not tooltip:
            return
        hover_text = pn.pane.HTML(
            f'''
            <div class="tooltip" style="border:0.5px; border-style:solid;
            border-radius: 5px; padding: 4px 10px;
            border-color:{color}; color:{color}">{label}
            <span class="tooltiptext">{tooltip}</span></div>
            ''', margin=(0, 5, 5, 0), align='center'
        )
        self.highlights.append(hover_text)

    def create_highs_highlights(self, row_sel):
        color = label = tooltip = None
        row_tmp = row_sel[DF_COLS_TMP]
        if row_tmp.max() > 95:
            color = '#984b44'
            label = 'Scorching!!'
            tooltip = self.generate_tooltip(row_tmp, 'max')
        elif row_tmp.max() > 85:
            color = '#b05b5a'
            label = 'Hot!'
            tooltip = self.generate_tooltip(row_tmp, 'max')
        elif row_tmp.max() > 75:
            color = '#c77560'
            label = 'Warm'
            tooltip = self.generate_tooltip(row_tmp, 'max')
        elif row_tmp.max() > 60:
            color = '#e9cc77'
            label = 'Comfortable'
            tooltip = self.generate_tooltip(row_tmp, 'max')
        self.create_hover_text(color, label, tooltip)

    def create_lows_highlights(self, row_sel):
        color = label = tooltip = None
        row_tmp = row_sel[DF_COLS_TMP]
        if row_tmp.min() <= 32:
            color = '#5ca0b4'
            label = 'Freezing!!'
            tooltip = self.generate_tooltip(row_tmp, 'min')
        elif row_tmp.min() < 45:
            color = '#83c2d5'
            label = 'Cold'
            tooltip = self.generate_tooltip(row_tmp, 'min')
        elif row_tmp.min() < 60:
            color = '#8dbf71'
            label = 'Cool'
            tooltip = self.generate_tooltip(row_tmp, 'min')
        self.create_hover_text(color, label, tooltip)

    def create_pcp_highlights(self, row_sel):
        color = label = tooltip = None
        row_pcp = row_sel[DF_COLS_PCP]
        if row_pcp.max() > 1.5:
            color = '#618402'
            label = 'Significant precip!!'
            tooltip = self.generate_tooltip(row_pcp, 'max')
        elif row_pcp.max() > 0.5:
            color = '#a2b001'
            label = 'Precip'
            tooltip = self.generate_tooltip(row_pcp, 'max')
        elif row_pcp.max() >  0:
            color = '#a2b001'
            label = 'Light precip'
            tooltip = self.generate_tooltip(row_pcp, 'max')
        elif row_pcp.max() == 0:
            color = '#744e03'
            label = 'Dry'
            tooltip = self.generate_tooltip(row_pcp, 'max')
        self.create_hover_text(color, label, tooltip)

    def create_wnd_highlights(self, row_sel):
        color = label = tooltip = None
        row_wnd = row_sel[DF_COLS_WND]
        if row_wnd.max() > 74:
            color = '#5e1d47'
            label = 'Destructive winds!!'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() > 56:
            color = '#663c60'
            label = 'Violent winds!'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() > 34:
            color = '#966289'
            label = 'Heavy winds!'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() > 20:
            color = '#e196d1'
            label = 'Windy'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() > 8:
            color = '#5d535'
            label = 'Breezy'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() > 0:
            color = '#cccccc'
            label = 'Light breeze'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        elif row_wnd.max() == 0:
            color = '#eeeeee'
            label = 'Calm'
            tooltip = self.generate_tooltip(row_wnd, 'max')
        self.create_hover_text(color, label, tooltip)

    def create_highlights(self, label, df_sel):
        if 'Past Years' in label:
            row_sel = df_sel.loc[self.datetime]
            self.create_highs_highlights(row_sel)
            self.create_lows_highlights(row_sel)
            self.create_pcp_highlights(row_sel)
            self.create_wnd_highlights(row_sel)

    def create_content(self):
        mday = str(self.datetime)[5:10]
        df_sels = [self.df.loc[self.df.index.strftime('%m-%d') == mday]]
        for days in [365, 90, 30, 14]:
            df_sels.append(self.df.loc[
                (self.df.index >= self.datetime - pd.Timedelta(days=days)) &
                (self.df.index <= self.datetime)
            ])

        labels = ['Past Years', 'Past 365 Days', 'Past 90 Days',
                  'Past 30 Days', 'Past 14 Days']
        tab_items = []
        self.highlights.objects = []
        for label, df_sel in zip(labels, df_sels):
            self.create_highlights(label, df_sel)

            if 'Year' not in label:
                time_label = df_sel.index.min()
                weather_label = (
                    f'Histograms from {time_label:%B %d, %Y} to '
                    f'{self.datetime:%B %d, %Y}')
            else:
                time_label = self.ts[:4]
                weather_label = (
                    f'Histograms on {self.datetime:%B %d}s '
                    f'since {time_label}')
            plots = hv.Layout([
                self.create_hist(df_sel, var)
                for var in self.df.columns[:-1]
                if not var.startswith('Climo')
            ]).cols(4).relabel(
                f'{self.name.title()} ({self.station_input.value}) '
                f'{weather_label}'
            ).opts(toolbar=None, transpose=True)
            tab_items.append((
                label, pn.pane.HoloViews(
                    plots, linked_axes=False, min_width=750, min_height=1200)
                )
            )
        self.tabs[:] = tab_items

    def update_station_input(self, event):
        self.progress.active = True
        try:
            self.progress.bar_color = 'warning'
            self.read_data(event.new)
            self.create_content()
            self.progress.bar_color = 'secondary'
        except Exception as e:
            self.progress.bar_color = 'danger'
        self.progress.active = False

    def update_date_input(self, event):
        self.progress.active = True
        try:
            self.progress.bar_color = 'warning'
            self.datetime = pd.to_datetime(event.new)
            self.create_content()
            self.progress.bar_color = 'secondary'
        except Exception as e:
            self.progress.bar_color = 'danger'
        self.progress.active = False

    def view(self):
        self.station_input = pn.widgets.AutocompleteInput(
            name='ASOS Station ID', options=self.stations, align='center',
            value='CMI', width=300)
        self.station_input.param.watch(self.update_station_input, 'value')

        self.read_data(self.station_input.value)
        self.datetime = self.df.index.max()

        self.date_input = pn.widgets.DatePicker(
            name='Date Selector', align='center',
            value=self.datetime.date(), width=300)
        self.date_input.param.watch(self.update_date_input, 'value')

        self.progress = pn.widgets.Progress(
            active=False, bar_color='secondary', width=300,
            margin=(-15, 10, 25, 10), align='center')

        title = pn.pane.Markdown(
            f'# <center>Weather<span style='
            f'"color:{RED}">Flash</span></center>',
            width=300, margin=(5, 10, -25, 10))

        subtitle = pn.pane.Markdown(
            f'<center>'
            f'*The selected date\'s field value is listed by the title and '
            f'highlighted in red if available. '
            f'The climatology is shown as a dashed line. '
            f'Note, the highlights below may not reflect '
            f'standard thresholds!<br><br>'
            f'The app is run on '
            f'<a href="https://www.python.org/" '
            f'target=_blank">Python</a>, '
            f'<a href="https://pandas.pydata.org/" '
            f'target=_blank">pandas</a>, '
            f'<a href="https://numpy.org/" '
            f'target=_blank">numpy</a>, '
            f'<a href="https://panel.holoviz.org/" '
            f'target=_blank">panel</a>, '
            f'<a href="http://holoviews.org/" '
            f'target=_blank">holoviews</a>, and '
            f'<a href="https://dashboard.heroku.com/" '
            f'target=_blank">Heroku</a>. '
            f'The app\'s visuals were inspired by '
            f'<a href="https://www.leagueofgraphs.com/" '
            f'target=_blank">League of Graphs</a> and '
            f'<a href="https://weatherspark.com/" '
            f'target=_blank">Weather Spark</a>.<br><br>'
            f'Comments and suggestions appreciated '
            f'<a href="https://github.com/ahuang11/weatherflash/issues" '
            f'target=_blank">here</a>!*<br>'
            f'<a href="https://github.com/ahuang11/weatherflash" '
            f'target="_blank">Source code</a> | '
            f'<a href="https://github.com/ahuang11/" '
            f'target="_blank">My GitHub</a> | '
            f'<a href="https://mesonet.agron.iastate.edu/request/daily.phtml"'
            f' target="_blank">ASOS data</a></center>',
            width=300, margin=(-10, 10), align='center')

        self.highlights = pn.GridBox(
            sizing_mode='stretch_width',
            margin=(5, 10, 0, 10),
            ncols=4, align='center')

        left_col = pn.Column(
            title, self.progress,
            self.station_input, self.date_input, pn.layout.Divider(),
            subtitle, pn.layout.Divider(), self.highlights,
            sizing_mode='stretch_height')

        self.tabs = pn.Tabs(
            sizing_mode='stretch_both', margin=(10, 35),
            tabs_location='right', dynamic=True
        )
        self.create_content()

        layout = pn.Row(
            left_col, self.tabs,
            sizing_mode='stretch_both'
        )
        return layout

weatherflash = WeatherFlash()
weatherflash.view().servable(title='WeatherFlash')
