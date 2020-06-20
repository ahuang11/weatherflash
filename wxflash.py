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
    'Min Temp F', 'Max Temp F', 'Precip In', 'Snow In',
    'Min Dewpoint F', 'Max Dewpoint F', 'Min Humidity %', 'Max Humidity %',
    'Min Feel F', 'Max Feel F', 'Max Wind Kts', 'Max Gust Kts',
    'Climo Max Temp F', 'Climo Min Temp F', 'Climo Precip In', 'Day Of Year'
]
DF_COLS_POSITIVE = ['Precip In', 'Snow In', 'Min Humidity %', 'Max Humidity %']

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
        margin-left: 8%;
        margin-right: 8%;
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
        fontsize={'labels': 16, 'ticks': 13}, **tools_kwds
    ),
    hv.opts.Text(text_color=GRAY, text_alpha=0.85, text_font='calibri',
                 **tools_kwds),
    backend='bokeh'
)


class WxFlash():
    def __init__(self):
        self.df_meta = pd.read_pickle(ASOS_META_PKL)
        self.stations = list(self.df_meta['stid'])
        self.stations += [station.lower() for station in self.stations]

    def read_data(self, station):
        _, self.name, _, _, _, self.ts, network = self.df_meta.loc[
            self.df_meta['stid'] == station.upper()].values[0]

        df = pd.read_csv(URL_FMT.format(station=station, network=network),
                         index_col='day', parse_dates=True)
        df = df.drop(columns='station').rename_axis('time')
        df = df.apply(pd.to_numeric, errors='coerce').dropna(
            subset=['max_temp_f'])
        df['day_of_year'] = df.index.dayofyear

        df.columns = df.columns.str.replace('_', ' ').str.title()
        df = df.rename(columns={
            'Climo High F': 'Climo Max Temp F',
            'Climo Low F': 'Climo Min Temp F',
            'Min Feel': 'Min Feel F',
            'Max Feel': 'Max Feel F',
            'Min Rh': 'Min Humidity %',
            'Max Rh': 'Max Humidity %',
            'Max Wind Speed Kts': 'Max Wind Kts',
            'Max Wind Gust Kts': 'Max Gust Kts'
        })[DF_COLS_ALL]
        for col in DF_COLS_POSITIVE:
            df.loc[df[col] < 0, col] = np.nan
        self.df = df

    @staticmethod
    def remove_white_borders(plot, element):
        p = plot.state
        p.border_fill_color = 'whitesmoke'

    def create_hist(self, df_mday, var):
        # keep histogram pairs consistent with the same xlim + ylim
        # since the pairs are likely to be min + max or somehow related
        # for more intuitive comparison between the pairs
        col_ind = list(df_mday.columns).index(var)
        if col_ind % 2 == 0:
            var_ref = df_mday.columns[col_ind + 1]
        else:
            var_ref = df_mday.columns[col_ind - 1]
        var_min = df_mday[[var, var_ref]].min().min()
        var_max = df_mday[[var, var_ref]].max().max()

        var_bins = np.linspace(var_min, var_max, 20).tolist()
        var_diff = np.diff(var_bins).min()
        if var_max == var_min:
            var_max += 0.01
        xlim = var_min - var_diff, var_max + var_diff

        var_freq, var_edge = np.histogram(
            df_mday[var].values, bins=var_bins)
        var_ref_freq, _ = np.histogram(
            df_mday[var_ref].values, bins=var_bins)
        ymax = max(var_freq.max(), var_ref_freq.max())
        ylim = (0, ymax + ymax / 5)

        # manual implementation of the sharey functionality
        if col_ind % 4 == 0:
            ylabel = 'Number of Years'
        else:
            ylabel = ''

        var_split = var.split()
        var_field = ' '.join(var_split[:-1])
        var_fmt = '.2f' if var not in ['Precip In', 'Snow In'] else '.2f'
        var_units = var_split[-1]

        var_hist = hv.Histogram((var_edge, var_freq)).opts(
            xlim=xlim, ylim=ylim, xlabel='', ylabel=ylabel, title=''
        ).redim.label(x=var, Frequency=f'{var_field} Count')

        plot = var_hist
        # highlight selected date
        var_sel = df_mday.loc[self.datetime.strftime('%Y-%m-%d'), var]
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
        var_text = hv.Text(var_min, ylim[-1] - ymax / 20, label,
                           halign='left', valign='top', fontsize=14)
        plot *= var_text

        if var_freq.max() == 0:
            plot *= hv.Text(
                (var_min + var_max) / 2, ylim[-1] / 2, 'Data N/A', fontsize=18)

        try:
            var_climo = df_mday.iloc[0][f'Climo {var}']
            var_vline = hv.VLine(var_climo)
            plot *= var_vline
        except KeyError:
            pass

        return plot

    def create_plots(self):
        mday = str(self.datetime)[5:10]
        df_mday = self.df.loc[self.df.index.strftime('%m-%d') == mday]
        plots = hv.Layout([
            self.create_hist(df_mday, var)
            for var in self.df.columns[:-1]
            if not var.startswith('Climo')
        ]).cols(4).relabel(
            f'{self.name.title()} ({self.station_input.value}) '
            f'Weather on {self.datetime:%B %d}s since {self.ts[:4]}'
        ).opts(toolbar=None)
        return plots

    def update_station_input(self, event):
        self.progress.active = True
        try:
            self.progress.bar_color = 'warning'
            self.read_data(self.station_input.value)
            plots = self.create_plots()
            self.plots.object = plots
            self.progress.bar_color = 'secondary'
        except Exception:
            self.progress.bar_color = 'danger'
        self.progress.active = False

    def update_date_input(self, event):
        self.progress.active = True
        try:
            self.progress.bar_color = 'warning'
            self.datetime = event.new
            plots = self.create_plots()
            self.plots.object = plots
            self.progress.bar_color = 'secondary'
        except Exception:
            self.progress.bar_color = 'danger'
        self.progress.active = False

    def view(self):
        self.station_input = pn.widgets.AutocompleteInput(
            name='ASOS Station ID', options=self.stations, align='center',
            value='CMI', width=275)
        self.station_input.param.watch(self.update_station_input, 'value')

        self.read_data(self.station_input.value)
        self.datetime = self.df.index.max()

        self.date_input = pn.widgets.DatePicker(
            name='Date Selector', align='center',
            value=self.datetime.date(), width=275)
        self.date_input.param.watch(self.update_date_input, 'value')

        plots = self.create_plots()

        self.plots = pn.pane.HoloViews(
            plots, sizing_mode='stretch_both')

        self.progress = pn.widgets.Progress(
            active=False, bar_color='secondary', width=275,
            margin=(-15, 10, 25, 10), align='center')

        title = pn.pane.Markdown(
            f'# <center>Weather<span style='
            f'"color:{RED}">Flash</span></center>',
            width=275, margin=(5, 10, -25, 10))

        subtitle = pn.pane.Markdown(
            f'<center>**Selected date\'s value shown and '
            f'highlighted in red; climatology shown as dashed line.</center>',
            width=275, margin=(-5, 10))

        layout = pn.Row(
            pn.Column(title, self.progress,
                      self.station_input, self.date_input,
                      subtitle, sizing_mode='stretch_height'),
            pn.Column(self.plots, sizing_mode='stretch_both', margin=(10, 35)),
            sizing_mode='stretch_both'
        )
        return layout

wxflash = WxFlash()
wxflash.view().servable()
