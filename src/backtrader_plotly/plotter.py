from .scheme import PlotScheme
from plotly.subplots import make_subplots
import backtrader as bt
import bisect
import collections
import datetime
import math
import numpy as np
import operator
import plotly.graph_objects as go

# Mapper from matplotlib to plotly
line_style_mapper = {
    '': 'solid',
    '--': 'dash',
    ':': 'dot',
    '-.': 'dashdot',
}

marker_style_mapper = {
    'o': 'circle',
    '^': 'triangle-up',
    'v': 'triangle-down',
}


class PInfo(object):
    def __init__(self, sch):
        self.sch = sch
        self.nrows = 0
        self.row = 0
        self.clock = None
        self.x = None
        self.xlen = 0
        self.sharex = None
        self.figs = list()
        self.cursors = list()
        self.daxis = collections.OrderedDict()
        self.vaxis = list()
        self.zorder = dict()
        self.coloridx = collections.defaultdict(lambda: -1)
        self.handles = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.legpos = collections.defaultdict(int)

        # self.prop = mfontmgr.FontProperties(size=self.sch.subtxtsize)

    def newfig(self, figid, numfig, mpyplot):
        fig = mpyplot.figure(figid + numfig)
        self.figs.append(fig)
        self.daxis = collections.OrderedDict()
        self.vaxis = list()
        self.row = 0
        self.sharex = None
        return fig

    def nextcolor(self, ax):
        self.coloridx[ax] += 1
        return self.coloridx[ax]

    def color(self, ax):
        return self.sch.color(self.coloridx[ax])

    def zordernext(self, ax):
        z = self.zorder[ax]
        if self.sch.zdown:
            return z * 0.9999
        return z * 1.0001

    def zordercur(self, ax):
        return self.zorder[ax]


class BacktraderPlotly(metaclass=bt.MetaParams):
    params = (
        ('scheme', PlotScheme()),
        ('show', True),
    )

    def __init__(self, **kwargs):
        for pname, pvalue in kwargs.items():
            setattr(self.p.scheme, pname, pvalue)

    def plot(self, strategy, figid=0, numfigs=1, iplot=True, start=None, end=None, **kwargs):
        if not strategy.datas:
            return

        if not len(strategy):
            return

        self.pinf = PInfo(self.p.scheme)
        self.sortdataindicators(strategy)
        self.calcrows(strategy)

        st_dtime = strategy.lines.datetime.plot()
        if start is None:
            start = 0
        if end is None:
            end = len(st_dtime)

        if isinstance(start, datetime.date):
            start = bisect.bisect_left(st_dtime, bt.date2num(start))

        if isinstance(end, datetime.date):
            end = bisect.bisect_right(st_dtime, bt.date2num(end))

        if end < 0:
            end = len(st_dtime) + 1 + end  # -1 =  len() -2 = len() - 1

        slen = len(st_dtime[start:end])
        d, m = divmod(slen, numfigs)
        pranges = list()
        for i in range(numfigs):
            a = d * i + start
            if i == (numfigs - 1):
                d += m  # add remainder to last stint
            b = a + d

            pranges.append([a, b, d])

        figs = []
        for numfig in range(numfigs):
            self.fig = make_subplots(rows=self.pinf.nrows,
                                     cols=1,
                                     shared_xaxes=True,
                                     vertical_spacing=0.02,
                                     specs=[[{'secondary_y': True}] for _ in range(self.pinf.nrows)],
                                     )
            figs.append(self.fig)

            self.pinf.pstart, self.pinf.pend, self.pinf.psize = pranges[numfig]
            self.pinf.xstart = self.pinf.pstart
            self.pinf.xend = self.pinf.pend

            self.pinf.clock = strategy
            self.pinf.xreal = self.pinf.clock.datetime.plot(self.pinf.pstart, self.pinf.psize)
            self.pinf.xlen = len(self.pinf.xreal)
            self.pinf.x = list(range(self.pinf.xlen))
            # self.pinf.pfillers = {None: []}
            # for key, val in pfillers.items():
            #     pfstart = bisect.bisect_left(val, self.pinf.pstart)
            #     pfend = bisect.bisect_right(val, self.pinf.pend)
            #     self.pinf.pfillers[key] = val[pfstart:pfend]

            # Do the plotting
            # Things that go always at the top (observers)
            self.pinf.xdata = self.pinf.x
            for ptop in self.dplotstop:
                self.plotind(None, ptop, subinds=self.dplotsover[ptop])

            # Create the rest on a per data basis
            dt0, dt1 = self.pinf.xreal[0], self.pinf.xreal[-1]
            for data in strategy.datas:
                if not data.plotinfo.plot:
                    continue

                self.pinf.xdata = self.pinf.x
                xd = data.datetime.plotrange(self.pinf.xstart, self.pinf.xend)
                if len(xd) < self.pinf.xlen:
                    self.pinf.xdata = xdata = []
                    xreal = self.pinf.xreal
                    dts = data.datetime.plot()
                    xtemp = list()
                    for dt in (x for x in dts if dt0 <= x <= dt1):
                        dtidx = bisect.bisect_left(xreal, dt)
                        xdata.append(dtidx)
                        xtemp.append(dt)

                    self.pinf.xstart = bisect.bisect_left(dts, xtemp[0])
                    self.pinf.xend = bisect.bisect_right(dts, xtemp[-1])

                for ind in self.dplotsup[data]:
                    self.plotind(data, ind, subinds=self.dplotsover[ind], upinds=self.dplotsup[ind], downinds=self.dplotsdown[ind])

                self.plotdata(data, self.dplotsover[data])

                for ind in self.dplotsdown[data]:
                    self.plotind(data, ind, subinds=self.dplotsover[ind], upinds=self.dplotsup[ind], downinds=self.dplotsdown[ind])

            # Figure style
            self.fig.update_layout(height=3840, hovermode='x unified')
            for i in range(self.pinf.nrows):
                self.fig['layout'][f'xaxis{i + 1}']['showticklabels'] = True
                # self.fig['layout'][f'xaxis{i + 1}']['tickmode'] = 'array'
                # self.fig['layout'][f'xaxis{i + 1}']['ticktext'] = ['1' for x in self.pinf.xreal]

        return figs

    def newaxis(self, obj, rowspan):
        # update the row index with the taken rows
        self.pinf.row += rowspan

        ax = self.pinf.row  # ax is column no. (1-indexed)

        # update the sharex information if not available
        if self.pinf.sharex is None:
            self.pinf.sharex = ax

        # save the mapping indicator - axis and return
        self.pinf.daxis[obj] = ax

        # Activate grid in all axes if requested
        # ax.yaxis.tick_right()
        # ax.grid(self.pinf.sch.grid, which='both')

        return ax

    def plotind(self, iref, ind, subinds=None, upinds=None, downinds=None, masterax=None, secondary_y=False):
        sch = self.p.scheme

        # check subind
        subinds = subinds or []
        upinds = upinds or []
        downinds = downinds or []

        # plot subindicators on self with independent axis above
        for upind in upinds:
            self.plotind(iref, upind)

        ax = masterax or self.newaxis(ind, rowspan=self.pinf.sch.rowsminor)
        indlabel = ind.plotlabel()

        # Scan lines quickly to find out if some lines have to be skipped for
        # legend (because matplotlib reorders the legend)
        toskip = 0
        for lineidx in range(ind.size()):
            line = ind.lines[lineidx]
            linealias = ind.lines._getlinealias(lineidx)
            lineplotinfo = getattr(ind.plotlines, '_%d' % lineidx, None)
            if not lineplotinfo:
                lineplotinfo = getattr(ind.plotlines, linealias, None)
            if not lineplotinfo:
                lineplotinfo = bt.AutoInfoClass()
            pltmethod = lineplotinfo._get('_method', 'plot')
            if pltmethod != 'plot':
                toskip += 1 - lineplotinfo._get('_plotskip', False)

        if toskip >= ind.size():
            toskip = 0

        for lineidx in range(ind.size()):
            line = ind.lines[lineidx]
            linealias = ind.lines._getlinealias(lineidx)

            lineplotinfo = getattr(ind.plotlines, '_%d' % lineidx, None)
            if not lineplotinfo:
                lineplotinfo = getattr(ind.plotlines, linealias, None)

            if not lineplotinfo:
                lineplotinfo = bt.AutoInfoClass()

            if lineplotinfo._get('_plotskip', False):
                continue

            # Legend label only when plotting 1st line
            if masterax and not ind.plotinfo.plotlinelabels:
                label = indlabel * (not toskip) or '_nolegend'
            else:
                label = (indlabel + '\n') * (not toskip)
                label += lineplotinfo._get('_name', '') or linealias

            toskip -= 1  # one line less until legend can be added

            # plot data
            lplot = line.plotrange(self.pinf.xstart, self.pinf.xend)

            # Global and generic for indicator
            if self.pinf.sch.linevalues and ind.plotinfo.plotlinevalues:
                plotlinevalue = lineplotinfo._get('_plotvalue', True)
                if plotlinevalue and not math.isnan(lplot[-1]):
                    label += f' {lplot[-1]:.{self.pinf.sch.decimalprecision}f}'

            plotkwargs = dict()
            linekwargs = lineplotinfo._getkwargs(skip_=True)

            if linekwargs.get('color', None) is None:
                if not lineplotinfo._get('_samecolor', False):
                    self.pinf.nextcolor(ax)
                plotkwargs['color'] = self.pinf.color(ax)

            plotkwargs.update(dict(aa=True, label=label))
            plotkwargs.update(**linekwargs)

            if ax in self.pinf.zorder:
                plotkwargs['zorder'] = self.pinf.zordernext(ax)

            # pltmethod = getattr(ax, lineplotinfo._get('_method', 'plot'))

            xdata, lplotarray = self.pinf.xreal, lplot  # use timestamp for x axis
            if lineplotinfo._get('_skipnan', False):
                # Get the full array and a mask to skipnan
                lplotarray = np.array(lplot)
                lplotmask = np.isfinite(lplotarray)

                # Get both the axis and the data masked
                lplotarray = lplotarray[lplotmask]
                xdata = np.array(xdata)[lplotmask]

            # Convert timestamp to datetime
            xdata = [bt.num2date(x) for x in xdata]

            self.pltmethod(ax, xdata, lplotarray, secondary_y, **plotkwargs)

            # Code to place a label at the right hand side with the last value
            # vtags = lineplotinfo._get('plotvaluetags', True)
            # if self.pinf.sch.valuetags and vtags:
            #     linetag = lineplotinfo._get('_plotvaluetag', True)
            #     if linetag and not math.isnan(lplot[-1]):
            #         # line has valid values, plot a tag for the last value
            #         self.drawtag(ax, len(self.pinf.xreal), lplot[-1],
            #                      facecolor='white',
            #                      edgecolor=self.pinf.color(ax))

            farts = (('_gt', operator.gt), ('_lt', operator.lt), ('', None),)
            for fcmp, fop in farts:
                fattr = '_fill' + fcmp
                fref, fcol = lineplotinfo._get(fattr, (None, None))
                if fref is not None:
                    y1 = np.array(lplot)
                    if isinstance(fref, int):
                        y2 = np.full_like(y1, fref)
                    else:  # string, naming a line, nothing else is supported
                        l2 = getattr(ind, fref)
                        prl2 = l2.plotrange(self.pinf.xstart, self.pinf.xend)
                        y2 = np.array(prl2)
                    kwargs = dict()
                    if fop is not None:
                        kwargs['where'] = fop(y1, y2)

                    falpha = self.pinf.sch.fillalpha
                    if isinstance(fcol, (list, tuple)):
                        fcol, falpha = fcol

                    ax.fill_between(self.pinf.xdata, y1, y2,
                                    facecolor=fcol,
                                    alpha=falpha,
                                    interpolate=True,
                                    **kwargs)

        # plot subindicators that were created on self
        for subind in subinds:
            self.plotind(iref, subind, subinds=self.dplotsover[subind], masterax=ax)

        # plot subindicators on self with independent axis below
        for downind in downinds:
            self.plotind(iref, downind)

    def pltmethod(self, ax, xdata, lplotarray, secondary_y, **plotkwargs):
        # print(ax, plotkwargs)

        opacity = 1
        line = dict(color=plotkwargs['color'])  # line or marker style
        if 'marker' in plotkwargs:
            # Scatter plot
            marker = dict(symbol=marker_style_mapper[plotkwargs['marker']])
            self.fig.add_trace(go.Scatter(mode='markers',
                                          x=np.array(xdata),
                                          y=np.array(lplotarray),
                                          name=plotkwargs['label'],
                                          opacity=opacity,
                                          marker=marker,
                                          line=line,
                                          ), row=ax, col=1, secondary_y=secondary_y
                               )
        elif 'width' in plotkwargs:
            # Bar plot
            if 'alpha' in plotkwargs:
                opacity = plotkwargs['alpha']
            self.fig.add_trace(go.Bar(x=np.array(xdata),
                                      y=np.array(lplotarray),
                                      name=plotkwargs['label'],
                                      opacity=opacity,
                                      width=plotkwargs['width'],
                                      ), row=ax, col=1, secondary_y=secondary_y
                               )
        else:
            # Line plot
            if 'ls' in plotkwargs:
                line['dash'] = line_style_mapper[plotkwargs['ls']]

            self.fig.add_trace(go.Scatter(x=np.array(xdata),
                                          y=np.array(lplotarray),
                                          name=plotkwargs['label'],
                                          opacity=opacity,
                                          line=line,
                                          ), row=ax, col=1, secondary_y=secondary_y
                               )

    def plotvolume(self, data, opens, highs, lows, closes, volumes, label):
        pmaster = data.plotinfo.plotmaster
        if pmaster is data:
            pmaster = None
        voloverlay = (self.pinf.sch.voloverlay and pmaster is None)

        # if sefl.pinf.sch.voloverlay:
        if voloverlay:
            rowspan = self.pinf.sch.rowsmajor
        else:
            rowspan = self.pinf.sch.rowsminor

        ax = self.newaxis(data.volume, rowspan=rowspan)
        # print(ax, 'vol')

        # if self.pinf.sch.voloverlay:
        if voloverlay:
            volalpha = self.pinf.sch.voltrans
        else:
            volalpha = 1.0

        maxvol = volylim = max(volumes)
        if maxvol:
            # Plot the overlay volume
            vollabel = label

            # Get x axis data
            xdata = self.pinf.xreal
            xdata = [bt.num2date(x) for x in xdata]

            self.fig.add_trace(go.Bar(x=np.array(xdata),
                                      y=np.array(volumes),
                                      name=label,
                                      ), row=ax, col=1, secondary_y=False
                               )
            self.fig['layout'][f'yaxis{2 * ax - 1}']['range'] = [0, maxvol / self.pinf.sch.volscaling]

    def plotdata(self, data, indicators):
        for ind in indicators:
            upinds = self.dplotsup[ind]
            for upind in upinds:
                self.plotind(data, upind,
                             subinds=self.dplotsover[upind],
                             upinds=self.dplotsup[upind],
                             downinds=self.dplotsdown[upind])

        opens = data.open.plotrange(self.pinf.xstart, self.pinf.xend)
        highs = data.high.plotrange(self.pinf.xstart, self.pinf.xend)
        lows = data.low.plotrange(self.pinf.xstart, self.pinf.xend)
        closes = data.close.plotrange(self.pinf.xstart, self.pinf.xend)
        volumes = data.volume.plotrange(self.pinf.xstart, self.pinf.xend)

        vollabel = 'Volume'
        pmaster = data.plotinfo.plotmaster
        if pmaster is data:
            pmaster = None

        datalabel = ''
        if hasattr(data, '_name') and data._name:
            datalabel += data._name

        voloverlay = (self.pinf.sch.voloverlay and pmaster is None)

        if not voloverlay:
            vollabel += ' ({})'.format(datalabel)

        # if self.pinf.sch.volume and self.pinf.sch.voloverlay:
        axdatamaster = None
        if self.pinf.sch.volume and voloverlay:
            volplot = self.plotvolume(data, opens, highs, lows, closes, volumes, vollabel)
            axvol = self.pinf.daxis[data.volume]
            ax = axvol
            self.pinf.daxis[data] = ax
            self.pinf.vaxis.append(ax)
        else:
            if pmaster is None:
                ax = self.newaxis(data, rowspan=self.pinf.sch.rowsmajor)
            elif getattr(data.plotinfo, 'sameaxis', False):
                axdatamaster = self.pinf.daxis[pmaster]
                ax = axdatamaster
            else:
                axdatamaster = self.pinf.daxis[pmaster]
                ax = axdatamaster
                self.pinf.vaxis.append(ax)

        if hasattr(data, '_compression') and hasattr(data, '_timeframe'):
            tfname = bt.TimeFrame.getname(data._timeframe, data._compression)
            datalabel += ' (%d %s)' % (data._compression, tfname)

        plinevalues = getattr(data.plotinfo, 'plotlinevalues', True)

        # Get x axis data
        xdata = self.pinf.xreal
        xdata = [bt.num2date(x) for x in xdata]
        if self.pinf.sch.style.startswith('line'):
            if self.pinf.sch.linevalues and plinevalues:
                datalabel += f' C:{closes[-1]:.{self.pinf.sch.decimalprecision}f}'

            if axdatamaster is None:
                color = self.pinf.sch.loc
            else:
                self.pinf.nextcolor(axdatamaster)
                color = self.pinf.color(axdatamaster)

            self.fig.add_trace(go.Scatter(x=np.array(xdata),
                                          y=np.array(closes),
                                          name=datalabel,
                                          ), row=ax, col=1, secondary_y=True
                               )

        else:
            if self.pinf.sch.linevalues and plinevalues:
                datalabel += (f' O:{opens[-1]:.{self.pinf.sch.decimalprecision}f} '
                              f'H:{highs[-1]:.{self.pinf.sch.decimalprecision}f} '
                              f'L:{lows[-1]:.{self.pinf.sch.decimalprecision}f} '
                              f'C:{closes[-1]:.{self.pinf.sch.decimalprecision}f}'
                              )
            if self.pinf.sch.style.startswith('candle'):
                self.fig.add_trace(go.Candlestick(x=np.array(xdata),
                                                  open=np.array(opens),
                                                  high=np.array(highs),
                                                  low=np.array(lows),
                                                  close=np.array(closes),
                                                  increasing_line_color=self.pinf.sch.barup,
                                                  decreasing_line_color=self.pinf.sch.bardown,
                                                  name=datalabel,
                                                  ), row=ax, col=1, secondary_y=True
                                   )
                self.fig['layout'][f'xaxis{ax}']['rangeslider']['visible'] = False

        for ind in indicators:
            self.plotind(data, ind, subinds=self.dplotsover[ind], masterax=ax, secondary_y=True)

        a = axdatamaster or ax

        for ind in indicators:
            downinds = self.dplotsdown[ind]
            for downind in downinds:
                self.plotind(data, downind,
                             subinds=self.dplotsover[downind],
                             upinds=self.dplotsup[downind],
                             downinds=self.dplotsdown[downind],
                             secondary_y=True)

        self.pinf.legpos[a] = len(self.pinf.handles[a])

    def show(self):
        if self.p.show:
            self.fig.show()

    def sortdataindicators(self, strategy):
        # These lists/dictionaries hold the subplots that go above each data
        self.dplotstop = list()
        self.dplotsup = collections.defaultdict(list)
        self.dplotsdown = collections.defaultdict(list)
        self.dplotsover = collections.defaultdict(list)

        # Sort observers in the different lists/dictionaries
        for x in strategy.getobservers():
            if not x.plotinfo.plot or x.plotinfo.plotskip:
                continue

            if x.plotinfo.subplot:
                self.dplotstop.append(x)
            else:
                key = getattr(x._clock, 'owner', x._clock)
                self.dplotsover[key].append(x)

        # Sort indicators in the different lists/dictionaries
        for x in strategy.getindicators():
            if not hasattr(x, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            if not x.plotinfo.plot or x.plotinfo.plotskip:
                continue

            x._plotinit()  # will be plotted ... call its init function

            # support LineSeriesStub which has "owner" to point to the data
            key = getattr(x._clock, 'owner', x._clock)
            if key is strategy:  # a LinesCoupler
                key = strategy.data

            if getattr(x.plotinfo, 'plotforce', False):
                if key not in strategy.datas:
                    datas = strategy.datas
                    while True:
                        if key not in strategy.datas:
                            key = key._clock
                        else:
                            break

            xpmaster = x.plotinfo.plotmaster
            if xpmaster is x:
                xpmaster = None
            if xpmaster is not None:
                key = xpmaster

            if x.plotinfo.subplot and xpmaster is None:
                if x.plotinfo.plotabove:
                    self.dplotsup[key].append(x)
                else:
                    self.dplotsdown[key].append(x)
            else:
                self.dplotsover[key].append(x)

    def calcrows(self, strategy):
        # Calculate the total number of rows
        rowsmajor = self.pinf.sch.rowsmajor
        rowsminor = self.pinf.sch.rowsminor
        nrows = 0

        datasnoplot = 0
        for data in strategy.datas:
            if not data.plotinfo.plot:
                # neither data nor indicators nor volume add rows
                datasnoplot += 1
                self.dplotsup.pop(data, None)
                self.dplotsdown.pop(data, None)
                self.dplotsover.pop(data, None)

            else:
                pmaster = data.plotinfo.plotmaster
                if pmaster is data:
                    pmaster = None
                if pmaster is not None:
                    # data doesn't add a row, but volume may
                    if self.pinf.sch.volume:
                        nrows += 0  # volume never adds row in Plotly subplot
                else:
                    # data adds rows, volume may
                    nrows += rowsmajor
                    if self.pinf.sch.volume and not self.pinf.sch.voloverlay:
                        nrows += 0  # volume never adds row in Plotly subplot

        if False:
            # Datas and volumes
            nrows += (len(strategy.datas) - datasnoplot) * rowsmajor
            if self.pinf.sch.volume and not self.pinf.sch.voloverlay:
                nrows += (len(strategy.datas) - datasnoplot) * rowsminor

        # top indicators/observers
        nrows += len(self.dplotstop)

        # indicators above datas
        nrows += sum(len(v) for v in self.dplotsup.values())
        nrows += sum(len(v) for v in self.dplotsdown.values())

        self.pinf.nrows = nrows
