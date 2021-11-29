from my_scheme import PlotScheme
from plotly.subplots import make_subplots
import backtrader as bt
import bisect
import collections
import datetime
import math
import numpy as np
import plotly.graph_objects as go


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


class MyPlottly(metaclass=bt.MetaParams):
    params = (
        ('scheme', PlotScheme()),
        ('filename', None),
        ('plotconfig', None),
        ('output_mode', 'show'),
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
            self.fig = fig = make_subplots(rows=self.pinf.nrows, cols=1)
            figs.append(fig)

            self.pinf.pstart, self.pinf.pend, self.pinf.psize = pranges[numfig]
            self.pinf.xstart = self.pinf.pstart
            self.pinf.xend = self.pinf.pend

            self.pinf.clock = strategy
            self.pinf.xreal = self.pinf.clock.datetime.plot(
                self.pinf.pstart, self.pinf.psize)
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

    def plotind(self, iref, ind, subinds=None, upinds=None, downinds=None, masterax=None):
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
                    label += ' %.2f' % lplot[-1]

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

            xdata, lplotarray = self.pinf.xdata, lplot
            if lineplotinfo._get('_skipnan', False):
                # Get the full array and a mask to skipnan
                lplotarray = np.array(lplot)
                lplotmask = np.isfinite(lplotarray)

                # Get both the axis and the data masked
                lplotarray = lplotarray[lplotmask]
                xdata = np.array(xdata)[lplotmask]

            self.pltmethod(ax, xdata, lplotarray, **plotkwargs)

        # plot subindicators that were created on self
        for subind in subinds:
            self.plotind(iref, subind, subinds=self.dplotsover[subind], masterax=ax)

    def pltmethod(self, ax, xdata, lplotarray, **plotkwargs):
        print(ax, plotkwargs)

    def plotvolume(self, data, opens, highs, lows, closes, volumes, label):
        pass

    def plotdata(self, data, indicators):
        pass

    def show(self):
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
