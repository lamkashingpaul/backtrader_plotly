from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import backtrader as bt
import datetime
import os
import plotly.io
import sys


class IchimokuStrategy(bt.Strategy):
    def __init__(self):
        self.ichimoku = bt.indicators.Ichimoku()
        self.ha_delta = bt.indicators.haDelta()
        self.macd_histo = bt.indicators.MACDHisto()


def main():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(IchimokuStrategy)

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, './datas/orcl-1995-2014.txt')

    data = bt.feeds.YahooFinanceCSVData(dataname=datapath,
                                        fromdate=datetime.datetime(2000, 1, 1),
                                        todate=datetime.datetime(2000, 12, 31),
                                        reverse=False)

    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcommission(commission=0.0)

    cerebro.run()

    scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)
    figs = cerebro.plot(BacktraderPlotly(show=True, scheme=scheme))
    figs = [x for fig in figs for x in fig]  # flatten output

    for fig in figs:
        plotly.io.to_html(fig, full_html=False)  # open html in the browser


if __name__ == '__main__':
    main()
