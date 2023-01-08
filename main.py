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


class SMACrossStrategy(bt.Strategy):
    params = (
        ('pfast', 10),
        ('pslow', 30),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(period=self.p.pfast)
        self.slow_sma = bt.indicators.SMA(period=self.p.pslow)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()

        elif self.crossover < 0:
            self.close()


def main():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(IchimokuStrategy)
    cerebro.addstrategy(SMACrossStrategy)

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
    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))

    # directly manipulate object using methods provided by `plotly`
    for i, each_run in enumerate(figs):
        for j, each_strategy_fig in enumerate(each_run):
            # open plot in browser
            each_strategy_fig.show()

            # save the html of the plot to a variable
            html = plotly.io.to_html(each_strategy_fig, full_html=False)

            # write html to disk
            plotly.io.write_html(each_strategy_fig, f'{i}_{j}.html', full_html=True)


if __name__ == '__main__':
    main()
