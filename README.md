# backtrader_plotly

Plot `backtrader`'s result using `plotly` instead of the default `matplotlib`

This is an experimental package, and it is done by replacing original `matplotlib` method calls.

## Installation
`$ pip install -i https://test.pypi.org/simple/ backtrader-plotly==1.0.2`

## Usage
```python
# import the package after installation
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme

# do whatever you want with `backtrader`
import backtrader as bt

# for instance
cerebro = bt.Cerebro()

# after adding data and strategy
cerebro.run()

# plot and save figures as `plotly` graph object
figs = cerebro.plot(BacktraderPlotly(show=True, scheme=PlotScheme()))
figs = [x for fig in figs for x in fig]  # flatten output

for fig in figs:
    plotly.io.to_html(fig, full_html=False)  # open html in the browser
    plotly.io.write_html(fig, file='plot.html')  # save the html file
```
