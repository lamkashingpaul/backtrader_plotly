# backtrader_plotly

Plot `backtrader`'s result using `plotly` instead of the default `matplotlib`

This is an experimental package, and it is done by replacing original `matplotlib` method calls.

## Installation

`$ pip install backtrader-plotly==1.4.0`

## Features

- Support for Filled Area Plotting and Toggling (Added from 1.4.0)

- New Scheme Arguments (Added from 1.3.0)

  Additional scheme arguments are added to provide extra control

  | Name of Argument      | Default Value | Description                                                                                                                                    |
  | :-------------------- | ------------: | :--------------------------------------------------------------------------------------------------------------------------------------------- |
  | decimal_places        |             5 | It is used to control the number of decimal places of price shown on the plot. For instance, forex price usually consists of 5 decimal places. |
  | max_legend_text_width |            16 | It is used to limit the legend text width to prevent it from occupying the page.                                                               |

## Usage

[Complete Working Example Here](main.py)

```python
# import the package after installation
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme
import plotly.io

# do whatever you want with `backtrader`
import backtrader as bt

# for instance
cerebro = bt.Cerebro()

# after adding data and strategy
cerebro.run()

# define plot scheme with new additional scheme arguments
scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)

# plot and save figures as `plotly` graph object
figs = cerebro.plot(BacktraderPlotly(show=True, scheme=scheme))
figs = [x for fig in figs for x in fig]  # flatten output
for fig in figs:
    plotly.io.to_html(fig, full_html=False)  # open html in the browser
    plotly.io.write_html(fig, file='plot.html')  # save the html file
```
