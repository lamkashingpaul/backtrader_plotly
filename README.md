# backtrader_plotly

Plot `backtrader`'s result using `plotly` instead of the default `matplotlib`

This is an experimental package, and it is done by replacing original `matplotlib` method calls.

## Installation

`$ pip install backtrader-plotly==1.5.0.dev1`

## Features

- Support for Multiple Strategies Plotting (Added from 1.5.0.dev1)

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

# add strategies
cerebro.addstrategy(IchimokuStrategy)
cerebro.addstrategy(SMACrossStrategy)

# after adding data and strategy
cerebro.run()

# define plot scheme with new additional scheme arguments
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
```
