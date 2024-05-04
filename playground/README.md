# Example Usage:
`from utils.analytics import Methods, Basics, Report`

#### library providing easy testing of sma, ema, simple lin reg, and multi-feature lin reg
- `methods = Methods()`

#### library providing some basic ds tools for quick testing
- `basics = Basics()`: to get started
- `basics.pad`: padding preds vs actual for mapping
- `basics.simple_plot`: quickly plot multiple sets of time series data
- `basics.basic_test`: clear formatting for variance tests, stdev tests, and more on time series data

#### Report
- `report = Report(<list of tickers>, <time>, <requested methods (see __init__)>)`: get started
- see report for a number of graphs & metrics you can draw up on the requested tickers
