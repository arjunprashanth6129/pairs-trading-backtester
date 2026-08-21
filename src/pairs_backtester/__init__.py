"""A pairs-trading (statistical arbitrage) backtester.

The package is deliberately split so that the one thing that can invalidate a
backtest - using information the strategy could not have had - is confined to a
single seam. `pairs.fit_pair` is the only place parameters are estimated, and it
only ever sees a formation slice. `signals` and `backtest` consume those
parameters and never re-estimate anything.
"""

__version__ = "0.1.0"
