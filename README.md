# StatArbitrageTools

**statarb.py will generate crypto pairs that have tested for cointegration with a confidence level > 99.5%, and with a correlation higher than 0.8, since Jan 1st 2021 (by default)**

**backtest.py can test the returns of a pairs trading strategy using the generated crypto pairs, assuming no fees or slippage**

**portfolio.py creates a portfolio of n cointegrated pairs with minimal mean pairwise correlation between each pair, in order to produce an uncorrelated returns stream with positive expectation. Performance of generated portfolios can be backtested and will output metrics including a sharpe ratio which assumes a risk-free rate of 0**

**Can also set up a Discord webhook to send daily alerts for entry opportunities**
