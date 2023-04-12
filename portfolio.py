from statarb import generateBestPairs, getCointCoeff, getPrices, dydxCrypto
from backtest import backtest
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime as dt
from math import comb, ceil, floor

TRADING_DAYS_IN_YEAR = 252
RISK_FREE_RATE = 4.8 / TRADING_DAYS_IN_YEAR

def generateCorrMatrix(coins: list[str], threshold: float, start: dt = dt(2021, 1, 1), end: dt = None) -> pd.DataFrame:
    if end is None:
        end = dt.today()
    
    bestPairs = generateBestPairs(crypto=coins, threshold=threshold, start=start, end=end)
    bestDict = {}
    for pair in bestPairs:
        prices = getPrices(crypto=pair, progress=False, start=start, end=end)
        coeff = getCointCoeff(coinPrices=prices[pair[0]], coin2Prices=prices[pair[1]], verbose=False)
        spread = []
        for c1, c2 in zip(prices[pair[0]], prices[pair[1]]):
            spread.append(c1 + (coeff["x1"] * c2))
        bestDict[f"{pair[0]}/{pair[1]}"] = spread

    return pd.DataFrame(bestDict).corr()

def generatePortfolio(corr: pd.DataFrame, size: int) -> list[tuple[str]]:
    sampleCorrs = []
    sampleSize = min(size, len(corr.index))
    numSamples = min(1000, comb(len(corr.index), sampleSize))
    for _ in range(numSamples):
        randomSample = corr.sample(sampleSize)
        sampleMatrix = randomSample[randomSample.index]
        np.fill_diagonal(sampleMatrix.values, np.nan)
        meanCorr = (sampleMatrix.sum().sum() / (len(sampleMatrix.index) ** 2 - len(sampleMatrix.index)))
        sampleCorrs.append(meanCorr)

    while len(corr.index) > size:
        sums = corr.sum()
        sums.sort_values(inplace=True)
        bottom = sums.tail(1).index
        corr.drop(index=bottom, inplace=True)
        corr.drop(columns=bottom, inplace=True)

    np.fill_diagonal(corr.values, np.nan)
    portfolio = list(map(str, corr.columns))
    pairs = []
    for pair in portfolio:
        pairs.append(tuple(pair.split("/")))

    print(f"\nMean pairwise correlation (optimal portfolio): {(corr.sum().sum() / (len(corr.index) ** 2 - len(corr.index))):.4f}")
    print(f"Mean pairwise correlation ({numSamples} random samples): {min(sampleCorrs):.4f} (min), {np.mean(sampleCorrs):.4f} (avg), {max(sampleCorrs):.4f} (max)")
    print("\nOptimal Portfolio: ", end="")
    print(*list(map("/".join, pairs)), sep=", ")
    return pairs

def backtestPortfolio(coins: list[tuple[str]], constant: float, leverage: int, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    curves = []
    returns = []
    for pair in coins:
        pairDict = getPrices(crypto=pair, progress=False, start=start, end=end)
        backtestResults = backtest(coins=pairDict, constant=constant, graph=False, leverage=leverage, start=start, end=end)
        curves.append(backtestResults[0])
        returns += backtestResults[1]

    totalCurve = []
    for i, v in enumerate(curves[0]):
        equity = v
        for curve in curves[1:]:
            equity += curve[i]
        totalCurve.append(equity)

    plt.figure()
    intervals = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D"))
    length = min(len(intervals), len(totalCurve))
    plt.plot(intervals[-length:], totalCurve[-length:],
             color="green" if totalCurve[-1] > totalCurve[0] else "red")
    
    btcPrices = getPrices(crypto=["BTC"], start=start, end=end, progress=False)["BTC"]
    buynhold = [totalCurve[0]]
    dailyBtc = []
    for i in range(len(btcPrices) - 1):
        dailyReturn = (btcPrices[i+1] - btcPrices[i]) / btcPrices[i]
        dailyBtc.append(dailyReturn * 100)
        buynhold.append(buynhold[i] * (dailyReturn + 1))

    plt.plot(intervals[-length:], buynhold[-length:], color="purple")
    plt.legend(["Portfolio", "BTC buy&hold"], loc=0, fontsize="x-small")
    plt.title("Portfolio Equity")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")

    sb.displot(returns, color="green" if np.mean(returns) > 0 else "red", kde=True, rug=True, height=6, aspect=1)
    plt.title(f"Trade Returns Distribution\nμ = {np.mean(returns):.2f}%, σ = {np.std(returns):.2f}%")
    plt.xlabel("Trade Returns (%)")
    plt.ylabel("Number of trades")
    plt.xticks(np.linspace(floor(min(returns)), ceil(max(returns)), 10))
    plt.tight_layout()

    dailyReturns = []
    for i in range(int(len(totalCurve) / 4), len(totalCurve) - 1):
        dailyReturns.append(((totalCurve[i+1] - totalCurve[i]) / totalCurve[i]) * 100)
    dailyBtc = dailyBtc[-len(dailyReturns):]

    portfolio_std = np.std(dailyReturns)
    portfolio_mean = np.mean(dailyReturns)
    benchmark_std = np.std(dailyBtc)
    benchmark_mean = np.mean(dailyBtc)

    returnsDict = {
        "Portfolio": dailyReturns,
        "BTC": dailyBtc
    }
    returns_df = pd.DataFrame(returnsDict)

    sb.displot(returns_df, legend=True, kde=True, height=6, aspect=1)
    plt.title(f"Daily Returns Distribution" +
              f"\n$μ^P$ = {portfolio_mean:.2f}%, $σ^P$ = {portfolio_std:.2f}%" +
              f"\n$μ^B$ = {benchmark_mean:.2f}%, $σ^B$ = {benchmark_std:.2f}%")
    plt.xlabel("Daily Returns (%)")
    plt.ylabel("Number of days")
    plt.xticks(np.linspace(floor(min(min(dailyReturns), min(dailyBtc))),
                           ceil(max(max(dailyReturns), max(dailyBtc))),
                           10))
    plt.tight_layout()

    print()
    print("-"*50, f"\nPortfolio Backtest: {len(curves)} pairs ({leverage if leverage >= 1 else 1}x leverage)\n" + "-"*50)
    print(f"Return ({len(totalCurve)} days): {(((totalCurve[-1] - totalCurve[0]) / totalCurve[0]) * 100):.3f}%")
    print(f"Strategy: ${totalCurve[0]:.2f}  ->  ${totalCurve[-1]:.2f}")
    print(f"BTC Hold: ${buynhold[0]:.2f}  ->  ${buynhold[-1]:.2f}")
    if portfolio_std == 0:
        sharpe = np.nan
    else:
        sharpe = ((portfolio_mean - RISK_FREE_RATE) / portfolio_std) * (TRADING_DAYS_IN_YEAR ** 0.5)
    print(f"Sharpe Ratio: {sharpe:.3f}")
    var = (portfolio_mean - 2.33 * portfolio_std) * -1
    print(f"Daily VaR: {var:.3f}%")
    print(f"Max Drawdown: {getMaxDrawdown(totalCurve):.2f}%")
    beta = returns_df.corr()['BTC'].to_numpy()[0] * (portfolio_std / benchmark_std)
    alpha = (portfolio_mean - RISK_FREE_RATE - (beta * (benchmark_mean - RISK_FREE_RATE))) * TRADING_DAYS_IN_YEAR
    print(f"α = {alpha:.3f}%, β = {beta:.5f}")

    plt.show()

def getMaxDrawdown(equityCurve: list[float]) -> float:
    mdd = 0
    peak = -99999

    for equity in equityCurve:
        if equity > peak:
            peak = equity
        dd = 100 * (peak - equity) / peak
        if dd > mdd:
            mdd = dd
    
    return mdd

def runPortfolioBacktest(coins: list[str], threshold: float, constant: float, leverage: int, size: int, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    corr = generateCorrMatrix(coins=coins, threshold=threshold, start=start, end=end)
    portfolio = generatePortfolio(corr=corr, size=size)
    backtestPortfolio(coins=portfolio, constant=constant, leverage=leverage, start=start, end=end)

if __name__ == "__main__":
    runPortfolioBacktest(coins=dydxCrypto, threshold=0.75, constant=1, leverage=2, size=10, start=dt(2022, 1, 1))
