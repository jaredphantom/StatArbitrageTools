from statarb import generateBestPairs, getCointCoeff, getPrices, testGoodPairs, binanceCrypto, dydxCrypto
from backtest import backtest
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from datetime import datetime as dt
from math import comb, ceil, floor

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
    print(f"Mean pairwise correlation ({numSamples} random samples): {min(sampleCorrs):.4f}, {np.mean(sampleCorrs):.4f}")
    print("\nOptimal Portfolio: ", end="")
    print(*pairs, sep=", ")
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

    print("-"*50, f"\nPortfolio Backtest: {len(curves)} pairs ({leverage if leverage >= 1 else 1}x leverage)\n" + "-"*50)
    print(f"Return ({len(totalCurve)} days): {(((totalCurve[-1] - totalCurve[0]) / totalCurve[0]) * 100):.3f}%")
    print(f"${totalCurve[0]:.2f}  ->  ${totalCurve[-1]:.2f}")
    print(f"Sharpe ratio: {calculateSharpe(totalCurve):.3f}")

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
    plt.title(f"Trade Returns Distribution\nμ = {np.mean(returns):.2f}, σ = {np.std(returns):.2f}")
    plt.xlabel("Trade Returns (%)")
    plt.ylabel("Number of trades")
    plt.xticks(np.linspace(floor(min(returns)), ceil(max(returns)), 10))
    plt.tight_layout()

    dailyReturns = []
    for i in range(int(len(totalCurve) / 4), len(totalCurve) - 1):
        dailyReturns.append(((totalCurve[i+1] - totalCurve[i]) / totalCurve[i]) * 100)
    dailyBtc = dailyBtc[-len(dailyReturns):]

    returnsDict = {
        "Portfolio": dailyReturns,
        "BTC": dailyBtc
    }
    returns_df = pd.DataFrame(returnsDict)

    sb.displot(returns_df, legend=True, kde=True, height=6, aspect=1)
    plt.title(f"Daily Returns Distribution - α = {(np.mean(dailyReturns) - np.mean(dailyBtc)):.2f}, β = {(np.std(dailyReturns) / np.std(dailyBtc)):.2f}" +
              f"\n$μ^P$ = {np.mean(dailyReturns):.2f}, $σ^P$ = {np.std(dailyReturns):.2f}" +
              f"\n$μ^B$ = {np.mean(dailyBtc):.2f}, $σ^B$ = {np.std(dailyBtc):.2f}")
    plt.xlabel("Daily Returns (%)")
    plt.ylabel("Number of days")
    plt.xticks(np.linspace(floor(min(min(dailyReturns), min(dailyBtc))),
                           ceil(max(max(dailyReturns), max(dailyBtc))),
                           10))
    plt.tight_layout()

    plt.show()

def calculateSharpe(equity: list[float]) -> float:
    returns = []
    for e1, e2 in zip(equity, equity[1:]):
        returns.append(((e2 - e1) / e1) * 100)
    
    std = np.std(returns)
    if std == 0:
        return np.nan
    mean = np.mean(returns)
    return ((mean - 0.013) / std) * (365 ** 0.5)

def runPortfolioBacktest(coins: list[str], threshold: float, constant: float, leverage: int, size: int, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    corr = generateCorrMatrix(coins=coins, threshold=threshold, start=start, end=end)
    portfolio = generatePortfolio(corr=corr, size=size)
    backtestPortfolio(coins=portfolio, constant=constant, leverage=leverage, start=start, end=end)

if __name__ == "__main__":
    runPortfolioBacktest(coins=dydxCrypto, threshold=0.75, constant=1, leverage=1, size=25, start=dt(2022, 1, 1))
