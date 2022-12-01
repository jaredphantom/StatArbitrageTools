from statarb import generateBestPairs, getCointCoeff, getPrices, returnEntry, testGoodPairs, allCrypto, binanceCrypto
from backtest import backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta as td

def generateCorrMatrix(coins: list[str], threshold: float, start: dt = dt(2021, 1, 1), end: dt = dt.today()) -> pd.DataFrame:
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

    print(f"\nMean pairwise correlation: {(corr.sum().sum() / (len(corr.index) ** 2 - len(corr.index))):.4f}")
    return pairs

def backtestPortfolio(coins: list[tuple[str]], constant: float, leverage: int, start: dt = dt(2021, 1, 1), end: dt = dt.today()):
    curves = []
    for pair in coins:
        pairDict = getPrices(crypto=pair, progress=False, start=start, end=end)
        coeff = getCointCoeff(coinPrices=pairDict[pair[0]], coin2Prices=pairDict[pair[1]], verbose=False)
        base, quote = returnEntry(coins=pairDict, coeff=coeff)
        curves.append(backtest(coins=pairDict, coeff=coeff, baseEquity=base, quoteEquity=quote, constant=constant, graph=False, leverage=leverage, start=start, end=end))

    totalCurve = []
    for i, v in enumerate(curves[0]):
        equity = v
        for curve in curves[1:]:
            equity += curve[i]
        totalCurve.append(equity)

    print("-"*50, f"\nPortfolio Backtest: {len(curves)} pairs ({leverage if leverage >= 1 else 1}x leverage)\n" + "-"*50)
    print(f"Return ({str(end - start).split(',')[0]}): {(((totalCurve[-1] - totalCurve[0]) / totalCurve[0]) * 100):.3f}%")
    print(f"${totalCurve[0]:.2f}  ->  ${totalCurve[-1]:.2f}")
    print(f"Sharpe ratio: {calculateSharpe(totalCurve):.3f}")

    plt.figure()
    plt.plot(np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D")), totalCurve, 
             color="green" if totalCurve[-1] > totalCurve[0] else "red")
    plt.title("Portfolio Equity")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.show()

def calculateSharpe(equity: list[float]) -> float:
    returns = []
    for e1, e2 in zip(equity, equity[1:]):
        returns.append(((e2 - e1) / e1) * 100)
    
    return (np.mean(returns) / np.std(returns)) * (365 ** 0.5)

def runPortfolioBacktest(coins: list[str], threshold: float, constant: float, leverage: int, size: int, start: dt = dt(2021, 1, 1), end: dt = dt.today()):
    corr = generateCorrMatrix(coins=coins, threshold=threshold, start=start, end=end)
    portfolio = generatePortfolio(corr=corr, size=size)
    backtestPortfolio(coins=portfolio, constant=constant, leverage=leverage, start=start, end=end)

if __name__ == "__main__":
    runPortfolioBacktest(coins=binanceCrypto, threshold=0.9, constant=1.5, leverage=2, size=25, start=dt(2021, 11, 1))
    testGoodPairs(pairs=generatePortfolio(corr=generateCorrMatrix(coins=binanceCrypto, threshold=0.9, start=dt(2021, 11, 1)), size=25), graph=False, start=dt(2021, 11, 1))
