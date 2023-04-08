import yfinance as yf
import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from arch.unitroot import engle_granger
from typing import Dict, Union

allCrypto = [
             "BTC", "ETH", "ALGO", "XRP", "ADA", "QNT", "HNT", "SOL", "BCH", "ETC", "EOS", "NEO",
             "DOGE", "SHIB", "XTZ", "DOT", "LTC", "MATIC", "LINK", "VET", "HBAR", "MANA", "SAND",
             "ANT", "XLM", "DAI", "TRX", "UNI", "AVAX", "ATOM", "FTT", "NEAR", "XMR", "FIL", "EGLD",
             "AAVE", "THETA", "CAKE", "FTM", "RUNE", "NEXO", "ENJ", "LRC", "ANKR", "CELR", "XNO", "HT", 
             "AMP", "MKR", "OKB", "CRO", "ZEC", "DASH", "BAT", "1INCH", "CELO", "HOT", "GALA", "YFI",
             "ONE", "HIVE", "ICX", "FLUX"
            ]

binanceCrypto = [
                 "BTC", "ETH", "ALGO", "XRP", "ADA", "QNT", "HNT", "SOL", "BCH", "ETC", "EOS", "NEO",
                 "DOGE", "XTZ", "DOT", "LTC", "MATIC", "LINK", "VET", "HBAR", "MANA", "SAND",
                 "ANT", "XLM", "TRX", "UNI", "AVAX", "ATOM", "FTT", "NEAR", "XMR", "FIL", "EGLD",
                 "AAVE", "THETA", "FTM", "RUNE", "ENJ", "LRC", "ANKR", "CELR", "MKR", "ZEC", "DASH", 
                 "BAT", "1INCH", "CELO", "HOT", "GALA", "YFI", "ONE", "ICX"
                ]

dydxCrypto = [
              "BTC", "ETH", "ALGO", "XRP", "ADA", "SOL", "BCH", "ETC", "EOS",
              "DOGE", "XTZ", "DOT", "LTC", "MATIC", "LINK", "CRV", "SUSHI",
              "XLM", "TRX", "UNI", "AVAX", "ATOM", "NEAR", "XMR", "FIL", 
              "AAVE", "RUNE", "ENJ", "ICP", "SNX", "COMP", "ZRX", "UMA",
              "MKR", "ZEC", "1INCH", "CELO", "YFI"
             ]

def getPrices(crypto: list[str], start: dt = dt(2021, 1, 1), end: dt = None, progress: bool = True) -> Dict[str, np.ndarray]:
    if end is None:
        end = dt.today()
    
    cryptoDict = {}

    for coin in crypto:
        data: pd.DataFrame = yf.download(tickers=f'{coin}-USD', start=start, end=end, progress=progress)
        data.index = pd.DatetimeIndex(data.index)
        data = data.asfreq("D")
        data.interpolate(inplace=True)
        closes = data["Close"].to_numpy()
        cryptoDict[coin] = closes

    return cryptoDict

def drawCorrGraphs(coins: Dict[str, np.ndarray], start: dt = dt(2021, 1, 1), end: dt = None, log: bool = True):
    if end is None:
        end = dt.today()
    
    plt.figure()
    intervals = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D"))
    for coin in coins:
        length = min(len(intervals), len(coins[coin]))
        if log:
            plt.plot(intervals[-length:], np.log1p(coins[coin])[-length:])
        else:
            plt.plot(intervals[-length:], coins[coin][-length:])

    plt.legend(list(coins.keys()), loc=0, fontsize="x-small")
    if log:
        plt.ylabel("log(1 + Price)")
    else:
        plt.ylabel("Price")
    plt.xlabel("Time")
    plt.draw()

    prices = pd.DataFrame(coins)
    plt.figure()
    sb.heatmap(prices.corr(), cmap="Blues", annot=True)
    plt.title("Price Correlation")
    plt.draw()

def getCorrelations(coins: Dict[str, np.ndarray], threshold: float = -1, above: bool = True) -> Dict[str, list[tuple[Union[str, float]]]]:
    correlations = {}
    for baseCoin in coins:
        corrDict = {}
        for otherCoin in coins:
            corr = float(pd.DataFrame({key: coins[key] for key in [baseCoin, otherCoin]}).corr()[otherCoin].to_numpy()[0])
            if corr >= threshold and otherCoin != baseCoin and above:
                corrDict[otherCoin] = corr
            if corr <= threshold and otherCoin != baseCoin and not above:
                corrDict[otherCoin] = corr
        for corr in corrDict:
            correlations[baseCoin] = correlations.setdefault(baseCoin, []) + [(corr, corrDict[corr])]

    return correlations

def getPairCorr(coins: Dict[str, np.ndarray]) -> float:
    try:
        return float(pd.DataFrame(coins).corr()[list(coins.keys())[1]].to_numpy()[0])
    except ValueError:
        return 0

def printCorrelations(correlations: Dict[str, list[tuple[Union[str, float]]]]):
    for coin in correlations:
        try:
            if len(correlations[coin]) == 0:
                continue
            else:
                print(f"\n{coin} Correlations: ")
                for corr in correlations[coin]:
                    print(f"{corr[0]}: {corr[1]:.3f}")
        except TypeError:
            continue

def getCointCoeff(coinPrices: list[float], coin2Prices: list[float], verbose: bool = True) -> pd.Series:
    cointStats = engle_granger(coinPrices, coin2Prices)
    if verbose:
        print(f"\n{cointStats.summary()}")
        print(f"\n{cointStats.cointegrating_vector}")
    return cointStats.cointegrating_vector

def getCointSummary(coinPrices: list[float], coin2Prices: list[float]) -> tuple[float]:
    cointStats = engle_granger(coinPrices, coin2Prices)
    return cointStats.stat, cointStats.pvalue

def drawSpreadGraph(coinPrices: list[float], coin2Prices: list[float], coeff: pd.Series, title: str,
                    start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    new = []
    for c1, c2 in zip(coinPrices, coin2Prices):
        new.append(c1 + (coeff["x1"] * c2))

    intervals = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D"))
    length = min(len(intervals), len(new))

    plt.figure()
    plt.plot(intervals[-length:], np.array(new)[-length:])
    plt.title(title)
    plt.ylabel("Price Spread ($)")
    plt.xlabel("Time")
    mean = np.mean(new)
    stddev = np.std(new)
    stddev_up = [mean + (1.5*stddev) for _ in range(len(new))]
    stddev_down = [mean - (1.5*stddev) for _ in range(len(new))]
    stddev_up2 = [mean + (2*stddev) for _ in range(len(new))]
    stddev_down2 = [mean - (2*stddev) for _ in range(len(new))]
    mean = [mean for _ in range(len(new))]
    plt.plot(intervals[-length:], mean[-length:])
    plt.plot(intervals[-length:], stddev_up[-length:], color="purple")
    plt.plot(intervals[-length:], stddev_down[-length:], color="purple")
    plt.plot(intervals[-length:], stddev_up2[-length:], color="pink")
    plt.plot(intervals[-length:], stddev_down2[-length:], color="pink")
    plt.draw()

def testGoodPairs(pairs: list[tuple[str]], graph: bool, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    for pair in pairs:
        testPair(*pair, graph=graph, start=start, end=end)

def testPair(testCoin1: str, testCoin2: str, graph: bool, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    testCrypto = [testCoin1, testCoin2]
    print("-"*50, f"\n\t\t{testCrypto[0]}/{testCrypto[1]}\n" + "-"*50)
    cryptoDict = getPrices(crypto=testCrypto, start=start, end=end, progress=False)
    printCorrelations(correlations=getCorrelations(coins=cryptoDict))
    cointCoeff = getCointCoeff(coinPrices=cryptoDict[testCrypto[0]], coin2Prices=cryptoDict[testCrypto[1]])
    calculateEntry(coins=cryptoDict, coeff=cointCoeff)
    
    if graph:
        drawCorrGraphs(coins=cryptoDict, start=start, end=end)
        drawSpreadGraph(coinPrices=cryptoDict[testCrypto[0]], coin2Prices=cryptoDict[testCrypto[1]], coeff=cointCoeff,
                        title=f"{testCrypto[0]} - {abs(cointCoeff['x1']):.3f} {testCrypto[1]}", start=start, end=end)
        plt.show()

def testAll(crypto: list[str], threshold: float, graph: bool, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()
    
    cryptoDict = getPrices(crypto=crypto, start=start, end=end)
    printCorrelations(correlations=getCorrelations(coins=cryptoDict, threshold=threshold))

    if graph:
        drawCorrGraphs(coins=cryptoDict, start=start, end=end)
        plt.show()

def generateBestPairs(crypto: list[str], threshold: float, start: dt = dt(2021, 1, 1), end: dt = None) -> list[tuple[str]]:
    if end is None:
        end = dt.today()
    
    cryptoDict = getPrices(crypto=crypto, start=start, end=end)
    bestPairs = []
    for c1 in cryptoDict:
        for c2 in cryptoDict:
            if c1 != c2:
                pairDict = {c1: cryptoDict[c1], c2: cryptoDict[c2]}
                corr = getPairCorr(coins=pairDict)
                if corr >= threshold:
                    cointSummary = getCointSummary(coinPrices=pairDict[c1], coin2Prices=pairDict[c2])
                    tstat = cointSummary[0]
                    if tstat <= -5:
                        if (c2, c1) not in bestPairs:
                            bestPairs.append((c1, c2))
    
    print(f"\n{bestPairs}")
    return bestPairs

def calculateEntry(coins: Dict[str, np.ndarray], coeff: pd.Series, maxEntry: float = 50):
    coin1Price, coin2Price = coins[list(coins.keys())[0]][-1], coins[list(coins.keys())[1]][-1]
    spreadGraph = []
    for c1, c2 in zip(coins[list(coins.keys())[0]], coins[list(coins.keys())[1]]):
        spreadGraph.append(c1 + (coeff["x1"] * c2))
    mean = np.mean(spreadGraph)
    stddev = np.std(spreadGraph)

    cointVector = abs(coeff["x1"])
    spread = coin1Price + (cointVector * coin2Price)
    divisor = spread / maxEntry
    baseVal = coin1Price / divisor
    quoteVal = (cointVector * coin2Price) / divisor
    if spreadGraph[-1] < mean:
        print(f"\n{list(coins.keys())[0]}: ${baseVal:.2f}, {list(coins.keys())[1]}: -${quoteVal:.2f}")
        if spreadGraph[-1] < mean - (1.5 * stddev):
            print("\nLONG!!!")
    else:
        print(f"\n{list(coins.keys())[0]}: -${baseVal:.2f}, {list(coins.keys())[1]}: ${quoteVal:.2f}")
        if spreadGraph[-1] > mean + (1.5 * stddev):
            print("\nSHORT!!!")

def returnEntry(coins: Dict[str, np.ndarray], coeff: pd.Series, maxEntry: float = 50) -> tuple[float]:
    coin1Price, coin2Price = coins[list(coins.keys())[0]][-1], coins[list(coins.keys())[1]][-1]
    cointVector = abs(coeff["x1"])
    spread = coin1Price + (cointVector * coin2Price)
    divisor = spread / maxEntry
    baseVal = coin1Price / divisor
    quoteVal = (cointVector * coin2Price) / divisor

    return baseVal, quoteVal

