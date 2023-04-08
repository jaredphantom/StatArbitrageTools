from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from datetime import datetime as dt
from statarb import getPrices, getCointCoeff, getCointSummary, generateBestPairs, returnEntry, allCrypto

class Positions(Enum):
    LONG = 1
    SHORT = -1

class Trader:

    def __init__(self, equity: float, leverage: int = 1) -> None:
        self._startEquity = equity
        self._equity = equity
        self._equityCurve = []
        self._position = False
        self._positionType = None
        self._openPrice = 0
        self._trades = 0
        self._pnlArray = []
        self._startEquities = []
        self._leverage = leverage if leverage >= 1 else 1
        self._decay = 0.9999

    def openPosition(self, position: Positions, price: float):
        if not self._position:
            self.plotPriceChange(price)
            self._openPrice = price
            self._positionType = position
            self._position = True

    def closePosition(self, price: float):
        if self._position:
            change = self.plotPriceChange(price)
            self._equity *= change
            self._pnlArray.append(self._equity - self._startEquity)
            self._startEquities.append(self._startEquity)
            self._trades += 1
            self.reset()
    
    def plotPriceChange(self, price: float) -> float:
        if self._position:
            self._equity *= self._decay
            priceChange = 1 + (self._positionType.value * ((price - self._openPrice) / self._openPrice) * self._leverage)
            if priceChange <= 0 and (self._positionType == Positions.SHORT or (self._positionType == Positions.LONG and self._leverage > 1)):
                self._equity = 0
            self._equityCurve.append(self._equity * priceChange)
            return priceChange
        else:
            self._equityCurve.append(self._equity)
            return 1

    def getEquity(self) -> float: 
        return self._equity

    def getStartEquity(self) -> float:
        return self._startEquity
    
    def setEquity(self, newEquity: float):
        self._startEquity = newEquity
        self._equity = newEquity
    
    def getEquityCurve(self) -> list[float]:
        return self._equityCurve

    def getNumTrades(self) -> int:
        return self._trades

    def getPnlArray(self) -> list[float]:
        return self._pnlArray
    
    def getStartEquities(self) -> list[float]:
        return self._startEquities

    def getCurrentPnl(self) -> float:
        return self._equityCurve[-1] - self._startEquity

    def checkPosition(self) -> bool:
        return self._position

    def reset(self):
        self._openPrice = 0
        self._positionType = None
        self._position = False

def rebalance(totalEquity: float, baseEquity: float, quoteEquity: float) -> tuple[float]:
    equityMultiplier = totalEquity / (baseEquity + quoteEquity)
    return equityMultiplier * baseEquity, equityMultiplier * quoteEquity

def splitTimeframe(start: dt, end: dt, interval: int = 3) -> list[dt]:
    times = []
    intv = interval if interval >= 2 else 2
    diff = (end - start) / intv
    for i in range(intv):
        times.append(start + (diff * i))
    times.append(end)
    return times    

def generateSpreadGraph(coinPrices: list[float], coin2Prices: list[float], coeff: pd.Series) -> list[float]:
    spreadGraph = []
    for c1, c2 in zip(coinPrices, coin2Prices):
        spreadGraph.append(c1 + (coeff["x1"] * c2))

    return spreadGraph

def backtest(coins: Dict[str, np.ndarray], constant: float = 1.5, graph: bool = True, leverage: int = 1,
             start: dt = dt(2021, 1, 1), end: dt = None) -> tuple[list[float]]:
    if end is None:
        end = dt.today()

    prices = getPrices(list(coins.keys()), progress=False, start=start, end=end)
    OverallPosition = None
    stdCoeff = constant if constant >= 1 else 1

    baseTrader = None
    quoteTrader = None

    pairPrices = list(zip(prices[list(coins.keys())[0]], prices[list(coins.keys())[1]]))
    totalDays = 0

    for index, values in enumerate(pairPrices):

        totalDays = index

        if index < int(len(pairPrices) / 4):
            continue

        c1, c2 = values[0], values[1]

        summary = getCointSummary(coinPrices=prices[list(coins.keys())[0]][:index + 1],
                                  coin2Prices=prices[list(coins.keys())[1]][:index + 1])
        tstat, pval = summary[0], summary[1]

        coeff = getCointCoeff(coinPrices=prices[list(coins.keys())[0]][:index + 1],
                              coin2Prices=prices[list(coins.keys())[1]][:index + 1],
                              verbose=False)

        tempDict = {}
        tempDict[list(coins.keys())[0]] = [c1]
        tempDict[list(coins.keys())[1]] = [c2]

        if baseTrader is None and quoteTrader is None:
            if tstat <= -5:
                baseEquity, quoteEquity = returnEntry(coins=tempDict, coeff=coeff)
                baseTrader = Trader(equity=baseEquity, leverage=leverage)
                quoteTrader = Trader(equity=quoteEquity, leverage=leverage)
                originalEquity = baseEquity + quoteEquity
            else:
                continue
        else:
            baseEquity, quoteEquity = returnEntry(coins=tempDict, coeff=coeff,
                                                  maxEntry=baseTrader.getStartEquity()+quoteTrader.getStartEquity())

        spreadGraph = generateSpreadGraph(coinPrices=prices[list(coins.keys())[0]][:index + 1],
                                          coin2Prices=prices[list(coins.keys())[1]][:index + 1],
                                          coeff=coeff)
        mean = np.mean(spreadGraph)
        stddev = np.std(spreadGraph)
        s = spreadGraph[-1]

        if s < mean - (stdCoeff * stddev) and pval <= 0.005 and OverallPosition is None:
            baseTrader.setEquity(baseEquity)
            quoteTrader.setEquity(quoteEquity)
            baseTrader.openPosition(Positions.LONG, c1)
            quoteTrader.openPosition(Positions.SHORT, c2)
            OverallPosition = Positions.LONG
        elif s > mean + (stdCoeff * stddev) and pval <= 0.005 and OverallPosition is None:
            baseTrader.setEquity(baseEquity)
            quoteTrader.setEquity(quoteEquity)
            baseTrader.openPosition(Positions.SHORT, c1)
            quoteTrader.openPosition(Positions.LONG, c2)
            OverallPosition = Positions.SHORT
        elif (s >= mean or pval > 0.15) and OverallPosition == Positions.LONG:
            baseTrader.closePosition(c1)
            quoteTrader.closePosition(c2)
            base, quote = rebalance(totalEquity=baseTrader.getEquity()+quoteTrader.getEquity(), baseEquity=baseEquity, quoteEquity=quoteEquity)
            baseTrader.setEquity(base)
            quoteTrader.setEquity(quote)
            OverallPosition = None 
        elif (s <= mean or pval > 0.15) and OverallPosition == Positions.SHORT:
            baseTrader.closePosition(c1)
            quoteTrader.closePosition(c2)
            base, quote = rebalance(totalEquity=baseTrader.getEquity()+quoteTrader.getEquity(), baseEquity=baseEquity, quoteEquity=quoteEquity)
            baseTrader.setEquity(base)
            quoteTrader.setEquity(quote)
            OverallPosition = None

        else:
            baseTrader.plotPriceChange(c1)
            quoteTrader.plotPriceChange(c2)

    if baseTrader is None or quoteTrader is None:
        print(f"\n{list(coins.keys())[0]}/{list(coins.keys())[1]}: No trades")
        return [50 for _ in range(totalDays + 1)], []

    totalStartEquity = baseTrader.getStartEquity() + quoteTrader.getStartEquity()
    currentEquity = baseTrader.getEquityCurve()[-1] + quoteTrader.getEquityCurve()[-1]
    equityChange = (currentEquity - originalEquity) / originalEquity
    baseHold = (prices[list(coins.keys())[0]][-1] - prices[list(coins.keys())[0]][0]) / prices[list(coins.keys())[0]][0]
    quoteHold = (prices[list(coins.keys())[1]][-1] - prices[list(coins.keys())[1]][0]) / prices[list(coins.keys())[1]][0]
    print(f"\n{list(coins.keys())[0]}/{list(coins.keys())[1]} (Strategy): {'+' if equityChange > 0 else ''}{(equityChange * 100):.2f}%")
    print(f"${originalEquity:.2f}  ->  ${currentEquity:.2f}")
    print(f"{list(coins.keys())[0]} (Hold): {'+' if baseHold > 0 else ''}{(baseHold * 100):.2f}%")
    print(f"{list(coins.keys())[1]} (Hold): {'+' if quoteHold > 0 else ''}{(quoteHold * 100):.2f}%")
    wins = 0
    returns = []
    for base_PnL, quote_PnL, base_equity, quote_equity in zip(baseTrader.getPnlArray(),
                                                              quoteTrader.getPnlArray(),
                                                              baseTrader.getStartEquities(),
                                                              quoteTrader.getStartEquities()):
        totalPnl = base_PnL + quote_PnL
        totalEquity = base_equity + quote_equity
        returns.append(round((totalPnl / totalEquity) * 100, 2))
        if totalPnl > 0:
            wins += 1
    try:
        print(f"Win rate: {round((wins / baseTrader.getNumTrades()) * 100, 2)}% ({wins}/{baseTrader.getNumTrades()})")
    except ZeroDivisionError:
        print(f"Win rate: NaN ({wins}/{baseTrader.getNumTrades()})")
    currentPnlPercent = round(((baseTrader.getCurrentPnl() + quoteTrader.getCurrentPnl()) / totalStartEquity) * 100, 2)
    print(f"Open position: {baseTrader.checkPosition()} " 
          f"{'(' + ('+' if currentPnlPercent > 0 else '') + str(currentPnlPercent) + '%)' if baseTrader.checkPosition() else ''}")

    totalCurve = []
    for c1, c2 in zip(baseTrader.getEquityCurve(), quoteTrader.getEquityCurve()):
        totalCurve.append(c1 + c2)

    totalCurve = [(baseTrader.getEquityCurve()[0] + quoteTrader.getEquityCurve()[0]) for _ in range((totalDays + 1) - len(totalCurve))] + totalCurve
    baseCurve = [baseTrader.getEquityCurve()[0] for _ in range((totalDays + 1) - len(baseTrader.getEquityCurve()))] + baseTrader.getEquityCurve()
    quoteCurve = [quoteTrader.getEquityCurve()[0] for _ in range((totalDays + 1) - len(quoteTrader.getEquityCurve()))] + quoteTrader.getEquityCurve()

    if graph:
        plt.figure()
        intervals = np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D"))
        length = min(len(intervals), len(totalCurve))
        plt.plot(intervals[-length:], totalCurve[-length:])
        plt.plot(intervals[-length:], baseCurve[-length:])
        plt.plot(intervals[-length:], quoteCurve[-length:])
        plt.legend(["Total"] + list(coins.keys()), loc=0, fontsize="small")
        plt.title(f"{list(coins.keys())[0]}/{list(coins.keys())[1]}")
        plt.xlabel("Time")
        plt.ylabel("Equity ($)")
        plt.show()

    return totalCurve, returns

def runBacktestAll(threshold: float, constant: float, graph: bool, leverage: int, start: dt = dt(2021, 1, 1), end: dt = None):
    if end is None:
        end = dt.today()

    for pair in generateBestPairs(crypto=allCrypto, threshold=threshold, start=start, end=end):
        cryptoDict = getPrices(crypto=pair, progress=False, start=start, end=end)
        backtest(coins=cryptoDict, constant=constant, graph=graph, leverage=leverage, start=start, end=end)
