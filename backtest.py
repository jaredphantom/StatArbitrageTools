from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from datetime import datetime as dt
from statarb import getPrices, getCointCoeff, generateBestPairs, returnEntry, allCrypto

class Positions(Enum):
    LONG = 1
    SHORT = -1

class Trader:
    def __init__(self, equity: float, leverage: int = 1) -> None:
        self._equity = equity
        self._equityCurve = []
        self._position = False
        self._positionType = None
        self._openPrice = 0
        self._trades = 0
        self._pnlArray = []
        self._leverage = leverage if leverage >= 1 else 1
        self._decay = 0.9997

    def openPosition(self, position: Positions, price: float):
        if not self._position:
            self.plotPriceChange(price)
            self._openPrice = price
            self._positionType = position
            self._position = True

    def closePosition(self, price: float):
        if self._position:
            change = self.plotPriceChange(price)
            tempEquity = self._equity
            self._equity *= change
            self._pnlArray.append(self._equity - tempEquity)
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
    
    def setEquity(self, newEquity: float):
        self._equity = newEquity
    
    def getEquityCurve(self) -> list[float]:
        return self._equityCurve

    def getNumTrades(self) -> int:
        return self._trades

    def getPnlArray(self) -> list[float]:
        return self._pnlArray

    def getCurrentPnl(self) -> float:
        return self._equityCurve[-1] - self._equity

    def checkPosition(self) -> bool:
        return self._position

    def reset(self):
        self._openPrice = 0
        self._positionType = None
        self._position = False

def rebalance(totalEquity: float, baseEquity: float, quoteEquity: float) -> tuple[float]:
    equityMultiplier = totalEquity / (baseEquity + quoteEquity)
    return equityMultiplier * baseEquity, equityMultiplier * quoteEquity

def backtest(coins: Dict[str, np.ndarray], coeff: pd.Series, baseEquity: float, quoteEquity: float, constant: float = 1.5, graph: bool = True,
             leverage: int = 1, start: dt = dt(2021, 1, 1), end: dt = dt.today()) -> list[float]:
    baseTrader = Trader(equity=baseEquity, leverage=leverage)
    quoteTrader = Trader(equity=quoteEquity, leverage=leverage)
    OverallPosition = None

    spreadGraph = []
    for c1, c2 in zip(coins[list(coins.keys())[0]], coins[list(coins.keys())[1]]):
        spreadGraph.append(c1 + (coeff["x1"] * c2))

    mean = np.mean(spreadGraph)
    stddev = np.std(spreadGraph)
    stdCoeff = constant if constant >= 1 else 1

    for c1, c2, s in zip(coins[list(coins.keys())[0]], coins[list(coins.keys())[1]], spreadGraph):

        if s < mean - (stdCoeff * stddev) and OverallPosition is None:
            baseTrader.openPosition(Positions.LONG, c1)
            quoteTrader.openPosition(Positions.SHORT, c2)
            OverallPosition = Positions.LONG
        elif s > mean + (stdCoeff * stddev) and OverallPosition is None:
            baseTrader.openPosition(Positions.SHORT, c1)
            quoteTrader.openPosition(Positions.LONG, c2)
            OverallPosition = Positions.SHORT
        elif s > mean and OverallPosition == Positions.LONG:
            baseTrader.closePosition(c1)
            quoteTrader.closePosition(c2)
            base, quote = rebalance(totalEquity=baseTrader.getEquity()+quoteTrader.getEquity(), baseEquity=baseEquity, quoteEquity=quoteEquity)
            baseTrader.setEquity(base)
            quoteTrader.setEquity(quote)
            OverallPosition = None 
        elif s < mean and OverallPosition == Positions.SHORT:
            baseTrader.closePosition(c1)
            quoteTrader.closePosition(c2)
            base, quote = rebalance(totalEquity=baseTrader.getEquity()+quoteTrader.getEquity(), baseEquity=baseEquity, quoteEquity=quoteEquity)
            baseTrader.setEquity(base)
            quoteTrader.setEquity(quote)
            OverallPosition = None

        else:
            baseTrader.plotPriceChange(c1)
            quoteTrader.plotPriceChange(c2)

    totalEquity = baseTrader.getEquity() + quoteTrader.getEquity()
    originalEquity = baseEquity + quoteEquity
    equityChange = (totalEquity - originalEquity) / originalEquity
    baseHold = (coins[list(coins.keys())[0]][-1] - coins[list(coins.keys())[0]][0]) / coins[list(coins.keys())[0]][0]
    quoteHold = (coins[list(coins.keys())[1]][-1] - coins[list(coins.keys())[1]][0]) / coins[list(coins.keys())[1]][0]
    print(f"\n{list(coins.keys())[0]}/{list(coins.keys())[1]} (Strategy): {'+' if equityChange > 0 else ''}{(equityChange * 100):.2f}%")
    print(f"${originalEquity:.2f}  ->  ${totalEquity:.2f}")
    print(f"{list(coins.keys())[0]} (Hold): {'+' if baseHold > 0 else ''}{(baseHold * 100):.2f}%")
    print(f"{list(coins.keys())[1]} (Hold): {'+' if quoteHold > 0 else ''}{(quoteHold * 100):.2f}%")
    wins = 0
    for base_PnL, quote_PnL in zip(baseTrader.getPnlArray(), quoteTrader.getPnlArray()):
        if base_PnL + quote_PnL > 0:
            wins += 1
    print(f"Win rate: {round((wins / baseTrader.getNumTrades()) * 100, 2)}% ({wins}/{baseTrader.getNumTrades()})")
    currentPnlPercent = round(((baseTrader.getCurrentPnl() + quoteTrader.getCurrentPnl()) / totalEquity) * 100, 2)
    print(f"Open position: {baseTrader.checkPosition()} " 
          f"{'(' + ('+' if currentPnlPercent > 0 else '') + str(currentPnlPercent) + '%)' if baseTrader.checkPosition() else ''}")

    totalCurve = []
    for c1, c2 in zip(baseTrader.getEquityCurve(), quoteTrader.getEquityCurve()):
        totalCurve.append(c1 + c2)

    if graph:
        plt.figure()
        plt.plot(np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D")), totalCurve)
        plt.plot(np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D")), baseTrader.getEquityCurve())
        plt.plot(np.arange(np.datetime64(start), np.datetime64(end), np.timedelta64(1, "D")), quoteTrader.getEquityCurve())
        plt.legend(["Total"] + list(coins.keys()), loc=0, fontsize="small")
        plt.title(f"{list(coins.keys())[0]}/{list(coins.keys())[1]}")
        plt.xlabel("Time")
        plt.ylabel("Equity ($)")
        plt.show()

    return totalCurve

def runBacktestAll(threshold: float, constant: float, graph: bool, leverage: int, start: dt = dt(2021, 1, 1), end: dt = dt.today()):
    for pair in generateBestPairs(crypto=allCrypto, threshold=threshold, start=start, end=end):
        cryptoDict = getPrices(crypto=pair, progress=False, start=start, end=end)
        cointCoeff = getCointCoeff(coinPrices=cryptoDict[list(cryptoDict.keys())[0]], coin2Prices=cryptoDict[list(cryptoDict.keys())[1]], verbose=False)
        base, quote = returnEntry(coins=cryptoDict, coeff=cointCoeff)
        backtest(coins=cryptoDict, coeff=cointCoeff, baseEquity=base, quoteEquity=quote, constant=constant, graph=graph, leverage=leverage, start=start, end=end)

if __name__ == "__main__":
    runBacktestAll(threshold=0.8, constant=1.5, graph=False, leverage=1)
