from backtest import Positions, generateSpreadGraph
from statarb import getCointCoeff, getPrices, getCointSummary
from discordSignals import webhook
from datetime import datetime as dt
import numpy as np
import logging

class Position:

    def __init__(self, asset1: str, asset2: str, date: dt, position: Positions) -> None:
        self._asset1 = asset1
        self._asset2 = asset2
        self._date = date
        self._position = position

    def checkSellSignal(self) -> bool:
        prices = getPrices(crypto=[self._asset1, self._asset2], start=self._date, progress=False)
        coeff = getCointCoeff(coinPrices=prices[self._asset1], coin2Prices=prices[self._asset2], verbose=False)
        spreadGraph = generateSpreadGraph(coinPrices=prices[self._asset1], coin2Prices=prices[self._asset2], coeff=coeff)
        mean = np.mean(spreadGraph)
        s = spreadGraph[-1]

        if s >= mean and self._position == Positions.LONG:
            return True
        elif s <= mean and self._position == Positions.SHORT:
            return True
        
        cointSummary = getCointSummary(coinPrices=prices[self._asset1], coin2Prices=prices[self._asset2])
        pval = cointSummary[1]

        if pval > 0.15:
            return True

        return False

    def getAssets(self) -> tuple[str]:
        return (self._asset1, self._asset2)

class PositionManager:

    def __init__(self, *positions: Position) -> None:
        self._positions = list(positions)

    def addPosition(self, position: Position) -> None:
        if self.checkPosition(*position.getAssets()) == -1:
            self._positions.append(position)

    def checkPosition(self, asset1: str, asset2: str) -> int:
        for index, position in enumerate(self._positions):
            assets = position.getAssets()
            if assets == (asset1, asset2) or assets == (asset2, asset1):
                return index
        
        return -1

    def removePosition(self, asset1: str, asset2: str) -> None:
        index = self.checkPosition(asset1, asset2)
        if index != -1:
            self._positions.pop(index)
            logging.info(f"Removed position: {(asset1, asset2)}")

    def checkSellSignals(self) -> list[tuple[str]]:
        removed = []
        for position in self._positions:
            isPositionClosed = position.checkSellSignal()
            if isPositionClosed:
                assets = position.getAssets()
                logging.info(f"Sell signal: {assets}")
                webhook.send(f"**Sell Signal: {'/'.join(assets)}**")
                self.removePosition(*assets)
                removed.append(assets)

        return removed        

    def getPositions(self) -> list[Position]:
        return self._positions
            
