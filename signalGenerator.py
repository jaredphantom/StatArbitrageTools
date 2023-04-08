from datetime import datetime as dt
from positionManager import PositionManager
from discordSignals import webhook
import numpy as np
import statarb

class SignalGenerator:
    
    def __init__(self, portfolio: list[tuple[str]], positionManager: PositionManager, start: dt = dt(2022, 1, 1), maxEntry: float = 50) -> None:
        self._portfolio = portfolio
        self._positionManager = positionManager
        self._start = start
        self._maxEntry = maxEntry

    def generateSellSignals(self) -> list[tuple[str]]:
        return self._positionManager.checkSellSignals()
    
    def generateBuySignals(self) -> None:
        positions = []
        for position in self._positionManager.getPositions():
            positions.append(position.getAssets())

        for pair in self._portfolio:
            if pair in positions or pair[::-1] in positions:
                continue

            cryptoDict = statarb.getPrices(crypto=pair, progress=False, start=self._start)
            cointCoeff = statarb.getCointCoeff(coinPrices=cryptoDict[pair[0]], coin2Prices=cryptoDict[pair[1]], verbose=False)
            pval = statarb.getCointSummary(coinPrices=cryptoDict[pair[0]], coin2Prices=cryptoDict[pair[1]])[1]
            base, quote = statarb.returnEntry(coins=cryptoDict, coeff=cointCoeff, maxEntry=self._maxEntry)

            spreadGraph = []
            for c1, c2 in zip(cryptoDict[list(cryptoDict.keys())[0]], cryptoDict[list(cryptoDict.keys())[1]]):
                spreadGraph.append(c1 + (cointCoeff["x1"] * c2))
            mean = np.mean(spreadGraph)
            stddev = np.std(spreadGraph)

            if pval <= 0.005:
                if spreadGraph[-1] < mean - (1 * stddev):
                    if spreadGraph[-1] < mean - (1.5 * stddev):
                        if spreadGraph[-1] < mean - (2 * stddev):
                            webhook.send(f"**Buy Signal <!!!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                            webhook.send(f"{list(cryptoDict.keys())[0]}: LONG ${base:.2f}\n{list(cryptoDict.keys())[1]}: SHORT ${quote:.2f}")
                        else:
                            webhook.send(f"**Buy Signal <!!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                            webhook.send(f"{list(cryptoDict.keys())[0]}: LONG ${base:.2f}\n{list(cryptoDict.keys())[1]}: SHORT ${quote:.2f}")
                    else:
                        webhook.send(f"**Buy Signal <!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                        webhook.send(f"{list(cryptoDict.keys())[0]}: LONG ${base:.2f}\n{list(cryptoDict.keys())[1]}: SHORT ${quote:.2f}")

                elif spreadGraph[-1] > mean + (1 * stddev):
                    if spreadGraph[-1] > mean + (1.5 * stddev):
                        if spreadGraph[-1] > mean + (2 * stddev):
                            webhook.send(f"**Buy Signal <!!!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                            webhook.send(f"{list(cryptoDict.keys())[0]}: SHORT ${base:.2f}\n{list(cryptoDict.keys())[1]}: LONG ${quote:.2f}")
                        else:
                            webhook.send(f"**Buy Signal <!!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                            webhook.send(f"{list(cryptoDict.keys())[0]}: SHORT ${base:.2f}\n{list(cryptoDict.keys())[1]}: LONG ${quote:.2f}")
                    else:
                        webhook.send(f"**Buy Signal <!>: {list(cryptoDict.keys())[0]}/{list(cryptoDict.keys())[1]}**")
                        webhook.send(f"{list(cryptoDict.keys())[0]}: SHORT ${base:.2f}\n{list(cryptoDict.keys())[1]}: LONG ${quote:.2f}")
            
