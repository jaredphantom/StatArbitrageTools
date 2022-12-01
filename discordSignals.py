from contextlib import redirect_stdout
import statarb, backtest, portfolio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta as td
from io import BytesIO, StringIO
from discord import Webhook, RequestsWebhookAdapter, File
from typing import Dict

webhook = Webhook.from_url(
    "",
    adapter=RequestsWebhookAdapter())

def sendBacktestResults(coins: Dict[str, np.ndarray], coeff: pd.Series, baseEquity: float, quoteEquity: float, 
                        constant: float = 1.5, graph: bool = False, start: dt = dt(2021, 1, 1), end: dt = dt.today()):

    f = StringIO()
    with redirect_stdout(f):
        backtest.backtest(coins=coins, coeff=coeff, baseEquity=baseEquity,
                          quoteEquity=quoteEquity, constant=constant, graph=graph, start=start, end=end)

    webhook.send(f.getvalue())

def sendSpreadGraph(coinPrices: list[float], coin2Prices: list[float], coeff: pd.Series, title: str,
                    start: dt = dt(2021, 1, 1), end: dt = dt.today()):

    statarb.drawSpreadGraph(coinPrices=coinPrices, coin2Prices=coin2Prices, coeff=coeff, title=title, 
                            start=start, end=end)

    with BytesIO() as buf:
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        file = File(buf, filename=f"{title}.png")
        webhook.send(file=file)
        plt.close()


def generateSignals(start: dt = dt(2021, 1, 1), end: dt = dt.today()):

    corr = portfolio.generateCorrMatrix(coins=statarb.binanceCrypto, threshold=0.9, start=start, end=end)
    goodPairs = portfolio.generatePortfolio(corr=corr, size=25)

    for pair in goodPairs:
        testCrypto = pair
        cryptoDict = statarb.getPrices(crypto=testCrypto, progress=False, start=start, end=end)
        cointCoeff = statarb.getCointCoeff(coinPrices=cryptoDict[testCrypto[0]], coin2Prices=cryptoDict[testCrypto[1]], verbose=False)
        base, quote = statarb.returnEntry(coins=cryptoDict, coeff=cointCoeff)

        spreadGraph = []
        for c1, c2 in zip(cryptoDict[list(cryptoDict.keys())[0]], cryptoDict[list(cryptoDict.keys())[1]]):
            spreadGraph.append(c1 + (cointCoeff["x1"] * c2))
        mean = np.mean(spreadGraph)
        stddev = np.std(spreadGraph)

        if spreadGraph[-1] < mean - (1.5 * stddev):
            webhook.send(f"\n**{list(cryptoDict.keys())[0]}**: **+**${base:.2f}   |   **{list(cryptoDict.keys())[1]}**: **-**${quote:.2f}")
            sendBacktestResults(cryptoDict, cointCoeff, base, quote, start=start, end=end)
            sendSpreadGraph(coinPrices=cryptoDict[testCrypto[0]], coin2Prices=cryptoDict[testCrypto[1]], coeff=cointCoeff,
                            title=f"{testCrypto[0]} - {abs(cointCoeff['x1']):.3f} {testCrypto[1]}", start=start, end=end)

        elif spreadGraph[-1] > mean + (1.5 * stddev):
            webhook.send(f"\n**{list(cryptoDict.keys())[0]}**: **-**${base:.2f}   |   **{list(cryptoDict.keys())[1]}**: **+**${quote:.2f}")
            sendBacktestResults(cryptoDict, cointCoeff, base, quote, start=start, end=end)
            sendSpreadGraph(coinPrices=cryptoDict[testCrypto[0]], coin2Prices=cryptoDict[testCrypto[1]], coeff=cointCoeff,
                            title=f"{testCrypto[0]} - {abs(cointCoeff['x1']):.3f} {testCrypto[1]}", start=start, end=end)

if __name__ == "__main__":
    try:
        generateSignals(start=dt(2022, 1, 1))
    except ValueError:
        generateSignals(start=dt(2022, 1, 1), end=(dt.today() - td(days=1)))
