from dydx3 import Client
from dydx3.constants import *
from dydx3.errors import *
from dotenv import dotenv_values
from backtest import Positions
from positionManager import PositionManager, Position, webhook
from signalGenerator import SignalGenerator
from positionManagerHandler import manager
from datetime import datetime as dt
from requests.exceptions import ConnectionError
from statarb import dydxCrypto
from portfolio import generateCorrMatrix, generatePortfolio
import logging, time

class DydxClient(Client):

    def __init__(self, depositedEquity: float, signalGenerator: SignalGenerator, positionManager: PositionManager, date: dt) -> None:
        super().__init__(host="https://api.dydx.exchange", **self.getEnv())
        self._depositedEquity = depositedEquity
        self._signalGenerator = signalGenerator
        self._positionManager = positionManager
        self._date = date
        self._init = False

    def getEnv(self) -> dict[str, str]:
        env = dotenv_values(".env")
        envDict = {
            "api_key_credentials": {
                "key": env["API_KEY"],
                "secret": env["SECRET_KEY"],
                "passphrase": env["PASSPHRASE"]
            },
            "stark_private_key": env["STARK_KEY"],
            "default_ethereum_address": env["WALLET_ADDRESS"]
        }

        return envDict
    
    def initPositions(self) -> None:
        positions = self.getPositions()
        for position in positions:
            assets = tuple(position[0].split("/"))
            if self._positionManager.checkPosition(*assets) != -1:
                if not self._init:
                    logging.info(f"{assets} - Position open")
            else:
                newPosition = Position(*assets, self._date, position[1])
                self._positionManager.addPosition(newPosition)
                logging.info(f"Added new position: {newPosition.getAssets()}")

        if len(positions) != len(self._positionManager.getPositions()):
            logging.warning("Position mismatch between Dydx account and local PositionManager")

        self._init = True

    def run(self) -> None:
        logging.info("Connected to exchange")
        
        while True:
            self.initPositions()
            if dt.today().hour == 8 or dt.today().hour == 20:
                self._signalGenerator.generateBuySignals()

            closed = self._signalGenerator.generateSellSignals()
            for pos in closed:
                self.closePosition(pos[0])
                self.closePosition(pos[1])

            logging.info(f"Portfolio PnL: {round(self.getProfitPercentage(), 2)}%")
            time.sleep(hour / 2)

    def closePosition(self, asset: str) -> None:
        market = f"{asset}-USD"
        position = self.private.get_positions(market=market, status=POSITION_STATUS_OPEN).data["positions"][0]
        side = position["side"]
        opposingSide = ORDER_SIDE_SELL if side == "LONG" else ORDER_SIDE_BUY
        size = position["size"].replace("-", "")
        price = str(
                    round(
                        float(self.public.get_stats(market=market).data["markets"][market]["close"]) * (1.01 if opposingSide == ORDER_SIDE_BUY else 0.99),
                        self.getTickSize(market=market)
                    )
                )

        try:
            order = self.private.create_order(
                position_id=int(self.getAccount()["positionId"]),
                market=market,
                side=opposingSide,
                order_type=ORDER_TYPE_MARKET,
                post_only=False,
                size=size,
                price=price,
                limit_fee="0.015",
                time_in_force=TIME_IN_FORCE_IOC,
                reduce_only=True,
                expiration_epoch_seconds=int(dt.now().timestamp()) + (2 * minute)
            )

            logging.info(order.data)

            realizedPnl = self.getRealizedPnl(market=market)
            webhook.send(f"{asset} position closed: {'+' if realizedPnl >= 0 else '-'}${abs(realizedPnl)}")

        except DydxApiError as e:
            logging.error(f"{e.status_code}: {e.msg}")

    def getAccount(self) -> dict:
        return self.private.get_account().data["account"]
    
    def getTickSize(self, market: str) -> int:
        tickSize = self.public.get_markets(market=market).data["markets"][market]["tickSize"]

        if tickSize.find(".") == -1:
            return 0
        if tickSize.find("1") - tickSize.find(".") == -1:
            return 0
        return tickSize.find("1") - tickSize.find(".")
    
    def getEquity(self) -> float:
        return float(self.getAccount()["equity"])
    
    def getProfitPercentage(self) -> float:
        currentEquity = self.getEquity()
        return ((currentEquity - self._depositedEquity) / self._depositedEquity) * 100
    
    def getRealizedPnl(self, market: str) -> float:
        return round(float(self.private.get_positions(market=market, status=POSITION_STATUS_CLOSED).data["positions"][0]["realizedPnl"]), 2)
    
    def getPositions(self) -> list:
        positions = []
        pair = []
        openPositions = self.getAccount()["openPositions"]

        for position in openPositions:
            if len(pair) == 0:
                side = Positions.LONG if openPositions[position]["side"] == "LONG" else Positions.SHORT
            if len(pair) < 2:
                pair.append(position)
            if len(pair) == 2:
                pair = [f"{pair[0].split('-')[0]}/{pair[1].split('-')[0]}"]
                pair.append(side)
                positions.append(pair)
                pair = []

        return positions

if __name__ == "__main__":
    logging.basicConfig(filename="logs/dydx_client.log",
                        filemode="a",
                        format="%(asctime)s | %(levelname)s - %(message)s",
                        datefmt="%d %b %H:%M:%S",
                        level=logging.INFO)

    minute = 60
    hour = minute * 60
    startDate = dt(2022, 1, 1)

    while True:
        try:
            corr = generateCorrMatrix(coins=dydxCrypto, threshold=0.75, start=startDate)
            portfolio = generatePortfolio(corr=corr, size=25)
            break
        except ValueError as e:
            logging.error(e)
            time.sleep(hour)

    signalGenerator = SignalGenerator(portfolio=portfolio, positionManager=manager, start=startDate)
    client = DydxClient(depositedEquity=200, signalGenerator=signalGenerator, positionManager=manager, date=startDate)

    while True:
        try:
            client.run()
        except ValueError as e:
            logging.error(e)
            time.sleep(hour)
        except ConnectionError:
            logging.warning("Disconnected from exchange")
        except DydxError as e:
            logging.error(e)
        finally:
            time.sleep(minute)
