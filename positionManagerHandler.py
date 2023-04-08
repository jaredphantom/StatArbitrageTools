from datetime import datetime as dt
from positionManager import Position, PositionManager
from backtest import Positions

pos1 = Position("ALGO", "LINK", date=dt(2022, 1, 1), position=Positions.LONG)

manager = PositionManager(pos1)
