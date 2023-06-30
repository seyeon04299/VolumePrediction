from src.constants import *
from enum import Enum

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

nasdaq_test_stocks = set(["ZVZZT", "ZWZZT"])    # these are nasdaq test stocks

class MessageType(Enum):
    ADD = 0
    DEL = 1
    CANCEL = 2
    REPLACE = 3
    EXEC = 4
    EXEC_PRICE = 5
    CROSS_TRADE = 6
    NON_CROSS_TRADE = 7


class Exchange(Enum):
    NYSE = 0
    NASDAQ = 1

