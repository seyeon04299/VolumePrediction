"""
constants.py

this file contains constants that we will be using often
"""

_min_to_ns = lambda x: int(x * 60 * 1e9)
_sec_to_ns = lambda x: int(x * 1e9)

# interval in nanoseconds
INTERVAL_SEC = 1
INTERVAL_NS = _sec_to_ns(INTERVAL_SEC)

# 4:00 in nanoseconds
START_TIME_MIN = 4 * 60 # 4:00
START_TIME_NS = _min_to_ns(START_TIME_MIN)

# 20:00 in nanoseconds
END_TIME_MIN = 20 * 60 + 1  # 20:00 or 8pm, we add 1 minute because
                            # there are messages that occur after 20:00 in 
                            # NYSE TAQ data 
END_TIME_NS = _min_to_ns(END_TIME_MIN)

# 9:30 (i.e. regular market start time) in nanoseconds
REGMKT_START_TIME_MIN = 9 * 60 + 30 # 9:30am
REGMKT_START_TIME_NS = _min_to_ns(REGMKT_START_TIME_MIN)

# 16:00 in nanoseconds
REGMKT_END_TIME_MIN = 16 * 60 # 16:00 or 4pm
REGMKT_END_TIME_NS = _min_to_ns(REGMKT_END_TIME_MIN)

# time in nanoseconds to hours
NANOSEC_TO_HOUR = 2.7777777777778e-13

# Number of intervals(or bins)
T_N = (END_TIME_NS - START_TIME_NS - 1) // INTERVAL_NS + 1
R_N = (REGMKT_START_TIME_NS - START_TIME_NS-1) // INTERVAL_NS +1   # start of regular market hours index
A_N = (REGMKT_END_TIME_NS - START_TIME_NS -1) // INTERVAL_NS +1      # start of afterhours market index

# Get ticker symbol and date from itch filename
TICKER_SYM = lambda x: x.split('.')[-3].split('/')[-1]
DATE_FROM_PATH = lambda x: x.split('/')[2]
YEAR_FROM_PATH = lambda x: (DATE_FROM_PATH(x))[-2:]
# sell and buy constants
ASK = 1
BID = 2

# nyse parameters
NYSE_MAX_CHANNEL = 11
